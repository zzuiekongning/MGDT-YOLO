# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
import torch.nn.functional as F

from mmcv.cnn import ConvModule, Scale
from mmengine.model import normal_init
# from mmcv.runner import force_fp32
# from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
#                         images_to_levels, multi_apply, multiclass_nms,
#                         reduce_mean, unmap, distance2bbox)
# from mmcv.ops import deform_conv2d

from .block import DFL, Proto, DyDCNv2
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_
# from ..models.dense_heads.anchor_head import AnchorHead
# from ..models.builder import build_loss
# from ..core.bbox.transforms import distance2bbox

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors


__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder', 'TOODHead'

EPS = 1e-12

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def bbox_limited(bboxes, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)

class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))
        
class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8, conv_cfg=None, norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.la_conv1 = nn.Conv2d( self.in_channels,  self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d( self.in_channels // la_down_rate,  self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.norm_cfg is None)

    def init_weights(self):
        normal_init(self.la_conv1, std=0.001)
        normal_init(self.la_conv2, std=0.001)
        self.la_conv2.bias.data.zero_()
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
                          self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat

class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 4  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        # print("===================1")
        # print(self.nl)
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            # print("ajfoajgoajgoajgoajg")
            # print(x[0].shape)            
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        from ultralytics.vit.utils.ops import get_cdn_group

        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        if not self.training:
            dec_scores = dec_scores.sigmoid_()
        return dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = len(feats)
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(torch.where(valid_mask, feats, 0))  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)
        # dynamic anchors + static content
        enc_outputs_bboxes = self.enc_bbox_head(features) + anchors  # (bs, h*w, 4)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # Unsigmoided
        refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # refer_bbox = torch.gather(enc_outputs_bboxes, 1, topk_ind.reshape(bs, self.num_queries).unsqueeze(-1).repeat(1, 1, 4))

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        if self.training:
            refer_bbox = refer_bbox.detach()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

class TOODHead(nn.Module):
    # Task Dynamic Align Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc, hidc, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc // 2, 3), Conv_GN(hidc // 2, hidc // 2, 3))
        # self.share_conv = nn.Sequential(DyDCNv2(hidc, hidc // 2, 3), DyDCNv2(hidc // 2, hidc // 2, 3))
        self.cls_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.reg_decomp = TaskDecomposition(hidc // 2, 2, 16)
        self.DyDCNV2 = DyDCNv2(hidc // 2, hidc // 2)
        self.spatial_conv_offset = nn.Conv2d(hidc, 3 * 3 * 3, 3, padding=1)
        self.offset_dim = 2 * 3 * 3
        self.cls_prob_conv1 = nn.Conv2d(hidc, hidc // 4, 1)
        self.cls_prob_conv2 = nn.Conv2d(hidc // 4, 1, 3, padding=1)
        self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc // 2, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        # print("===================1")
        # print(self.nl)
        # print(self.temp)
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            stack_res_list = [self.share_conv[0](x[i])]
            stack_res_list.extend(m(stack_res_list[-1]) for m in self.share_conv[1:])
            feat = torch.cat(stack_res_list, dim=1)
            
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)
            # print("===================2")

            # reg alignment
            offset_and_mask = self.spatial_conv_offset(feat)
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            reg_feat = self.DyDCNV2(reg_feat, offset, mask)
            
            # cls alignment
            cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat))).sigmoid()
            # mid_1 = self.scale[i](self.cv2(reg_feat))
            # mid_2 = self.cv3(cls_feat * cls_prob)
            # print("===================3")
            # print(mid_2.shape)
            # print(mid_1.shape)
            # x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(cls_feat * cls_prob)), 1)
            x[i] = torch.cat((self.cv2(F.relu(reg_feat)), self.cv3(cls_feat * cls_prob)), 1)

        if self.training:  # Training path
            # print("ajfoajgoajgoajgoajg")
            # print(x[0].shape)
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

# class TOODHead_origin(AnchorHead):
#     """TOOD: Task-aligned One-stage Object Detection.

#     TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
#     Learning (TAL).

#     todo: list link of the paper.
#     """
#     def __init__(self,
#                  num_classes,
#                  in_channels,
#                  stacked_convs=4,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
#                  num_dcn_on_head=0,
#                  anchor_type='anchor_free',
#                  initial_loss_cls=dict(
#                      type='TaskAlignedLoess',
#                      use_sigmoid=True,
#                      gamma=2.0,
#                      alpha=0.25,
#                      loss_weight=1.0),
#                  **kwargs):
#         self.stacked_convs = stacked_convs
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.num_dcn_on_head = num_dcn_on_head
#         self.anchor_type = anchor_type
#         self.epoch = 0 # which would be update in head hook!
#         super(TOODHead, self).__init__(num_classes, in_channels, **kwargs)

#         self.sampling = False
#         if self.train_cfg:
#             self.initial_epoch = self.train_cfg.initial_epoch
#             self.initial_assigner = build_assigner(self.train_cfg.initial_assigner)
#             self.initial_loss_cls = build_loss(initial_loss_cls)
#             self.alingment_assigner = build_assigner(self.train_cfg.assigner)
#             self.alpha = self.train_cfg.alpha
#             self.beta = self.train_cfg.beta
#             # SSD sampling=False so use PseudoSampler
#             sampler_cfg = dict(type='PseudoSampler')
#             self.sampler = build_sampler(sampler_cfg, context=self)

#     def _init_layers(self):
#         """Initialize layers of the head."""
#         self.relu = nn.ReLU(inplace=True)
#         self.inter_convs = nn.ModuleList()
#         print("jgojsgojsogj")
#         print(self.stacked_convs)
#         for i in range(self.stacked_convs):
#             if i < self.num_dcn_on_head:
#                 conv_cfg = dict(type='DCNv2', deform_groups=4)
#             else:
#                 conv_cfg = self.conv_cfg
#             chn = self.in_channels if i == 0 else self.feat_channels
#             print("sjgosjgosjgsobmsobj")
#             self.inter_convs.append(
#                 ConvModule(
#                     chn,
#                     self.feat_channels,
#                     3,
#                     stride=1,
#                     padding=1,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=self.norm_cfg))
#         print("jotqjtoqugabnaog")
#         self.cls_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8, self.conv_cfg, self.norm_cfg)
#         self.reg_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8, self.conv_cfg, self.norm_cfg)

#         self.tood_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
#         self.tood_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)

#         self.cls_prob_conv1 = nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1)
#         self.cls_prob_conv2 = nn.Conv2d(self.feat_channels // 4, 1, 3, padding=1)
#         self.reg_offset_conv1 = nn.Conv2d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1)
#         self.reg_offset_conv2 = nn.Conv2d(self.feat_channels // 4, 4 * 2, 3, padding=1)

#         self.scales = nn.ModuleList([Scale(1.0) for _ in self.anchor_generator.strides])

#     def init_weights(self):
#         """Initialize weights of the head."""
#         for m in self.inter_convs:
#             normal_init(m.conv, std=0.01)

#         self.cls_decomp.init_weights()
#         self.reg_decomp.init_weights()

#         bias_cls = bias_init_with_prob(0.01)
#         normal_init(self.tood_cls, std=0.01, bias=bias_cls)
#         normal_init(self.tood_reg, std=0.01)

#         normal_init(self.cls_prob_conv1, std=0.01)
#         bias_cls = bias_init_with_prob(0.01)
#         normal_init(self.cls_prob_conv2, std=0.01, bias=bias_cls)
#         normal_init(self.reg_offset_conv1, std=0.001)
#         normal_init(self.reg_offset_conv2, std=0.001)
#         self.reg_offset_conv2.bias.data.zero_()

#     def forward(self, feats):
#         """Forward features from the upstream network.

#         Args:
#             feats (tuple[Tensor]): Features from the upstream network, each is
#                 a 4D-tensor.

#         Returns:
#             tuple: Usually a tuple of classification scores and bbox prediction
#                 cls_scores (list[Tensor]): Classification scores for all scale
#                     levels, each is a 4D-tensor, the channels number is
#                     num_anchors * num_classes.
#                 bbox_preds (list[Tensor]): Box energies / deltas for all scale
#                     levels, each is a 4D-tensor, the channels number is
#                     num_anchors * 4.
#         """
#         num_imgs = len(feats[0])
#         featmap_sizes = [featmap.size()[-2:] for featmap in feats]
#         device = feats[0].device
#         anchor_list = self.get_anchor_list(
#             featmap_sizes, num_imgs, device=device)
#         level_anchor_list = [torch.cat([anchor_list[i][j] for i in range(len(anchor_list))]) for j in range(len(anchor_list[0]))]

#         return multi_apply(self.forward_single, feats, self.scales, level_anchor_list, self.anchor_generator.strides)


#     def forward_single(self, x, scale, anchor, stride):
#         """Forward feature of a single scale level.

#         Args:
#             x (Tensor): Features of a single scale level.
#             scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
#                 the bbox prediction.
#             anchor (Tensor): Anchors of a single scale level.
#             stride (tuple[Tensor]): Stride of the current scale level.

#         Returns:
#             tuple:
#                 cls_score (Tensor): Cls scores for a single scale level
#                     the channels number is num_anchors * num_classes.
#                 bbox_pred (Tensor): Box energies / deltas for a single scale
#                     level, the channels number is num_anchors * 4.
#         """
#         b, c, h, w = x.shape

#         # extract task interactive features
#         inter_feats = []
#         for i, inter_conv in enumerate(self.inter_convs):
#             x = inter_conv(x)
#             inter_feats.append(x)
#         feat = torch.cat(inter_feats, 1)

#         # task decomposition
#         avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
#         cls_feat = self.cls_decomp(feat, avg_feat)
#         reg_feat = self.reg_decomp(feat, avg_feat)

#         # cls prediction and alignment
#         cls_logits = self.tood_cls(cls_feat)
#         cls_prob = F.relu(self.cls_prob_conv1(feat))
#         cls_prob = self.cls_prob_conv2(cls_prob)
#         cls_score = (cls_logits.sigmoid() * cls_prob.sigmoid()).sqrt()

#         # reg prediction and alignment
#         if self.anchor_type == 'anchor_free':
#             reg_dist = scale(self.tood_reg(reg_feat).exp()).float()
#             reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
#             reg_bbox = distance2bbox(self.anchor_center(anchor) / stride[0], reg_dist).reshape(b, h, w, 4).permute(0, 3, 1, 2)  # (b, c, h, w)
#         elif self.anchor_type == 'anchor_based':
#             reg_dist = scale(self.tood_reg(reg_feat)).float()
#             reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
#             reg_bbox = self.bbox_coder.decode(anchor, reg_dist).reshape(b, h, w, 4).permute(0, 3, 1, 2) / stride[0]
#         else:
#             raise NotImplementedError
#         reg_offset = F.relu(self.reg_offset_conv1(feat))
#         reg_offset = self.reg_offset_conv2(reg_offset)
#         bbox_pred = self.deform_sampling(reg_bbox.contiguous(), reg_offset.contiguous())

#         return cls_score, bbox_pred

#     def get_anchor_list(self, featmap_sizes, num_imgs, device='cuda'):
#         """Get anchors according to feature map sizes.

#         Args:
#             featmap_sizes (list[tuple]): Multi-level feature map sizes.
#             num_imgs (int): the number of images in a batch
#             device (torch.device | str): Device for returned tensors

#         Returns:
#             anchor_list (list[Tensor]): Anchors of each image.
#         """
#         # since feature map sizes of all images are the same, we only compute
#         # anchors for one time
#         multi_level_anchors = self.anchor_generator.grid_anchors(
#             featmap_sizes, device)
#         anchor_list = [multi_level_anchors for _ in range(num_imgs)]

#         return anchor_list

#     def deform_sampling(self, feat, offset):
#         """ Sampling the feature x according to offset.

#         Args:
#             feat (Tensor): Feature
#             offset (Tensor): Spatial offset for for feature sampliing
#         """
#         # it is an equivalent implementation of bilinear interpolation
#         b, c, h, w = feat.shape
#         weight = feat.new_ones(c, 1, 1, 1)
#         y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
#         return y

#     def anchor_center(self, anchors):
#         """Get anchor centers from anchors.

#         Args:
#             anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

#         Returns:
#             Tensor: Anchor centers with shape (N, 2), "xy" format.
#         """
#         anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
#         anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
#         return torch.stack([anchors_cx, anchors_cy], dim=-1)

#     def loss_single(self, anchors, cls_score, bbox_pred, labels,
#                     label_weights, bbox_targets, alignment_metrics, stride, num_total_samples):
#         """Compute loss of a single scale level.

#         Args:
#             anchors (Tensor): Box reference for each scale level with shape
#                 (N, num_total_anchors, 4).
#             cls_score (Tensor): Box scores for each scale level
#                 Has shape (N, num_anchors * num_classes, H, W).
#             bbox_pred (Tensor): Box energies / deltas for each scale
#                 level with shape (N, num_anchors * 4, H, W).
#             labels (Tensor): Labels of each anchors with shape
#                 (N, num_total_anchors).
#             label_weights (Tensor): Label weights of each anchor with shape
#                 (N, num_total_anchors)
#             bbox_targets (Tensor): BBox regression targets of each anchor wight
#                 shape (N, num_total_anchors, 4).
#             num_total_samples (int): Number os positive samples that is
#                 reduced over all GPUs.

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         assert stride[0] == stride[1], 'h stride is not equal to w stride!'
#         anchors = anchors.reshape(-1, 4)
#         cls_score = cls_score.permute(0, 2, 3, 1).reshape(
#             -1, self.cls_out_channels).contiguous()
#         bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
#         bbox_targets = bbox_targets.reshape(-1, 4)
#         labels = labels.reshape(-1)

#         # classification loss
#         if self.epoch < self.initial_epoch:
#             label_weights = label_weights.reshape(-1)
#             loss_cls = self.initial_loss_cls(
#                 cls_score, labels, label_weights, avg_factor=1.0)
#         else:
#             alignment_metrics = alignment_metrics.reshape(-1)
#             loss_cls = self.loss_cls(
#                 cls_score, labels, alignment_metrics, avg_factor=1.0)  # num_total_samples)

#         # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
#         bg_class_ind = self.num_classes
#         pos_inds = ((labels >= 0)
#                     & (labels < bg_class_ind)).nonzero().squeeze(1)

#         if len(pos_inds) > 0:
#             pos_bbox_targets = bbox_targets[pos_inds]
#             pos_bbox_pred = bbox_pred[pos_inds]
#             pos_anchors = anchors[pos_inds]

#             pos_decode_bbox_pred = pos_bbox_pred
#             pos_decode_bbox_targets = pos_bbox_targets / stride[0]

#             # regression loss
#             if self.epoch < self.initial_epoch:
#                 pos_bbox_weight = self.centerness_target(
#                         pos_anchors, pos_bbox_targets)
#             else:
#                 pos_bbox_weight = alignment_metrics[pos_inds]
#             loss_bbox = self.loss_bbox(
#                 pos_decode_bbox_pred,
#                 pos_decode_bbox_targets,
#                 weight=pos_bbox_weight,
#                 avg_factor=1.0)
#         else:
#             loss_bbox = bbox_pred.sum() * 0
#             pos_bbox_weight = torch.tensor(0).cuda()

#         return loss_cls, loss_bbox, alignment_metrics.sum(), pos_bbox_weight.sum()

#     @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
#     def loss(self,
#              cls_scores,
#              bbox_preds,
#              gt_bboxes,
#              gt_labels,
#              img_metas,
#              gt_bboxes_ignore=None):
#         """Compute losses of the head.

#         Args:
#             cls_scores (list[Tensor]): Box scores for each scale level
#                 Has shape (N, num_anchors * num_classes, H, W)
#             bbox_preds (list[Tensor]): Box energies / deltas for each scale
#                 level with shape (N, num_anchors * 4, H, W)
#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels (list[Tensor]): class indices corresponding to each box
#             img_metas (list[dict]): Meta information of each image, e.g.,
#                 image size, scaling factor, etc.
#             gt_bboxes_ignore (list[Tensor] | None): specify which bounding
#                 boxes can be ignored when computing the loss.

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
#         assert len(featmap_sizes) == self.anchor_generator.num_levels

#         device = cls_scores[0].device
#         anchor_list, valid_flag_list = self.get_anchors(
#             featmap_sizes, img_metas, device=device)
#         label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

#         cls_reg_targets = self.get_targets(
#             cls_scores,
#             bbox_preds,
#             anchor_list,
#             valid_flag_list,
#             gt_bboxes,
#             img_metas,
#             gt_bboxes_ignore_list=gt_bboxes_ignore,
#             gt_labels_list=gt_labels,
#             label_channels=label_channels)
#         if cls_reg_targets is None:
#             return None

#         (anchor_list, labels_list, label_weights_list, bbox_targets_list,
#          bbox_weights_list, num_total_pos, num_total_neg, alignment_metrics_list) = cls_reg_targets

#         num_total_samples = reduce_mean(
#             torch.tensor(num_total_pos, dtype=torch.float,
#                          device=device)).item()
#         num_total_samples = max(num_total_samples, 1.0)

#         losses_cls, losses_bbox,\
#             cls_avg_factors, bbox_avg_factors = multi_apply(
#                 self.loss_single,
#                 anchor_list,
#                 cls_scores,
#                 bbox_preds,
#                 labels_list,
#                 label_weights_list,
#                 bbox_targets_list,
#                 alignment_metrics_list,
#                 self.anchor_generator.strides,
#                 num_total_samples=num_total_samples)

#         cls_avg_factor = sum(cls_avg_factors)
#         cls_avg_factor = reduce_mean(cls_avg_factor).item()
#         if cls_avg_factor < EPS:
#             cls_avg_factor = 1
#         losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

#         bbox_avg_factor = sum(bbox_avg_factors)
#         bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
#         if bbox_avg_factor < EPS:
#             bbox_avg_factor = 1
#         losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

#         return dict(
#             loss_cls=losses_cls,
#             loss_bbox=losses_bbox)

#     def centerness_target(self, anchors, bbox_targets):
#         # only calculate pos centerness targets, otherwise there may be nan
#         # for bbox-based
#         # gts = self.bbox_coder.decode(anchors, bbox_targets)
#         # for point-based
#         gts = bbox_targets
#         anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
#         anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
#         l_ = anchors_cx - gts[:, 0]
#         t_ = anchors_cy - gts[:, 1]
#         r_ = gts[:, 2] - anchors_cx
#         b_ = gts[:, 3] - anchors_cy

#         left_right = torch.stack([l_, r_], dim=1)
#         top_bottom = torch.stack([t_, b_], dim=1)
#         centerness = torch.sqrt(
#             (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
#             (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
#         assert not torch.isnan(centerness).any()
#         return centerness

#     @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
#     def get_bboxes(self,
#                    cls_scores,
#                    bbox_preds,
#                    img_metas,
#                    cfg=None,
#                    rescale=False,
#                    with_nms=True):
#         """Transform network output for a batch into bbox predictions.

#         Args:
#             cls_scores (list[Tensor]): Box scores for each scale level
#                 with shape (N, num_anchors * num_classes, H, W).
#             bbox_preds (list[Tensor]): Box energies / deltas for each scale
#                 level with shape (N, num_anchors * 4, H, W).
#             img_metas (list[dict]): Meta information of each image, e.g.,
#                 image size, scaling factor, etc.
#             cfg (mmcv.Config | None): Test / postprocessing configuration,
#                 if None, test_cfg would be used. Default: None.
#             rescale (bool): If True, return boxes in original image space.
#                 Default: False.
#             with_nms (bool): If True, do nms before return boxes.
#                 Default: True.

#         Returns:
#             list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
#                 The first item is an (n, 5) tensor, where the first 4 columns
#                 are bounding box positions (tl_x, tl_y, br_x, br_y) and the
#                 5-th column is a score between 0 and 1. The second item is a
#                 (n,) tensor where each item is the predicted class label of the
#                 corresponding box.
#         """
#         cfg = self.test_cfg if cfg is None else cfg
#         assert len(cls_scores) == len(bbox_preds)
#         num_levels = len(cls_scores)

#         result_list = []
#         for img_id in range(len(img_metas)):
#             cls_score_list = [
#                 cls_scores[i][img_id].detach() for i in range(num_levels)
#             ]
#             bbox_pred_list = [
#                 bbox_preds[i][img_id].detach() for i in range(num_levels)
#             ]
#             img_shape = img_metas[img_id]['img_shape']
#             scale_factor = img_metas[img_id]['scale_factor']
#             proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
#                                                 img_shape, scale_factor,
#                                                 cfg, rescale, with_nms)
#             result_list.append(proposals)
#         return result_list

#     def _get_bboxes_single(self,
#                            cls_scores,
#                            bbox_preds,
#                            img_shape,
#                            scale_factor,
#                            cfg,
#                            rescale=False,
#                            with_nms=True):
#         """Transform outputs for a single batch item into labeled boxes.

#         Args:
#             cls_scores (list[Tensor]): Box scores for a single scale level
#                 with shape (num_anchors * num_classes, H, W).
#             bbox_preds (list[Tensor]): Box energies / deltas for a single
#                 scale level with shape (num_anchors * 4, H, W).
#             mlvl_anchors (list[Tensor]): Box reference for a single scale level
#                 with shape (num_total_anchors, 4).
#             img_shape (tuple[int]): Shape of the input image,
#                 (height, width, 3).
#             scale_factor (ndarray): Scale factor of the image arrange as
#                 (w_scale, h_scale, w_scale, h_scale).
#             cfg (mmcv.Config | None): Test / postprocessing configuration,
#                 if None, test_cfg would be used.
#             rescale (bool): If True, return boxes in original image space.
#                 Default: False.
#             with_nms (bool): If True, do nms before return boxes.
#                 Default: True.

#         Returns:
#             tuple(Tensor):
#                 det_bboxes (Tensor): BBox predictions in shape (n, 5), where
#                     the first 4 columns are bounding box positions
#                     (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
#                     between 0 and 1.
#                 det_labels (Tensor): A (n,) tensor where each item is the
#                     predicted class label of the corresponding box.
#         """
#         assert len(cls_scores) == len(bbox_preds)
#         mlvl_bboxes = []
#         mlvl_scores = []
#         for cls_score, bbox_pred, stride in zip(
#                 cls_scores, bbox_preds, self.anchor_generator.strides):
#             assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
#             assert stride[0] == stride[1]

#             scores = cls_score.permute(1, 2, 0).reshape(
#                 -1, self.cls_out_channels)
#             bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]

#             nms_pre = cfg.get('nms_pre', -1)
#             if nms_pre > 0 and scores.shape[0] > nms_pre:
#                 max_scores, _ = scores.max(dim=1)
#                 _, topk_inds = max_scores.topk(nms_pre)
#                 bbox_pred = bbox_pred[topk_inds, :]
#                 scores = scores[topk_inds, :]

#             bboxes = bbox_limited(bbox_pred, max_shape=img_shape)
#             mlvl_bboxes.append(bboxes)
#             mlvl_scores.append(scores)

#         mlvl_bboxes = torch.cat(mlvl_bboxes)
#         if rescale:
#             mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
#         mlvl_scores = torch.cat(mlvl_scores)
#         # Add a dummy background class to the backend when using sigmoid
#         # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
#         # BG cat_id: num_class
#         padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
#         mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

#         if with_nms:
#             det_bboxes, det_labels = multiclass_nms(
#                 mlvl_bboxes,
#                 mlvl_scores,
#                 cfg.score_thr,
#                 cfg.nms,
#                 cfg.max_per_img)
#             return det_bboxes, det_labels
#         else:
#             return mlvl_bboxes, mlvl_scores

#     def get_targets(self,
#                     cls_scores,
#                     bbox_preds,
#                     anchor_list,
#                     valid_flag_list,
#                     gt_bboxes_list,
#                     img_metas,
#                     gt_bboxes_ignore_list=None,
#                     gt_labels_list=None,
#                     label_channels=1,
#                     unmap_outputs=True):
#         """Get targets for TOOD head.

#         This method is almost the same as `AnchorHead.get_targets()`. Besides
#         returning the targets as the parent method does, it also returns the
#         anchors as the first element of the returned tuple.
#         """
#         num_imgs = len(img_metas)
#         assert len(anchor_list) == len(valid_flag_list) == num_imgs

#         # anchor number of multi levels
#         num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
#         num_level_anchors_list = [num_level_anchors] * num_imgs

#         # concat all level anchors and flags to a single tensor
#         for i in range(num_imgs):
#             assert len(anchor_list[i]) == len(valid_flag_list[i])
#             anchor_list[i] = torch.cat(anchor_list[i])
#             valid_flag_list[i] = torch.cat(valid_flag_list[i])

#         all_cls_scores = torch.cat(
#             [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in
#              cls_scores], 1)
#         all_bbox_preds = torch.cat(
#             [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0] for bbox_pred, stride in zip(bbox_preds, self.anchor_generator.strides)],
#             1)

#         # compute targets for each image
#         if gt_bboxes_ignore_list is None:
#             gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
#         if gt_labels_list is None:
#             gt_labels_list = [None for _ in range(num_imgs)]
#         # anchor_list: list(b * [-1, 4])
#         (all_anchors, all_labels, all_label_weights, all_bbox_targets,
#          all_bbox_weights, pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list,
#          assign_metrics_list, assign_ious_list, inside_flags_list) = multi_apply(
#              self._get_target_single,
#              all_cls_scores,
#              all_bbox_preds,
#              anchor_list,
#              valid_flag_list,
#              num_level_anchors_list,
#              gt_bboxes_list,
#              gt_bboxes_ignore_list,
#              gt_labels_list,
#              img_metas,
#              label_channels=label_channels,
#              unmap_outputs=unmap_outputs)
#         # no valid anchors
#         if any([labels is None for labels in all_labels]):
#             return None
#         # sampled anchors of all images
#         num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
#         num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
#         # split targets to a list w.r.t. multiple levels
#         anchors_list = images_to_levels(all_anchors, num_level_anchors)
#         labels_list = images_to_levels(all_labels, num_level_anchors)
#         label_weights_list = images_to_levels(all_label_weights,
#                                               num_level_anchors)
#         bbox_targets_list = images_to_levels(all_bbox_targets,
#                                              num_level_anchors)
#         bbox_weights_list = images_to_levels(all_bbox_weights,
#                                              num_level_anchors)

#         if self.epoch < self.initial_epoch:
#             norm_alignment_metrics_list = [bbox_weights[:, :, 0] for bbox_weights in bbox_weights_list]
#         else:
#             # for alignment metric
#             all_norm_alignment_metrics = []
#             for i in range(num_imgs):
#                 inside_flags = inside_flags_list[i]
#                 image_norm_alignment_metrics = all_label_weights[i].new_zeros(all_label_weights[i].shape[0])
#                 image_norm_alignment_metrics_inside = all_label_weights[i].new_zeros(inside_flags.long().sum())
#                 pos_assigned_gt_inds = pos_assigned_gt_inds_list[i]
#                 pos_inds = pos_inds_list[i]
#                 class_assigned_gt_inds = torch.unique(pos_assigned_gt_inds)
#                 for gt_inds in class_assigned_gt_inds:
#                     gt_class_inds = pos_inds[pos_assigned_gt_inds == gt_inds]
#                     pos_alignment_metrics = assign_metrics_list[i][gt_class_inds]
#                     pos_ious = assign_ious_list[i][gt_class_inds]
#                     pos_norm_alignment_metrics = pos_alignment_metrics / (pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
#                     image_norm_alignment_metrics_inside[gt_class_inds] = pos_norm_alignment_metrics

#                 image_norm_alignment_metrics[inside_flags] = image_norm_alignment_metrics_inside
#                 all_norm_alignment_metrics.append(image_norm_alignment_metrics)

#             norm_alignment_metrics_list = images_to_levels(all_norm_alignment_metrics,
#                                                   num_level_anchors)

#         return (anchors_list, labels_list, label_weights_list,
#                 bbox_targets_list, bbox_weights_list, num_total_pos,
#                 num_total_neg, norm_alignment_metrics_list)


#     def _get_target_single(self,
#                            cls_scores,
#                            bbox_preds,
#                            flat_anchors,
#                            valid_flags,
#                            num_level_anchors,
#                            gt_bboxes,
#                            gt_bboxes_ignore,
#                            gt_labels,
#                            img_meta,
#                            label_channels=1,
#                            unmap_outputs=True):
#         """Compute regression, classification targets for anchors in a single
#         image.

#         Args:
#             flat_anchors (Tensor): Multi-level anchors of the image, which are
#                 concatenated into a single tensor of shape (num_anchors ,4)
#             valid_flags (Tensor): Multi level valid flags of the image,
#                 which are concatenated into a single tensor of
#                     shape (num_anchors,).
#             num_level_anchors Tensor): Number of anchors of each scale level.
#             gt_bboxes (Tensor): Ground truth bboxes of the image,
#                 shape (num_gts, 4).
#             gt_bboxes_ignore (Tensor): Ground truth bboxes to be
#                 ignored, shape (num_ignored_gts, 4).
#             gt_labels (Tensor): Ground truth labels of each box,
#                 shape (num_gts,).
#             img_meta (dict): Meta info of the image.
#             label_channels (int): Channel of label.
#             unmap_outputs (bool): Whether to map outputs back to the original
#                 set of anchors.

#         Returns:
#             tuple: N is the number of total anchors in the image.
#                 labels (Tensor): Labels of all anchors in the image with shape
#                     (N,).
#                 label_weights (Tensor): Label weights of all anchor in the
#                     image with shape (N,).
#                 bbox_targets (Tensor): BBox targets of all anchors in the
#                     image with shape (N, 4).
#                 bbox_weights (Tensor): BBox weights of all anchors in the
#                     image with shape (N, 4)
#                 pos_inds (Tensor): Indices of postive anchor with shape
#                     (num_pos,).
#                 neg_inds (Tensor): Indices of negative anchor with shape
#                     (num_neg,).
#         """
#         inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
#                                            img_meta['img_shape'][:2],
#                                            self.train_cfg.allowed_border)
#         if not inside_flags.any():
#             return (None, ) * 7
#         # assign gt and sample anchors
#         anchors = flat_anchors[inside_flags, :]

#         num_level_anchors_inside = self.get_num_level_anchors_inside(
#             num_level_anchors, inside_flags)
#         if self.epoch < self.initial_epoch:
#             assign_result = self.initial_assigner.assign(anchors, num_level_anchors_inside,
#                                                  gt_bboxes, gt_bboxes_ignore,
#                                                  gt_labels)
#             assign_ious = assign_result.max_overlaps
#             assign_metrics = None
#         else:
#             assign_result = self.assigner.assign(cls_scores[inside_flags, :], bbox_preds[inside_flags, :],
#                                              anchors, num_level_anchors_inside,
#                                              gt_bboxes, gt_bboxes_ignore,
#                                              gt_labels, self.alpha, self.beta)
#             assign_ious = assign_result.max_overlaps
#             assign_metrics = assign_result.assign_metrics

#         sampling_result = self.sampler.sample(assign_result, anchors,
#                                               gt_bboxes)

#         num_valid_anchors = anchors.shape[0]
#         bbox_targets = torch.zeros_like(anchors)
#         bbox_weights = torch.zeros_like(anchors)
#         labels = anchors.new_full((num_valid_anchors, ),
#                                   self.num_classes,
#                                   dtype=torch.long)
#         label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds
#         if len(pos_inds) > 0:
#             # point-based
#             pos_bbox_targets = sampling_result.pos_gt_bboxes
#             bbox_targets[pos_inds, :] = pos_bbox_targets
#             bbox_weights[pos_inds, :] = 1.0

#             if gt_labels is None:
#                 # Only rpn gives gt_labels as None
#                 # Foreground is the first class since v2.5.0
#                 labels[pos_inds] = 0
#             else:
#                 labels[pos_inds] = gt_labels[
#                     sampling_result.pos_assigned_gt_inds]
#             if self.train_cfg.pos_weight <= 0:
#                 label_weights[pos_inds] = 1.0
#             else:
#                 label_weights[pos_inds] = self.train_cfg.pos_weight
#         if len(neg_inds) > 0:
#             label_weights[neg_inds] = 1.0

#         # map up to original set of anchors
#         if unmap_outputs:
#             num_total_anchors = flat_anchors.size(0)
#             anchors = unmap(anchors, num_total_anchors, inside_flags)
#             labels = unmap(
#                 labels, num_total_anchors, inside_flags, fill=self.num_classes)
#             label_weights = unmap(label_weights, num_total_anchors,
#                                   inside_flags)
#             bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
#             bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

#         return (anchors, labels, label_weights, bbox_targets, bbox_weights,
#                 pos_inds, neg_inds, sampling_result.pos_assigned_gt_inds, assign_metrics, assign_ious, inside_flags)

#     def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
#         split_inside_flags = torch.split(inside_flags, num_level_anchors)
#         num_level_anchors_inside = [
#             int(flags.sum()) for flags in split_inside_flags
#         ]
#         return num_level_anchors_inside