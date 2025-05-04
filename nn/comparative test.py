import os
from mmdet.apis import init_detector, single_gpu_test
from mmdet.datasets import build_dataset, build_dataloader
from mmcv import Config
from mmcv.runner import load_checkpoint
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test ATSS on custom dataset')
    # parser.add_argument('--config', type=str, default='configs/faster_rcnn_r50_fpn_1x_coco.py', help='Path to the model config file')
    # parser.add_argument('--checkpoint', type=str, default='checkpoints/faster_rcnn_r50_fpn_1x_coco.pth', help='Path to the model checkpoint file')
    # parser.add_argument('--data_root', type=str, default='data/my_dataset', help='Root directory of the dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda:0)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 配置文件路径
    config_path ='/media/robot/7846E2E046E29DDE/comparative_model_and_cfg/mmdetection-main/configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py'
    checkpoint_path = '/media/robot/7846E2E046E29DDE/comparative_model_and_cfg/ATSS/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth'
    data_root = '/media/robot/7846E2E046E29DDE/piglet_pic_from_net/'

    cfg = Config.fromfile(config_path)

    # 修改数据集路径
    if 'data' in cfg:
        cfg.data.train.ann_file = os.path.join(data_root, 'train/labels/instances_train.json')
        cfg.data.train.img_prefix = os.path.join(data_root, 'train/images/')
        cfg.data.val.ann_file = os.path.join(data_root, 'test/labels/instances_val.json')
        cfg.data.val.img_prefix = os.path.join(data_root, 'test/images/')
        cfg.data.test.ann_file = os.path.join(data_root, 'test/labels/instances_val.json')
        cfg.data.test.img_prefix = os.path.join(data_root, 'test/images/')
        cfg.data.samples_per_gpu = 16  # 每个 GPU 加载的图片数量
    else:
        raise AttributeError("The config file does not contain a 'data' field. "
                             "Please check if '_base_' configurations are properly included.")

    # 修改配置文件中的数据集路径
    # cfg = Config.fromfile(config_path)
    # cfg.data.train.ann_file = os.path.join(data_root, 'train/labels/instances_train.json')
    # cfg.data.train.img_prefix = os.path.join(data_root, 'train/images/')
    # cfg.data.val.ann_file = os.path.join(data_root, 'test/labels/instances_val.json')
    # cfg.data.val.img_prefix = os.path.join(data_root, 'test/images/')
    # cfg.data.test.ann_file = os.path.join(data_root, 'test/labels/instances_val.json')
    # cfg.data.test.img_prefix = os.path.join(data_root, 'test/images/')
    # cfg.data.samples_per_gpu = 16  # 每个 GPU 加载的图片数量

    # 初始化模型
    model = init_detector(cfg, checkpoint_path, device=args.device)

    # 构建数据集和数据加载器
    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=16,
        workers_per_gpu=8,
        dist=False,
        shuffle=False
    )

    # 测试模型性能
    print('Starting evaluation...')
    outputs = single_gpu_test(model, dataloader, show_score_thr=0.3)

    # 输出评估结果
    eval_result = dataset.evaluate(outputs, metric=['bbox'])
    print('Evaluation results:', eval_result)

if __name__ == '__main__':
    main()
