import torch
from ultralytics.nn.modules import TOODHead
import torch.nn as nn

# # 假设输入张量
# x = torch.randn(4, 128, 80, 80)  # [batch_size, channels, height, width]

# # 定义 GroupNorm
# num_groups = 16  # 确保 128 % 16 == 0
# norm_layer = nn.GroupNorm(num_groups=num_groups, num_channels=128)

# # 应用归一化
# output = norm_layer(x)
# print(output.shape)  # 应保持 [4, 128, 80, 80]


# 示例：3层特征图，每层维度为 (batch_size, channels, height, width)
batch_size = 1
c =  256
h = 80
w = 80
# feature_shapes = [256, 80, 80]
features = torch.rand(batch_size, c, h, w)



# 初始化检测头
nc = 2  # 类别数量
hidc = 256  # 隐藏层通道数
# block_num = 2  # 块的数量
ch = [128] # 输入通道数

detect_head = TOODHead(nc= nc, hidc = hidc, ch=ch)

# 前向传播
output = detect_head.forward(features)

# 打印输出结果
print("Output:", output)



# def test_detect_head():
#     features = [torch.rand(1, 256, 80, 80), torch.rand(1, 256, 40, 40), torch.rand(1, 256, 20, 20)]
#     detect_head = Detect_DyHead(nc=80, hidc=256, block_num=2, ch=[256, 256, 256])
#     output = detect_head(features)
#     assert isinstance(output, list), "Output should be a list."
#     assert len(output) == 3, "Output should contain predictions for each feature map."
#     print("Detect Head test passed!")

# test_detect_head()
