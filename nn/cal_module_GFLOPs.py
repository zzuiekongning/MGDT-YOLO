# import sys
# sys.path.insert(0, "/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics")
import torch
from thop import profile
from thop import clever_format

import ultralytics.nn.modules
print(ultralytics.nn.modules.__file__)  # 应该打印你的本地路径
from ultralytics.nn.modules import (C2f, MSPA_C2f)
# from ultralytics.nn.modules import C2f
# from ultralytics.nn.modules import MSPA_C2f

input_channels = 32
output_channels = 32
n = 1

height = 160
width = 160

# 示例：原始和重新设计的 C2f 模块
original_model = C2f(input_channels,output_channels,n=1)  # 原始 C2f 模块
new_model = MSPA_C2f(input_channels,output_channels,n=1)  # 重新设计后的模块


# 输入张量的形状 (batch_size, channels, height, width)
input_tensor = torch.randn(1, input_channels, height, width)

# 计算 FLOPs 和参数量
flops_orig, params_orig = profile(original_model, inputs=(input_tensor,))
flops_new, params_new = profile(new_model, inputs=(input_tensor,))

# 格式化为更易读的形式
flops_orig, params_orig = clever_format([flops_orig, params_orig], "%.3f")
flops_new, params_new = clever_format([flops_new, params_new], "%.3f")

print(f"Original C2f: FLOPs={flops_orig}, Params={params_orig}")
print(f"MSPA C2f: FLOPs={flops_new}, Params={params_new}")
