import torch
import torch.nn as nn

from ultralytics.nn.modules import MSPA_C2f
#from ultralytics.nn.modules import SPRModule
input_channels = 128
output_channels = 128
dummy_input = torch.randn(1, input_channels, 160, 160)  # 假设输入通道为64
#sp_inp = torch.chunk(dummy_input, 4, 1)

# for inp in sp_inp:
#     print("----------------")
#     print(inp.shape)
#     print("----------------")

# spr_module = SPRModule(32)
# output = spr_module(dummy_input)
# print(output.shape)
mspa_c2f = MSPA_C2f(input_channels,output_channels)
output = mspa_c2f(dummy_input)