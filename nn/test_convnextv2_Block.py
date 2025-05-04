import torch
import torch.nn as nn

from ultralytics.nn.modules import (ConvNeXtV2_Block, RepVGGBlock)
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
convnextv2_block = ConvNeXtV2_Block(input_channels)
repvgg_block = RepVGGBlock(input_channels, output_channels)
convnextv2_block_output = convnextv2_block(dummy_input)
repvgg_block_output = repvgg_block(dummy_input)
print("----------------")
print(convnextv2_block_output.shape)
print("----------------")
print("****************")
print(repvgg_block_output.shape)
print("****************")