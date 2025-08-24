# import sys
# sys.path.insert(0, "/media/robot/7846E2E046E29DDE/paper_code_source/our_ultralytics")
import torch
from thop import profile
from thop import clever_format

import ultralytics.nn.modules
print(ultralytics.nn.modules.__file__)  # Should print your local path
from ultralytics.nn.modules import (C2f, MSPA_C2f)
# from ultralytics.nn.modules import C2f
# from ultralytics.nn.modules import MSPA_C2f

input_channels = 32
output_channels = 32
n = 1

height = 160
width = 160

# Example: Original and redesigned C2f module
original_model = C2f(input_channels, output_channels, n=1)  # Original C2f module
new_model = MSPA_C2f(input_channels, output_channels, n=1)  # Redesigned C2f module

# Input tensor shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, input_channels, height, width)

# Compute FLOPs and parameters
flops_orig, params_orig = profile(original_model, inputs=(input_tensor,))
flops_new, params_new = profile(new_model, inputs=(input_tensor,))

# Format into a more readable form
flops_orig, params_orig = clever_format([flops_orig, params_orig], "%.3f")
flops_new, params_new = clever_format([flops_new, params_new], "%.3f")

print(f"Original C2f: FLOPs={flops_orig}, Params={params_orig}")
print(f"MSPA C2f: FLOPs={flops_new}, Params={params_new}")

