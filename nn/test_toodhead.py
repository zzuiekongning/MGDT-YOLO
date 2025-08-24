import torch
from ultralytics.nn.modules import TOODHead
import torch.nn as nn

# Example: Input tensor
# x = torch.randn(4, 128, 80, 80)  # [batch_size, channels, height, width]

# # Define GroupNorm
# num_groups = 16  # Ensure 128 % 16 == 0
# norm_layer = nn.GroupNorm(num_groups=num_groups, num_channels=128)

# # Apply normalization
# output = norm_layer(x)
# print(output.shape)  # Should remain [4, 128, 80, 80]


# Example: 3-layer feature map, each with shape (batch_size, channels, height, width)
batch_size = 1
c = 256
h = 80
w = 80
# feature_shapes = [256, 80, 80]
features = torch.rand(batch_size, c, h, w)

# Initialize the detection head
nc = 2  # Number of classes
hidc = 256  # Hidden layer channels
# block_num = 2  # Number of blocks
ch = [128]  # Input channels

detect_head = TOODHead(nc=nc, hidc=hidc, ch=ch)

# Forward pass
output = detect_head.forward(features)

# Print the output
print("Output:", output)




# def test_detect_head():
#     features = [torch.rand(1, 256, 80, 80), torch.rand(1, 256, 40, 40), torch.rand(1, 256, 20, 20)]
#     detect_head = Detect_DyHead(nc=80, hidc=256, block_num=2, ch=[256, 256, 256])
#     output = detect_head(features)
#     assert isinstance(output, list), "Output should be a list."
#     assert len(output) == 3, "Output should contain predictions for each feature map."
#     print("Detect Head test passed!")

# test_detect_head()
