import torch
import torch.nn as nn
import torch.nn.functional as F


x = torch.randn(1, 3, 32, 32)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride = 2, padding= 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding= 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride = 2) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + skip
    

block = ResidualBlock(3, 8)
y = block(x)
print("Residual output shape:", y.shape)