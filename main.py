import torch as th
from models.basic_blocks import BasicBlock

x = th.randn(1,16,32,32)

block_with_skip = BasicBlock(16, 32, use_skip=True)
block_no_skip = BasicBlock(16, 32, use_skip=False)

y1 = block_with_skip(x)
y2 = block_no_skip(x)

print("With Skip:", y1.shape)
print("Without Skip:", y2.shape)

# print("Input mean:", x.mean().item())
# print("With skip mean:", y1.mean().item())
# print("No skip mean:", y2.mean().item())
