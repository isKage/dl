"""
参数绑定
"""
import torch
from torch import nn
from torch.nn import functional as F

# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8, 1)
)

X = torch.rand(size=(2, 4))
net(X)

# 检查参数是否相同，注意ReLU不是层
print(net[2].weight.data[0] == net[4].weight.data[0])

# 更新第2层参数，检查第2层和第4层是否仍然相同
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
