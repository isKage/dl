"""
保存和下载
"""
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


# 从熟悉的多层感知机开始尝试一下
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 我们将模型的`参数`存储在一个叫做`mlp.params`的文件中
torch.save(net.state_dict(), './model/mlp.params')

# 实例化原始多层感知机模型的一个备份，导入参数
clone = MLP()
clone.load_state_dict(torch.load('./model/mlp.params'))
print(clone.eval())

# 验证输出是否一样
Y_clone = clone(X)
print(Y_clone == Y)