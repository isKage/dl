"""
自定义层
"""
import torch
import torch.nn.functional as F
from torch import nn


# 没有参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

# 现在，我们可以将层作为组件合并到更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# 我们可以在向该网络发送随机数据后，检查均值是否为0
Y = net(torch.rand(4, 8))
print(Y.mean())
print("====================================================================================")


# 有参数的层
# 该层需要输入参数：`in_units`和`units`，分别表示输入数和输出数，这决定了weights的矩阵形状和bias的向量形状。
class MyLinear(nn.Module):
    # 传入in_units, units参数用于定义矩阵形状
    def __init__(self, in_units, units):
        super().__init__()
        # nn.Parameter()函数用于告诉模型这是希望学习的参数
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


dense = MyLinear(5, 3)  # 输入特征为5维，输出结果为3维
# print(dense.weight)

# 前向传播
X = torch.rand(2, 5)
y_hat = dense(X)
print(y_hat)
