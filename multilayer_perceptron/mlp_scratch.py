import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from d2l import torch as d2l

# 1. 加载数据
train_iter = torchvision.datasets.FashionMNIST(
    root="../data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_iter = torchvision.datasets.FashionMNIST(
    root="../data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# 2. 利用DataLoader加载数据集
batch_size = 256
dataloader_workers = 4
train_iter = DataLoader(train_iter, batch_size, shuffle=True, num_workers=dataloader_workers)
test_iter = DataLoader(test_iter, batch_size, shuffle=False, num_workers=dataloader_workers)

# 3. 搭建隐藏层：只有一层隐藏层，包含256个隐藏单元
num_inputs, num_outputs, num_hiddens = 28 * 28, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))  # 第一层weight
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))  # 第一层bias
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))  # 第二层/输出层weight
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))  # 第二层/输出层bias
params = [W1, b1, W2, b2]


# 4. ReLU激活函数，自己定义
def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


# 5. 完全手搭模型nn
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # @ <=> dot
    return (H @ W2 + b2)


# 6. 损失函数
loss = nn.CrossEntropyLoss()

# 7. 开始训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)  # 优化器
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)  # 仅仅用于画图
