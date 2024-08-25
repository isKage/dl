"""
重要
"""

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor(), ]
    # 如果图片过大，需要resize可以在这个函数里设定是否resize
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


# 1. 获取数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 2. 初始化模型参数
# PyTorch不会隐式地调整输入的形状。因此，我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))  # 28*28的图片平展到784，输出10个类


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# 3. 损失
loss = nn.CrossEntropyLoss(reduction='none')

# 4. 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 5. 训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)  # 展示动态图片(jupyter)
