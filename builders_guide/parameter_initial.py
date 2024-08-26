"""
参数初始化
"""

import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def init_normal(m):
    '''初始化参数'''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(net[0].weight.data[0])
print(net[0].bias.data[0])


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


# 对不同的层使用不同的初始化函数
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(net[0].weight[:2])
