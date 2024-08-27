from matplotlib import pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.data import TensorDataset, DataLoader


def plot(x, y, title, xlabel, ylabel, xlim, legend):
    # 使用 plt.plot() 而不是 plt.scatter() 来绘制折线图
    plt.plot(x[0], y[0], label=legend[0])  # 绘制第一条线
    plt.plot(x[1], y[1], label=legend[1])  # 绘制第二条线
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.legend(legend)  # 在有 label 的情况下，这里才有作用
    plt.show()


# 1. 生成数据
T = 1000  # 总共产生1000个点, 横轴
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# 调用 plot 函数，y 需要是一个一维张量而不是嵌套列表
# plot(x=[time, ], y=[x, ], title='sequences_model', xlabel='time', ylabel='x', xlim=[1, 1000])

# 2. 转换为模型的 特征－标签对
tau = 4
features = torch.zeros((T - tau, tau))  # T - tau 行, tau 列, 初始化这个特征矩阵
# 如果 x = [x1, x2, x3, x4, x5, x6, x7, x8] 是一个时间序列，T = 8
# 设定 tau = 3，则我们的任务是通过 [x1, x2, x3] 来预测 x4，通过 [x2, x3, x4] 来预测 x5，以此类推。
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_dataset = TensorDataset(features[:n_train], labels[:n_train])
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 3. 搭建网络
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features=4, out_features=10),
        nn.ReLU(),
        nn.Linear(in_features=10, out_features=1))
    net.apply(init_weights)
    return net


# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
# 当 reduction='none' 时，nn.MSELoss 不会对损失值进行任何归约操作，而是返回一个与输入形状相同的张量，其中每个元素对应输入样本的损失
# 当 reduction='mean' or 'sum' 时，返回标量
loss_fn = nn.MSELoss(reduction='mean')


# 4. 开始训练
def train(net, train_iter, loss_fn, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    net.train()
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss_fn(net(X), y)
            l.backward()
            trainer.step()
        print("epoch: {}, loss: {}".format(epoch, d2l.evaluate_loss(net, train_iter, loss_fn)))  # 评估的是损失平均到参与训练的样本的平均损失
        torch.save(net.state_dict(), './model/net_{}.params'.format(epoch))

net = get_net()
train(net, train_iter, loss_fn, 5, 0.01)
