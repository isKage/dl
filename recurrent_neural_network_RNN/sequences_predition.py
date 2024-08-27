from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from sequences_model import features, time, x, T, n_train, tau


def plot(x, y, title, xlabel, ylabel, xlim, legend):
    for i in range(len(x)):
        plt.plot(x[i], y[i], label=legend[i])  # 绘制第i条线
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.legend(legend)  # 在有 label 的情况下，这里才有作用
    plt.show()


net = nn.Sequential(
    nn.Linear(in_features=4, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=1)
)

net.load_state_dict(torch.load('./model/net_4.params'))

# 1. 单步预测
onestep_preds = net(features)
plot(x=[time, time[tau:]],
     y=[x.detach().numpy(), onestep_preds.detach().numpy()],
     title='sequence prediction - one step',
     xlabel='time',
     ylabel='x',
     legend=['data', '1-step preds'],
     xlim=[1, 1000],
     )

# 2. k步预测
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))

plot(x=[time, time[tau:], time[n_train + tau:]],
     y=[x.detach().numpy(),
        onestep_preds.detach().numpy(),
        multistep_preds[n_train + tau:].detach().numpy(),
        ],
     title='sequence prediction - multistep',
     xlabel='time',
     ylabel='x',
     legend=['data', '1-step preds', 'multistep preds'],
     xlim=[1, 1000]
     )

# 3. k步预测，k = 1, 4, 16, 64
max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
plot(x=[time[tau + i - 1: T - max_steps + i] for i in steps],
     y=[features[:, (tau + i - 1)].detach().numpy() for i in steps],
     title='sequence prediction - multistep k = 1, 4, 16, 64',
     xlabel='time',
     ylabel='x',
     legend=[f'{i}-step preds' for i in steps],
     xlim=[5, 1000],
     )
