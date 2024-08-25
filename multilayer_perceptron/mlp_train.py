import torch
import torchvision
from torch.utils.data import DataLoader

from mlp_model import *

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

# 2. 获取数据集长度
train_data_size = len(train_iter)
test_data_size = len(test_iter)

# 3. 利用DataLoader加载数据集
batch_size = 256
train_iter = DataLoader(train_iter, batch_size, shuffle=True)
test_iter = DataLoader(test_iter, batch_size, shuffle=False)

# 4. 利用pytorch搭建神经网络 [搭建隐藏层：只有一层隐藏层，包含256个隐藏单元]
net = net()

# 5. 损失函数
loss_fn = nn.CrossEntropyLoss()

# 6. 优化器
learning_rate = 0.1
optimizer = torch.optim.SGD(
    params=net.parameters(),
    lr=learning_rate,
)

# 7. 设置训练网络的参数
total_train_step = 0  # 训练次数
total_test_step = 0  # 测试次数 == epoch
epochs = 10  # 训练迭代次数

# 8. 开始训练
for epoch in range(epochs):
    print("------------- 第 {} 轮训练开始 -------------".format(epoch))

    # 训练步骤
    net.train()
    for data in train_iter:
        # 输入输出
        images, targets = data
        outputs = net(images)

        # 损失函数
        loss = loss_fn(outputs, targets)

        # 清零梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))

    # 测试步骤(不更新参数)
    net.eval()
    total_test_loss = 0  # 测试集损失累积
    total_accuracy = 0  # 分类问题正确率
    with torch.no_grad():
        for data in test_iter:
            images, targets = data
            outputs = net(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            # 正确率
            accuracy = (outputs.argmax(axis=1) == targets).sum()
            total_accuracy += accuracy

    # 在测试集上的损失
    print("##### 在测试集上的loss: {} #####".format(total_test_loss))

    # 在测试集上的正确率
    print("##### 在测试集上的正确率: {} #####".format(total_accuracy / test_data_size))

    # 保存每次训练的模型
    torch.save(net.state_dict(), "../multilayer_perceptron/model/mlp_{}.pth".format(epoch)) # 推荐
    print("##### 模型成功保存 #####")
