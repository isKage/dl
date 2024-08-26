from kaggle_house_price_model_selection import *
import matplotlib.pyplot as plt


# 画图
def plot(num_epochs, train_ls, xlabel, ylabel, xlim, yscale):
    plt.plot(np.arange(1, num_epochs + 1), train_ls, label='train: log rmse')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.yscale(yscale)
    plt.legend()
    plt.show()


# 5. 开始预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    plot(
        num_epochs=num_epochs,
        train_ls=train_ls,
        xlabel='epoch',
        ylabel='log rmse',
        xlim=[1, num_epochs],
        yscale='log'
    )
    print('训练log rmse = {:.6f}'.format(float(train_ls[-1])))

    # 将网络应用于测试集
    preds = net(test_features).detach().numpy()

    # 将其重新格式化导出为文件'submission.csv'
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./data/submission.csv', index=False)


# 6. 预测
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
