"""
读取数据教程
"""

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 1. 读取数据集
"""
通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
并除以255使得所有像素的数值均在0～1之间
"""
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=False)


# 2. 两个可视化数据集的函数（不重要）
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 3. 读取小批量
def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


# 4. 训练
batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

# 4. 训练时间
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

# def get_dataloader_workers():
#     """使用4个进程来读取数据"""
#     return 4
#
# # 整合以上所有功能
# def load_data_fashion_mnist(batch_size, resize=None):
#     """下载Fashion-MNIST数据集，然后将其加载到内存中"""
#     trans = [transforms.ToTensor(), ]
#     # 如果图片过大，需要resize可以在这个函数里设定是否resize
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=False)
#     mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=False)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
#             data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))
#
#
# # 训练
# train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break
