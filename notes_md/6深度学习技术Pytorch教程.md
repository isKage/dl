# Pytorch教程

主题：深度学习技术、Pytorch教程、层和块、参数管理、延后初始化、自定义层、读写文件




## 1 层和块

- 回顾一下多层感知机

```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```



### 1. 自定义块

#### - 定义

在下面的代码片段中，我们从零开始编写一个块。 它包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。 注意，下面的`MLP`类继承了表示块的类。 我们的实现只需要提供我们自己的构造函(Python中的`__init__`函数)和前向传播函数。

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

我们首先看一下前向传播函数，它以`X`作为输入， 计算带有激活函数的隐藏表示，并输出其未规范化的输出值。 在这个`MLP`实现中，两个层都是实例变量。

接着我们实例化多层感知机的层，然后在每次调用前向传播函数时调用这些层。 注意一些关键细节： 首先，我们定制的`__init__`函数通过`super().__init__()` 调用父类的`__init__`函数， 省去了重复编写模版代码的痛苦。 然后，我们实例化两个全连接层， 分别为`self.hidden`和`self.out`。 注意，除非我们实现一个新的运算符， 否则我们不必担心反向传播函数或参数初始化， 系统将自动生成这些。



#### - 使用

```python
net = MLP()
net(X)
```



### 2. 顺序块

实现`Sequential`类：

`Sequential`的设计是为了把其他模块串起来。 为了构建我们自己的简化的`MySequential`， 我们只需要定义两个关键函数：

1. 一种将块逐个追加到列表中的函数；
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。



#### - 复现Sequential块

下面的`MySequential`类提供了与默认`Sequential`类相同的功能。

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中._module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```



#### - 使用

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

> Tips: 会自动按照顺序运行这些层



### 3. 在前向传播函数中执行代码

#### - 完全自主灵活的编程

在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层， 其权重（`self.rand_weight`）在实例化时被随机初始化，之后为常量。 这个权重不是一个模型参数，因此它永远不会被反向传播更新。 然后，神经网络将这个固定层的输出通过一个全连接层。

注意，在返回输出之前，模型做了一些不寻常的事情： 它运行了一个while循环，在L1范数大于1的条件下， 将输出向量除以2，直到它满足条件为止。 最后，模型返回了`X`中所有项的和。 

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```python
net = FixedHiddenMLP()
net(X)
```



### 4. 混合搭配各种组合块

#### - 嵌套

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

> Tips: 按照顺序执行



## 2 参数管理

我们首先看一下具有单隐藏层的多层感知机。

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

### 1. 参数访问

当通过`Sequential`类定义模型时， 我们可以通过索引来访问模型的任意层。 这就像模型是一个列表一样，每层的参数都在其属性中。 如下所示，我们可以检查第2个全连接层的参数(0, 1, 2)

```python
print(net[2].state_dict())
```

```python
>>> OrderedDict([
>>>     ('weight', tensor([[-0.0427, -0.2939, -0.1894,  0.0220, -0.1709, -0.1522, 
>>>                         -0.0334, -0.2263]])),
>>>     ('bias', tensor([0.0887]))
>>> ])
```



#### - 目标参数

下面的代码从第二个全连接层（即第三个神经网络层）提取偏置， 提取后返回的是一个参数类实例，并进一步访问该参数的值

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad) # 返回梯度，因为目前尚未计算，所有为None
```

```python
>>> <class 'torch.nn.parameter.Parameter'>

>>> Parameter containing:
>>> tensor([0.0887], requires_grad=True)

>>> tensor([0.0887])

>>> None
```



#### - 一次性访问所有参数

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

```python
>>> ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))

>>> ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) 
>>> ('2.weight',torch.Size([1, 8])) ('2.bias', torch.Size([1]))
# 没有第1层，是因为第1层为relu激活函数，没有参数
```

从这，得到另一种访问网络参数的方式，如下所示

```python
# 第2层bias的值
net.state_dict()['2.bias'].data
```

```python
>>> tensor([0.0887])
```



#### - 从嵌套块收集参数

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

查看网络

```python
print(rgnet)
```

```python
>>> Sequential(
>>>   (0): Sequential(
>>>     (block 0): Sequential(
>>>       (0): Linear(in_features=4, out_features=8, bias=True)
>>>       (1): ReLU()
>>>       (2): Linear(in_features=8, out_features=4, bias=True)
>>>       (3): ReLU()
>>>     )
>>>     (block 1): Sequential(
>>>       (0): Linear(in_features=4, out_features=8, bias=True)
>>>       (1): ReLU()
>>>       (2): Linear(in_features=8, out_features=4, bias=True)
>>>       (3): ReLU()
>>>     )
>>>     (block 2): Sequential(
>>>       (0): Linear(in_features=4, out_features=8, bias=True)
>>>       (1): ReLU()
>>>       (2): Linear(in_features=8, out_features=4, bias=True)
>>>       (3): ReLU()
>>>     )
>>>     (block 3): Sequential(
>>>       (0): Linear(in_features=4, out_features=8, bias=True)
>>>       (1): ReLU()
>>>       (2): Linear(in_features=8, out_features=4, bias=True)
>>>       (3): ReLU()
>>>     )
>>>   )
>>>   (1): Linear(in_features=4, out_features=1, bias=True)
>>> )
```



### 2. 初始化参数

PyTorch的`nn.init`模块提供了多种预置初始化方法

#### - 内置初始化

- init_normal

下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0

```python
def init_normal(m):
    if type(m) == nn.Linear:
        # m表示module
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
        
net.apply(init_normal)  # 对net网络每一层，逐层使用函数init_normal

print(net[0].weight.data[0])
print(net[0].bias.data[0])
```

```python
>>> tensor([ 0.0220,  0.0175, -0.0022,  0.0043])
>>> tensor(0.)
```

- init_constant

类似的，还可以将所有参数初始化为给定的常数，比如初始化为1

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0])
print(net[0].bias.data[0])
```

- init_xavier

我们还可以对某些块应用不同的初始化方法。 例如，下面我们使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42

```python
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
```

```python
>>> tensor([ 0.2457,  0.4410, -0.0183, -0.5038])
>>> tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
```



#### - 自定义初始化

我们使用以下的分布为任意权重参数w定义初始化方法：
$$
\begin{aligned}

​    w \sim \begin{cases}

​        U(5, 10) & \textrm{ 可能性 } \frac{1}{4} \\

​            0    & \textrm{ 可能性 } \frac{1}{2} \\

​        U(-10, -5) & \textrm{ 可能性 } \frac{1}{4}

​    \end{cases}

\end{aligned}
$$

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])
```

```python
>>> tensor([[-0.0000, -0.0000, -8.5102, -0.0000],
>>>         [ 8.6104, -0.0000, -0.0000,  6.8386]], grad_fn=<SliceBackward0>)
```



> 注意，我们始终可以直接设置参数。

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```python
>>> tensor([42.0000, 10.3334,  6.0616,  9.3095])
```



### 3. 参数绑定

有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8, 1)    
)

net(X)

# 检查参数是否相同，注意ReLU不是层
print(net[2].weight.data[0] == net[4].weight.data[0])

# 更新第2层参数，检查第2层和第4层是否仍然相同
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```python
>>> tensor([True, True, True, True, True, True, True, True])
>>> tensor([True, True, True, True, True, True, True, True])
```



## 3 自定义层

### 1. 不带参数的层

首先，我们构造一个没有任何参数的自定义层。下面的`CenteredLayer`类要从其输入中减去均值。 要构建它，我们只需继承基础层类并实现前向传播功能。

```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

让我们向该层提供一些数据，验证它是否能按预期工作。

```python
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```python
>>> tensor([-2., -1.,  0.,  1.,  2.])
```

现在，我们可以将层作为组件合并到更复杂的模型中。

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

作为额外的健全性检查，我们可以在向该网络发送随机数据后，检查均值是否为0。 由于我们处理的是浮点数，因为存储精度的原因，我们仍然可能会看到一个非常小的非零数。

```python
Y = net(torch.rand(4, 8))
Y.mean()
```

```python
>>> tensor(7.4506e-09, grad_fn=<MeanBackward0>)
```



### 2. 带参数的层

#### - 搭建自定义层

现在，让我们实现自定义版本的全连接层。 该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。 在此实现中，我们使用修正线性单元作为激活函数。 

该层需要输入参数：`in_units`和`units`，分别表示输入数和输出数，这决定了weights的矩阵形状和bias的向量形状。

```python
class MyLinear(nn.Module):
    # 传入in_units, units参数用于定义矩阵形状
    def __init__(self, in_units, units):
        super().__init__()
        # nn.Parameter()函数用于告诉模型这是希望学习的参数
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

实例化`MyLinear`类并访问其模型参数。

```python
dense = MyLinear(5, 3)  # 输入特征为5维，输出结果为3维
print(dense.weight)
```

```python
>>> Parameter containing:
>>> tensor([[-0.5637, -0.6687, -2.3779],
>>>         [-1.3287,  0.2991, -0.5195],
>>>         [ 0.4195,  0.6346,  1.0412],
>>>         [ 0.4689,  0.6360, -1.6022],
>>>         [ 0.5129, -0.1303, -0.5330]], requires_grad=True)
```



#### - 前行传播

例如选取一个输入，得到输出

```python
X = torch.rand(2, 5)
y_hat = dense(X)
print(y_hat)
```

```python
>>> tensor([[0.3253, 1.9079, 0.0000],
>>>         [0.0000, 1.0152, 0.4838]])
```



#### - 嵌套

我们还可以使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层。

```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```



## 4 读写文件(模型)

### 1. 加载和保存张量

#### - 张量

对于单个张量，我们可以直接调用`load`和`save`函数分别读写它们。 这两个函数都要求我们提供一个名称，`save`要求将要保存的变量作为输入。

```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')  # x为张量，'x-file'文件名
```

我们现在可以将存储在文件中的数据读回内存。

```python
x2 = torch.load('x-file')
print(x2)
```

```python
>>> tensor([0, 1, 2, 3])
```



#### - 张量列表

我们可以存储一个张量列表，然后把它们读回内存。

```python
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```python
>>> (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))
```



#### - 字符串映射到张量的字典

我们甚至可以写入或读取从字符串映射到张量的字典。 当我们要读取或写入模型中的所有权重时，这很方便。

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
```

```python
>>> {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
```



### 2. 加载和保存模型参数

为了恢复模型，我们需要用代码生成架构， 然后从磁盘加载参数。 让我们从熟悉的多层感知机开始尝试一下。

```flow
st=>start: modol
op=>operation: load parameters
e=>end: full net
st(right)->op(right)->e
```

#### - 搭建网络

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```



#### - 存储参数

我们将模型的`参数`存储在一个叫做`mlp.params`的文件中

```python
torch.save(net.state_dict(), './model/mlp.params')
```



#### - 创建网络备份，然后导入参数

为了恢复模型，我们实例化了原始多层感知机模型的一个备份。 这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数。

```python
clone = MLP()
clone.load_state_dict(torch.load('./model/mlp.params'))
print(clone.eval())
```

```python
>>> MLP(
>>>   (hidden): Linear(in_features=20, out_features=256, bias=True)
>>>   (output): Linear(in_features=256, out_features=10, bias=True)
>>> )
```



#### - 验证

由于两个实例具有相同的模型参数，在输入相同的`X`时， 两个实例的计算结果应该相同。 让我们来验证一下。

```python
Y_clone = clone(X)
print(Y_clone == Y)
```

```python
>>> tensor([[True, True, True, True, True, True, True, True, True, True],
>>>         [True, True, True, True, True, True, True, True, True, True]])
```



































