# 目录



# 机器学习基础

## 归一化和标准化

- 归一化：
  $$
  \frac{x-min}{max - min}
  $$

- 标准化：
  $$
  \frac{x-\mu}{\sigma}
  $$
  

## 正则化

- 可以规范权重参数的值，让权重参数倾向于0，消除一些权重参数的影响，达到dropout的效果，避免过拟合。
- 常用的是L2正则化，L1正则化，dropout。
- 其他方法解决过拟合的方法：数据增强、提早停止训练。



# 深度学习基础

## 一维、二维、三维卷积

- 一维卷积：只在一个方向移动的卷积
  - ![img](http://5b0988e595225.cdn.sohucs.com/images/20180427/0b24d32d46ef48159aed54a60eba2f68.png)
- 二维卷积：在h,w两个方向移动的卷积，包括多通道卷积。
  - ![img](https://img-blog.csdnimg.cn/20190505144536601.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDg0OTI3Mw==,size_16,color_FFFFFF,t_70)



- 三维卷积：在h,w，c上三个方向移动的卷积

  ![img](https://pic3.zhimg.com/v2-86e2bd970d07f9d6e1d921b248e45a3a_b.jpg)

## 梯度消失与梯度爆炸

- 梯度消失：主要是网络层数过多，或使用了不合适的激活函数，如Sigmoid导致的每层激活函数都会缩放到一个-1,1的区间，都乘以这个数字，因此会越乘越小。
  
  - 解决方法：使用残差网络，Relu激活函数，等都可以改善这个问题。
  
- 梯度爆炸：一般出现在**深层网络**和**权值初始化值太大**的情况下，在深层神经网络或循环神经网络中，**误差的梯度可在更新中累积相乘**。如果网络层之间的**梯度值大于 1.0**，那么**重复相乘会导致梯度呈指数级增长**，梯度变的非常大，然后导致网络权重的大幅更新，并因此使网络变得不稳定。

  - 解决方法：使用参数初始化和梯度裁剪

  

## loss权重设计

- 反传过程中，是 loss 对 参数求偏导，乘以-的lr，加上原来的参数，也就是说，如果给loss 加一个权重，如果给一个大权重，最终这个权重相当于乘到了lr上，也就是加大了lr。需要lr大的可以加大的loss权重，需要学的慢一点就加小的权重。



## 多标签分类，二分类，多分类

- 多标签分类和 二分类，用sigmod+bceloss
- 多分类用 softmax + crossentropyloss，或linear层加crossentropyloss
- 多分类一般认为分类的目标只有一个，而多标签分类用于分类的目标多个，如在yolov3中将多分类loss 改成了多标签分类loss，考虑到在同一个框中的物体，可能会属于多个类别。

## Batchnorm

- 训练时：计算一个mini-batch之内的均值和方差
- 测试时：用之前所有batch的平均的均值和方差
- 如果只想冻住batchnorm 不冻住其他参数，则需要 F.batch_norm(training=False), 如果是 nn.Batchnorm2d的话他是module类的 ，也有 training 参数。
- 有的模型会在使用预训练模型的时候，把batchnorm设置为 training=False，也就是冻住了，因为有的时候新训练的模型batchsize小，batchnorm可能会收到不好的影响，如果给预训练模型的batchnorm 冻住，就可以使用预训练模型的均值和方差。这样有助于训练，并且可以提高运算速度。
- 参数解析：
  - num_features：输入的样本
  - eps:  防止除于0，默认为1e-5
  - momentum: 为测试时均值和方差准备的 动量值，默认为0.1
  - affine ： 默认为True，可学习参数beta， gamma，若为False，则不学习。
  - track_running_stats=True表示跟踪整个训练过程中的batch的统计特性，得到方差和均值，而不只是仅仅依赖与当前输入的batch的统计特性。相反的，如果track_running_stats=False那么就只是计算当前输入的batch的统计特性中的均值和方差了
  - https://blog.csdn.net/qq_39777550/article/details/108038677



## SGD

```python
def cal_dx(func,x,eps=1e-8):
  d_func = func(x+eps) - func(x)
  dx = d_func / eps
  return dx

def solve(func,x,lr=0.01,num_iters=10000):
  print('init x:', x)
  for i in range(num_iters):
    dx = cal_dx(func,x)
    d_func = func(x)-0
    grad_x = d_func*dx
    x = x - grad_x*lr
    # print('iter{.4d}: x={.4f}'.format(i,x))
    return x

def check(func,x,eps=1e-8):
    if func(x) < eps:
        print('done')
    else:
        print('failed')

if __name__ == '__main__':
    init = 3
    func = lambda x: 2*x**2+2*x-12
    x = solve(func,init)
    print(x)
    check(func,x)
```



## MaxPooling

```python
class MaxPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=2):
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]

        self.stride = stride

        self.x = None
        self.in_height = None
        self.in_width = None

        self.out_height = None
        self.out_width = None
        # 要记录下在当前的滑动窗中最大值的索引，反向求导要用到
        self.arg_max = None

    def __call__(self, x):
        self.x = x
        self.in_height = np.shape(x)[0]
        self.in_width = np.shape(x)[1]

        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1

        out = np.zeros((self.out_height, self.out_width))
        self.arg_max = np.zeros_like(out, dtype=np.int32)

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = np.max(x[start_i: end_i, start_j: end_j])
                self.arg_max[i, j] = np.argmax(x[start_i: end_i, start_j: end_j])
            
        self.arg_max = self.arg_max
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.x)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                # 将索引展开成二维的
                index = np.unravel_index(self.arg_max[i, j], self.kernel_size)
                dx[start_i:end_i, start_j:end_j][index] = d_loss[i, j] #
        return dx
```



## Avg Pooling

```python
class AvgPooling2D:
    def __init__(self, kernel_size=(2, 2), stride=2):
        self.stride = stride
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]

    def __call__(self, x):
        self.x = x
        self.in_height = x.shape[0]
        self.in_width = x.shape[1]

        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1
        out = np.zeros((self.out_height, self.out_width))

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = np.mean(x[start_i: end_i, start_j: end_j])
        return out

    def backward(self, d_loss):
        dx = np.zeros_like(self.x)

        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                dx[start_i: end_i, start_j: end_j] = d_loss[i, j] / (self.w_width * self.w_height)
        return dx
```

- 测试

  ```python
  np.set_printoptions(precision=4, suppress=True, linewidth=120)
  x_numpy = np.random.random((1, 1, 6, 9))
  x_tensor = torch.tensor(x_numpy, requires_grad=True)
  
  max_pool_tensor = torch.nn.AvgPool2d((2, 2), 2)
  max_pool_numpy = AvgPooling2D((2, 2), stride=2)
  
  out_numpy = max_pool_numpy(x_numpy[0, 0])
  out_tensor = max_pool_tensor(x_tensor)
  
  d_loss_numpy = np.random.random(out_tensor.shape)
  d_loss_tensor = torch.tensor(d_loss_numpy, requires_grad=True)
  out_tensor.backward(d_loss_tensor)
  
  dx_numpy = max_pool_numpy.backward(d_loss_numpy[0, 0])
  dx_tensor = x_tensor.grad
  # print('input \n', x_numpy)
  print("out_numpy \n", out_numpy)
  print("out_tensor \n", out_tensor.data.numpy())
  
  print("dx_numpy \n", dx_numpy)
  print("dx_tensor \n", dx_tensor.data.numpy())
  ```



## CONV

```python
import numpy as np


def conv_naive(x, out_c, ksize, padding=0, stride=1):
    # x = [b, h, w, in_c]
    b, in_c, h, w = x.shape
    kernel = np.random.rand(ksize, ksize, in_c, out_c)
    if padding > 0:
        pad_x = np.zeros((b, in_c, h+2*padding, w+2*padding))
        pad_x[:,:,padding:-padding,padding:-padding] = x
    else:
        pad_x = x

    out_h = (h+2*padding-ksize)//stride+1
    out_w = (w+2*padding-ksize)//stride+1
    out = np.zeros((b, out_c, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            roi_x = pad_x[:,:,i*stride:i*stride+ksize,j*stride:j*stride+ksize]
            # roi_x = [b, in_c, ksize, ksize, in_c] -> [b, in_c, ksize, ksize, out_c]
            # kernel = [ksize, ksize, in_c, out_c]
            # conv = [b, ksize, ksize, in_c, out_c] -> [b, 1, 1, out_c]
            conv = np.tile(np.expand_dims(roi_x, -1), (1,1,1,1,out_c))* np.transpose(kernel, axes=(2,0,1,3))
            out[:,:,i,j] = np.squeeze(np.sum(conv, axis=(1,2,3), keepdims=True), axis=1)
    return out

if __name__ == '__main__':
    x = np.random.rand(1,3,10,10)
    out = conv_naive(x, 15, ksize=3, padding=1, stride=2)
    print(out.shape)
```



## BN

```python
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```





## 深度度量学习

- triplet loss



# CV note

## 分组卷积



## 空洞卷积(Dilated Convolution)

## [FPN](https://zhuanlan.zhihu.com/p/397293649)(特征金字塔网络)

- 先下采样得到多层特征，用深层特征与做上采样，与之前同等大小的底层feature融合，因此金字塔结构可以既包含底层语义又包含高级语义

## 金字塔与SSD的区别

- 金字塔是多个不同层的特征融合成一个size，SSD是不融合，直接用多个size去做预测





