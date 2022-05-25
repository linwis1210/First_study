- [ 目录](#head1)
- [<span id = "1"> 机器学习基础</span>](#head2)
	- [ 归一化和标准化](#head3)
	- [ 正则化](#head4)
- [ 深度学习基础](#head5)
	- [ 一维、二维、三维卷积](#head6)
	- [ 梯度消失与梯度爆炸](#head7)
	- [ loss权重设计](#head8)
	- [ 多标签分类，二分类，多分类](#head9)
	- [ Batchnorm](#head10)
	- [ SGD](#head11)
	- [ MaxPooling](#head12)
	- [Avg Pooling](#head13)
	- [ CONV](#head14)
	- [ BN](#head15)
	- [ Loss函数](#head16)
	- [ 深度度量学习](#head17)
	- [ 解决样本不均衡方法](#head18)
- [CV note](#head19)
	- [ 分组卷积](#head20)
	- [空洞卷积(Dilated Convolution)](#head21)
	- [ FPN（特征金字塔网络)](#head22)
	- [ 金字塔与SSD的区别](#head23)
# <span id="head1"> 目录</span>

# <span id="head2"><span id = "1"> 机器学习基础</span></span>

## <span id="head3"> 归一化和标准化</span>

- 归一化：
  $$
  \frac{x-min}{max - min}
  $$

- 标准化：
  $$
  \frac{x-\mu}{\sigma}
  $$
  

## <span id="head4"> 正则化</span>

- 可以规范权重参数的值，让权重参数倾向于0，消除一些权重参数的影响，达到dropout的效果，避免过拟合。
- 常用的是L2正则化，L1正则化，dropout。
- 其他方法解决过拟合的方法：数据增强、提早停止训练。



# <span id="head5"> 深度学习基础</span>

## <span id="head6"> 一维、二维、三维卷积</span>

- 一维卷积：只在一个方向移动的卷积
  - ![img](http://5b0988e595225.cdn.sohucs.com/images/20180427/0b24d32d46ef48159aed54a60eba2f68.png)
- 二维卷积：在h,w两个方向移动的卷积，包括多通道卷积。
  - ![img](https://img-blog.csdnimg.cn/20190505144536601.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDg0OTI3Mw==,size_16,color_FFFFFF,t_70)



- 三维卷积：在h,w，c上三个方向移动的卷积

  ![img](https://pic3.zhimg.com/v2-86e2bd970d07f9d6e1d921b248e45a3a_b.jpg)

## <span id="head7"> 梯度消失与梯度爆炸</span>

- 梯度消失：主要是网络层数过多，或使用了不合适的激活函数，如Sigmoid导致的每层激活函数都会缩放到一个-1,1的区间，都乘以这个数字，因此会越乘越小。
  
  - 解决方法：使用残差网络，Relu激活函数，等都可以改善这个问题。
  
- 梯度爆炸：一般出现在**深层网络**和**权值初始化值太大**的情况下，在深层神经网络或循环神经网络中，**误差的梯度可在更新中累积相乘**。如果网络层之间的**梯度值大于 1.0**，那么**重复相乘会导致梯度呈指数级增长**，梯度变的非常大，然后导致网络权重的大幅更新，并因此使网络变得不稳定。

  - 解决方法：使用参数初始化和梯度裁剪

  

## <span id="head8"> loss权重设计</span>

- 反传过程中，是 loss 对 参数求偏导，乘以-的lr，加上原来的参数，也就是说，如果给loss 加一个权重，如果给一个大权重，最终这个权重相当于乘到了lr上，也就是加大了lr。需要lr大的可以加大的loss权重，需要学的慢一点就加小的权重。



## <span id="head9"> 多标签分类，二分类，多分类</span>

- 多标签分类和 二分类，用sigmod+bceloss
- 多分类用 softmax + crossentropyloss，或linear层加crossentropyloss
- 多分类一般认为分类的目标只有一个，而多标签分类用于分类的目标多个，如在yolov3中将多分类loss 改成了多标签分类loss，考虑到在同一个框中的物体，可能会属于多个类别。

## <span id="head10"> Batchnorm</span>

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



## <span id="head11"> SGD</span>

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



## <span id="head12"> MaxPooling</span>

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



## <span id="head13">Avg Pooling</span>

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
  print("out_numpy \n", out_numpy)
  print("out_tensor \n", out_tensor.data.numpy())
  
  print("dx_numpy \n", dx_numpy)
  print("dx_tensor \n", dx_tensor.data.numpy())
  ```

## <span id="head14"> CONV</span>

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



## <span id="head15"> BN</span>

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



## <span id="head16"> Loss函数</span>

- MSE Loss（均方损失函数）

  - ```python
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)　
    　#这里注意一下两个入参：
    
    reduce = False #返回向量形式的 loss　
    reduce = True  #返回标量形式的loss
    size_average = True #返回 loss.mean();
    如果 size_average = False  #返回 loss.sum()
    ```

  

- Cross Entropy Loss（交叉熵损失）

  - 交叉熵描述了两个概率分布之间的距离，当交叉熵越小说明二者之间越相似，反之则两者之间越不相似。

    ![img](https://www.zhihu.com/equation?tex=H%28p%2Cq%29%3D-%5Csum_%7Bx%7D%28p%28x%29logq%28x%29%2B%281-p%28x%29%29log%281-q%28x%29%29%29)

    

- Dice Loss

  - Dice系数原理：一种集合相似度度量指标，通常用于计算两个样本的相似度，值阈为[0, 1]。计算公式如下：

    ![img](https://www.zhihu.com/equation?tex=Dice+%3D+%5Cfrac%7B2+%2A+%28pred+%5Cbigcap+true%29%7D%7Bpred+%5Cbigcup+true%7D)

  - Dice Loss原理:

    ![img](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjPSX60xFEp8lg3jKtBVlR05wafmAiaS6PuIOGzgVvFOhH8TicgWW8Qpdj8GUud7pxSgXTR9WIoOtw4Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

  - 对正负样本严重不平衡的场景有着不错的性能

  - 一种区域相关的loss，像素点的loss以及梯度值不仅和该点的label以及预测值相关，和其他点的label以及预测值也相关

  - https://www.freesion.com/article/799887071/

  

- Soft IoU Loss

  - Iou（交并比）

    ![img](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjPSX60xFEp8lg3jKtBVlR05E34wM1ICoMGRINPMe9Of5Q2hUZOmicIofueDUPHmTXHMguyPgygia1mQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

  - 通过 IoU 计算损失也是使用预测的概率值：

    ​	![img](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjPSX60xFEp8lg3jKtBVlR05jaIbjF940lEzCN0ic5ceMSjZScERjWk2cYommIKnv4Ij7kzeRv5z50g/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

    其中 C 表示总的类别数。

    

- Weighted Loss（带权重的损失）

  - 对输出概率分布向量中的每个值进行加权，即希望模型更加关注数量较少的样本，以缓解分类时存在的类别不均衡问题。

    ![img](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjPSX60xFEp8lg3jKtBVlR05AVtWrVNx353bUibMfMrg4iaKjCzuhibRZKQA0oxA2WAxl4XOKMsHIg86A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

    要减少假阴性样本的数量，可以增大 pos_weight；要减少假阳性样本的数量，可以减小 pos_weight。

    

- Focal Loss

  - 容易学习的样本模型可以很轻松地将其预测正确，模型只要将大量容易学习的样本分类正确，loss就可以减小很多，从而导致模型不怎么顾及难学习的样本，所以我们要想办法让模型更加关注难学习的样本。对于较难学习的样本，将 BCE Loss 修改为：

    ![img](https://mmbiz.qpic.cn/mmbiz_png/AIR6eRePgjPSX60xFEp8lg3jKtBVlR05b5u6kErt8FDfEDgjYMOP3RVxkN6wbKttKvXWD8Ybvn2I44cibicI2aBQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)
    $$
    一般 \gamma = 2
    $$
    ​	

  - 解决了 正负样本不平衡、容易学习的样本和难学习样本的不平衡问题，使得模型更能学习到该学习的内容。

- [详解](https://mp.weixin.qq.com/s?__biz=MzIzNjc0MTMwMA==&mid=2247577739&idx=1&sn=857e16cc7f8fe1948280cc163357c829&chksm=e8d0c750dfa74e46eb9736a91022df47793c47d92fbd001c6f07252c125415be22f0a737657c&mpshare=1&srcid=0524cQ6zhkZyCZE26OB1SWoj&sharer_sharetime=1653387167819&sharer_shareid=0f803306403d0eb60321d56214c218ba&from=groupmessage&scene=1&subscene=10000&clicktime=1653387274&enterid=1653387274&ascene=1&devicetype=android-30&version=2800133d&nettype=WIFI&abtest_cookie=AAACAA%3D%3D&lang=zh_CN&exportkey=ATDqh7w1CkEHlXJttHjBpM8%3D&pass_ticket=me1jZHZiEj%2FGFp500c1L7RPOZmKLDYxuKDW3M8jEhX6w5dfgUiOG17%2Br51hoLVoZ&wx_header=3)

  

## <span id="head17"> 深度度量学习</span>

​	目的：学习数据分布，拉近类内距离，增大类间距离，得到一个嵌入空间(embedding space)的映射。

- Contractive Loss

  - 由两个样本组成的样本对

    ![img](https://www.zhihu.com/equation?tex=L%28x_i%2Cx_j%3Bf%29%3D%7B%5Cbf+1%7D%5C%7By_i%3Dy_j%5C%7D++%5C%7Cf_i-f_j%5C%7C_2%5E2+%2B+%7B%5Cbf+1%7D%5C%7By_i+%5Cneq+y_j%5C%7D+max%280%2C+m-%5C%7Cf_i-f_j%5C%7C_2%29%5E2)

  - 当两个样本同类时，loss小，当两个样本不同类时，loss大。

- Triplet Loss

  - 输入为三元组，一个为anchor，成为锚点，一个为正样本，一个为负样本。
  - ![img](https://www.zhihu.com/equation?tex=L%28x%2Cx%5E%2B%2Cx%5E-%3Bf%29%3Dmax+%5Cleft%280%2C+%5C%7Cf-f%5E%2B%5C%7C_2%5E2+-%5C%7Cf-f%5E-%5C%7C_2%5E2+%2B+m+%5Cright%29)

- MS Loss

  - 通过定义自相似性和相对相似性，在训练过程中更加全面地考虑了局部样本分布，从而能更高效精确的对重要样本对进行采用和加权。
  - ![img](https://www.zhihu.com/equation?tex=L_%7BMS%7D%3D%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Em+%5Cleft%5C%7B%5Cfrac%7B1%7D%7B%5Calpha%7Dlog%5B1%2B%5Csum_%7Bk+%5Cin+P_i%7De%5E%7B-%5Calpha%28S_%7Bik%7D-%5Clambda%29%7D%5D+%2B+%5Cfrac%7B1%7D%7B%5Cbeta%7Dlog%5B1%2B%5Csum_%7Bk+%5Cin+N_i%7De%5E%7B%5Cbeta%28S_%7Bik%7D-%5Clambda%29%7D%5D+%5Cright%5C%7D)

- https://zhuanlan.zhihu.com/p/82199561



## <span id="head18"> 解决样本不均衡方法</span>

- 数据增强：通过对数据集做各种操作以增加数量少的样本。
- Loss函数：采用weighted loss 或 focal loss
- sample：采样是指对图像像素点的选择/拒绝,是一种空间操作,可以使用采样上采样或下采样)来增大或者缩小图像。

# <span id="head19">CV note</span>

## <span id="head20"> 分组卷积</span>



## <span id="head21">空洞卷积(Dilated Convolution)</span>

## <span id="head22"> FPN（特征金字塔网络)</span>

- 先下采样得到多层特征，用深层特征与做上采样，与之前同等大小的底层feature融合，因此金字塔结构可以既包含底层语义又包含高级语义
- https://zhuanlan.zhihu.com/p/397293649

## <span id="head23"> 金字塔与SSD的区别</span>

- 金字塔是多个不同层的特征融合成一个size，SSD是不融合，直接用多个size去做预测





