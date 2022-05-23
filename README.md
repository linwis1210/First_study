# 机器学习基础

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

## 分组卷积



## 空洞卷积(Dilated Convolution)





# CV note

## [FPN](https://zhuanlan.zhihu.com/p/397293649)(特征金字塔网络)

- 先下采样得到多层特征，用深层特征与做上采样，与之前同等大小的底层feature融合，因此金字塔结构可以既包含底层语义又包含高级语义

## 金字塔与SSD的区别

- 金字塔是多个不同层的特征融合成一个size，SSD是不融合，直接用多个size去做预测





