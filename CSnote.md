# linux基础

- POLL机制

  - `poll`是linux的事件轮询机制函数，每个进程可以管理一个`pollfd`队列，由`poll`函数进行事件注册和查询。

  - ```c++
    struct pollfd {
      int   fd;         /* file descriptor */
      short events;     /* requested events */
      short revents;    /* returned events */
    };
    ```

  - fd是文件描述符，用来指示linux给当前pollfd分配的文件。编程时需要给`events`注册我们想要的事件，之后使用`poll`函数对pollfd队列进行轮询，轮询结束后，`revents`由内核设置为实际发生的事件。如果`fd`是负数，那么会忽略`events`，而且`revents`会置为0。

  - https://blog.csdn.net/qq_35976351/article/details/85108232



# 网络基础

## HTTP

<center><img src="https://img-blog.csdnimg.cn/6b9bfd38d2684b3f9843ebabf8771212.png" width="80%"></center>

### HTTP基本概念：

- HTTP 是超文本传输协议，也就是**H**yperText **T**ransfer **P**rotocol。

  - **HTTP 是一个在计算机世界里专门在「两点」之间「传输」文字、图片、音频、视频等「超文本」数据的「约定和规范」**

- GET 与 POST

  - GET：语义是请求获取指定的资源。GET 方法是安全、幂等、可被缓存的。
  - POST：是根据请求负荷（报文主体）对指定的资源做出处理，具体的处理方式视资源类型而不同。POST 不安全，不幂等，（大部分实现）不可缓存。

  - 

  
  
  