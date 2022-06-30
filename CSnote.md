@[TOC](文章目录)

# linux基础

## linux常用命令

```
service NetworkManager restart //没网时重启网络管理器
ifconfig //配置SSH获取Ip地址
```

## linux函数

- open函数：

  ```c++
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  // 打开一个已经存在的文件
      int open(const char *pathname, int flags);
          /* 参数：
              - pathname：要打开的文件路径
              - flags：对文件的操作权限设置还有其他的设置
                O_RDONLY,  O_WRONLY,  O_RDWR  这三个设置是互斥的
          返回值：返回一个新的文件描述符，如果调用失败，返回-1
  
      errno：属于Linux系统函数库，库里面的一个全局变量，记录的是最近的错误号。
  	*/
      #include <stdio.h>
      void perror(const char *s);作用：打印errno对应的错误描述
          //s参数：用户描述，比如hello,最终输出的内容是  hello:xxx(实际的错误描述)
   
    //创建一个新文件
    int open(const char *pathname, int flags, mode_t mode);
          /*参数：
              - pathname：要创建的文件的路径
              - flags：对文件的操作权限和其他的设置
                  - 必选项：O_RDONLY,  O_WRONLY, O_RDWR  这三个之间是互斥的
                  - 可选项：O_CREAT 文件不存在，创建新文件
              - mode：八进制的数，表示创建出的新的文件的操作权限，比如：0775
              最终的权限是：mode & ~umask
              0777   ->   111111111
          &   0775   ->   111111101
          ----------------------------
                          111111101
          按位与：0和任何数都为0
          umask的作用就是抹去某些权限。
  
          flags参数是一个int类型的数据，占4个字节，32位。
          flags 32个位，每一位就是一个标志位。
      */
  
  注意：flags是程序对文件操作的权限，而mode是文件本身的权限。
  ```

- read 和 write 函数

  ```c++
  #include <unistd.h>
      ssize_t read(int fd, void *buf, size_t count);
          参数：
              - fd：文件描述符，open得到的，通过这个文件描述符操作某个文件
              - buf：需要读取数据存放的地方，数组的地址（传出参数）
              - count：指定的数组的大小
          返回值：
              - 成功：
                  >0: 返回实际的读取到的字节数
                  =0：文件已经读取完了
              - 失败：-1 ，并且设置errno
  
      #include <unistd.h>
      ssize_t write(int fd, const void *buf, size_t count);
          参数：
              - fd：文件描述符，open得到的，通过这个文件描述符操作某个文件
              - buf：要往磁盘写入的数据，数据
              - count：要写的数据的实际的大小
          返回值：
              成功：实际写入的字节数
              失败：返回-1，并设置errno
  */
  ```

- lseek函数

  ```c++
  /*  
      标准C库的函数
      #include <stdio.h>
      int fseek(FILE *stream, long offset, int whence);
  
      Linux系统函数
      #include <sys/types.h>
      #include <unistd.h>
      off_t lseek(int fd, off_t offset, int whence);
          参数：
              - fd：文件描述符，通过open得到的，通过这个fd操作某个文件
              - offset：偏移量
              - whence:
                  SEEK_SET
                      设置文件指针的偏移量
                  SEEK_CUR
                      设置偏移量：当前位置 + 第二个参数offset的值
                  SEEK_END
                      设置偏移量：文件大小 + 第二个参数offset的值
          返回值：返回文件指针的位置
  
  
      作用：
          1.移动文件指针到文件头
          lseek(fd, 0, SEEK_SET);
  
          2.获取当前文件指针的位置
          lseek(fd, 0, SEEK_CUR);
  
          3.获取文件长度
          lseek(fd, 0, SEEK_END);
  
          4.拓展文件的长度，当前文件10b, 110b, 增加了100个字节
          lseek(fd, 100, SEEK_END)
          注意：需要写一次数据
  
  */
  ```

- statu 函数

  ```c
  /*
      #include <sys/types.h>
      #include <sys/stat.h>
      #include <unistd.h>
  
      int stat(const char *pathname, struct stat *statbuf);
          作用：获取一个文件相关的一些信息
          参数:
              - pathname：操作的文件的路径
              - statbuf：结构体变量，传出参数，用于保存获取到的文件的信息
          返回值：
              成功：返回0
              失败：返回-1 设置errno
  
      int lstat(const char *pathname, struct stat *statbuf);
          参数:
              - pathname：操作的文件的路径
              - statbuf：结构体变量，传出参数，用于保存获取到的文件的信息
          返回值：
              成功：返回0
              失败：返回-1 设置errno
  
  */
  ```

- 文件属性操作函数

  ```c++
  /*
      #include <unistd.h>
      int access(const char *pathname, int mode);
          作用：判断某个文件是否有某个权限，或者判断文件是否存在
          参数：
              - pathname: 判断的文件路径
              - mode:
                  R_OK: 判断是否有读权限
                  W_OK: 判断是否有写权限
                  X_OK: 判断是否有执行权限
                  F_OK: 判断文件是否存在
          返回值：成功返回0， 失败返回-1
  */
  
  /*
      #include <sys/stat.h>
      int chmod(const char *pathname, mode_t mode);
          修改文件的权限
          参数：
              - pathname: 需要修改的文件的路径
              - mode:需要修改的权限值，八进制的数
          返回值：成功返回0，失败返回-1
  
  */
  
  /*
      #include <unistd.h>
      #include <sys/types.h>
      int truncate(const char *path, off_t length);
          作用：缩减或者扩展文件的尺寸至指定的大小
          参数：
              - path: 需要修改的文件的路径
              - length: 需要最终文件变成的大小
          返回值：
              成功返回0， 失败返回-1
  */
  ```

- 目录操作函数

  ```c++
  /*
      #include <unistd.h>
      int chdir(const char *path);
          作用：修改进程的工作目录
              比如在/home/nowcoder 启动了一个可执行程序a.out, 进程的工作目录 /home/nowcoder
          参数：
              path : 需要修改的工作目录
  
      #include <unistd.h>
      char *getcwd(char *buf, size_t size);
          作用：获取当前工作目录
          参数：
              - buf : 存储的路径，指向的是一个数组（传出参数）
              - size: 数组的大小
          返回值：
              返回的指向的一块内存，这个数据就是第一个参数
  */
  
  /*
      #include <sys/stat.h>
      #include <sys/types.h>
      int mkdir(const char *pathname, mode_t mode);
          作用：创建一个目录
          参数：
              pathname: 创建的目录的路径
              mode: 权限，八进制的数
          返回值：
              成功返回0， 失败返回-1
  */
  
  /*
      #include <stdio.h>
      int rename(const char *oldpath, const char *newpath);
      	作用：改文件名字
  
  */
  ```

- 目录遍历函数：

  ```c++
  /*
      // 打开一个目录
      #include <sys/types.h>
      #include <dirent.h>
      DIR *opendir(const char *name);
          参数：
              - name: 需要打开的目录的名称
          返回值：
              DIR * 类型，理解为目录流
              错误返回NULL
  
  
      // 读取目录中的数据
      #include <dirent.h>
      struct dirent *readdir(DIR *dirp);
          - 参数：dirp是opendir返回的结果
          - 返回值：
              struct dirent，代表读取到的文件的信息
              读取到了末尾或者失败了，返回NULL
  
      // 关闭目录
      #include <sys/types.h>
      #include <dirent.h>
      int closedir(DIR *dirp);
  
  */
  ```

- dup和dup2函数

  ```c++
  /*
      #include <unistd.h>
      int dup(int oldfd);
          作用：复制一个新的文件描述符
          fd=3, int fd1 = dup(fd),
          fd指向的是a.txt, fd1也是指向a.txt
          从空闲的文件描述符表中找一个最小的，作为新的拷贝的文件描述符
  
  
  */
  
  /*
      #include <unistd.h>
      int dup2(int oldfd, int newfd);
          作用：重定向文件描述符
          oldfd 指向 a.txt, newfd 指向 b.txt
          调用函数成功后：newfd 和 b.txt 做close, newfd 指向了 a.txt
          oldfd 必须是一个有效的文件描述符
          oldfd和newfd值相同，相当于什么都没有做
  */
  ```

- fcntl函数

  ```c++
  /*
      #include <unistd.h>
      #include <fcntl.h>
  
      int fcntl(int fd, int cmd, ...);
      参数：
          fd : 表示需要操作的文件描述符
          cmd: 表示对文件描述符进行如何操作
              - F_DUPFD : 复制文件描述符,复制的是第一个参数fd，得到一个新的文件描述符（返回值）
                  int ret = fcntl(fd, F_DUPFD);
  
              - F_GETFL : 获取指定的文件描述符文件状态flag
                获取的flag和我们通过open函数传递的flag是一个东西。
  
              - F_SETFL : 设置文件描述符文件状态flag
                必选项：O_RDONLY, O_WRONLY, O_RDWR 不可以被修改
                可选性：O_APPEND, O)NONBLOCK
                  O_APPEND 表示追加数据
                  NONBLOK 设置成非阻塞
          
          阻塞和非阻塞：描述的是函数调用的行为。
  */
  ```

- fork函数

  ```c++
  /*
      #include <sys/types.h>
      #include <unistd.h>
  
      pid_t fork(void);
          函数的作用：用于创建子进程。
          返回值：
              fork()的返回值会返回两次。一次是在父进程中，一次是在子进程中。
              在父进程中返回创建的子进程的ID,
              在子进程中返回0
              如何区分父进程和子进程：通过fork的返回值。
              在父进程中返回-1，表示创建子进程失败，并且设置errno
  
          父子进程之间的关系：
          区别：
              1.fork()函数的返回值不同
                  父进程中: >0 返回的子进程的ID
                  子进程中: =0
              2.pcb中的一些数据
                  当前的进程的id pid
                  当前的进程的父进程的id ppid
                  信号集
  
          共同点：
              某些状态下：子进程刚被创建出来，还没有执行任何的写数据的操作
                  - 用户区的数据
                  - 文件描述符表
          
          父子进程对变量是不是共享的？
              - 刚开始的时候，是一样的，共享的。如果修改了数据，不共享了。
              - 读时共享（子进程被创建，两个进程没有做任何的写的操作），写时拷贝。
          
  */
  ```

- execl函数

  ```
  execl：执行函数
  execlp：按PATH指定的目录搜索可执行文件
  ```

- exit 函数

  ```
  // C语言标准库，进程退出时会调用调用_exit(0)
  exit(0)
  ```

- wait 函数

  ```c++
  /*
      #include <sys/types.h>
      #include <sys/wait.h>
      pid_t wait(int *wstatus);
          功能：等待任意一个子进程结束，如果任意一个子进程结束了，次函数会回收子进程的资源。
          参数：int *wstatus
              进程退出时的状态信息，传入的是一个int类型的地址，传出参数。
          返回值：
              - 成功：返回被回收的子进程的id
              - 失败：-1 (所有的子进程都结束，调用函数失败)
  
      调用wait函数的进程会被挂起（阻塞），直到它的一个子进程退出或者收到一个不能被忽略的信号时才被唤醒（相当于继续往下执行）
      如果没有子进程了，函数立刻返回，返回-1；如果子进程都已经结束了，也会立即返回，返回-1.
  
  */
  ```

- waitpid 函数

  ```c++
  /*
      #include <sys/types.h>
      #include <sys/wait.h>
      pid_t waitpid(pid_t pid, int *wstatus, int options);
          功能：回收指定进程号的子进程，可以设置是否阻塞。
          参数：
              - pid:
                  pid > 0 : 某个子进程的pid
                  pid = 0 : 回收当前进程组的所有子进程    
                  pid = -1 : 回收所有的子进程，相当于 wait()  （最常用）
                  pid < -1 : 某个进程组的组id的绝对值，回收指定进程组中的子进程
              - options：设置阻塞或者非阻塞
                  0 : 阻塞
                  WNOHANG : 非阻塞
              - 返回值：
                  > 0 : 返回子进程的id
                  = 0 : options=WNOHANG, 表示还有子进程或者
                  = -1 ：错误，或者没有子进程了
  */
  ```

- #### pipe 函数

  ```c++
  /*
      #include <unistd.h>
      int pipe(int pipefd[2]);
          功能：创建一个匿名管道，用来进程间通信。
          参数：int pipefd[2] 这个数组是一个传出参数。
              pipefd[0] 对应的是管道的读端
              pipefd[1] 对应的是管道的写端
          返回值：
              成功 0
              失败 -1
  
      管道默认是阻塞的：如果管道中没有数据，read阻塞，如果管道满了，write阻塞
  
      注意：匿名管道只能用于具有关系的进程之间的通信（父子进程，兄弟进程）
      
      管道读写特点：
      读管道：
          管道中有数据，read返回实际读到的字节数。
          管道中无数据：
              写端被全部关闭，read返回0（相当于读到文件的末尾）
              写端没有完全关闭，read阻塞等待
  
      写管道：
          管道读端全部被关闭，进程异常终止（进程收到SIGPIPE信号）
          管道读端没有全部关闭：
              管道已满，write阻塞
              管道没有满，write将数据写入，并返回实际写入的字节数
  */
  ```

- #### mkfifo 函数

  ```c++
  /*
      创建fifo文件
      1.通过命令： mkfifo 名字
      2.通过函数：int mkfifo(const char *pathname, mode_t mode);
  
      #include <sys/types.h>
      #include <sys/stat.h>
      int mkfifo(const char *pathname, mode_t mode);
          参数：
              - pathname: 管道名称的路径
              - mode: 文件的权限 和 open 的 mode 是一样的
                      是一个八进制的数
          返回值：成功返回0，失败返回-1，并设置错误号
  
  */
  ```

- #### mmap 和 munmap函数

  ```c++
  /*
      #include <sys/mman.h>
      void *mmap(void *addr, size_t length, int prot, int flags,int fd, off_t offset);
          - 功能：将一个文件或者设备的数据映射到内存中
          - 参数：
              - void *addr: NULL, 由内核指定
              - length : 要映射的数据的长度，这个值不能为0。建议使用文件的长度。
                      获取文件的长度：stat lseek
              - prot : 对申请的内存映射区的操作权限
                  -PROT_EXEC ：可执行的权限
                  -PROT_READ ：读权限
                  -PROT_WRITE ：写权限
                  -PROT_NONE ：没有权限
                  要操作映射内存，必须要有读的权限。
                  PROT_READ、PROT_READ|PROT_WRITE
              - flags :
                  - MAP_SHARED : 映射区的数据会自动和磁盘文件进行同步，进程间通信，必须要设置这个选项
                  - MAP_PRIVATE ：不同步，内存映射区的数据改变了，对原来的文件不会修改，会重新创建一个新的文件。（copy on write）
              - fd: 需要映射的那个文件的文件描述符
                  - 通过open得到，open的是一个磁盘文件
                  - 注意：文件的大小不能为0，open指定的权限不能和prot参数有冲突。
                      prot: PROT_READ                open:只读/读写 
                      prot: PROT_READ | PROT_WRITE   open:读写
              - offset：偏移量，一般不用。必须指定的是4k的整数倍，0表示不便宜。
          - 返回值：返回创建的内存的首地址
              失败返回MAP_FAILED，(void *) -1
  
      int munmap(void *addr, size_t length);
          - 功能：释放内存映射
          - 参数：
              - addr : 要释放的内存的首地址
              - length : 要释放的内存的大小，要和mmap函数中的length参数的值一样。
  */
  ```

  PS：内存映射可以用来进程间的通信以及copy函数

- #### kill、raise、abort 函数

  ```c++
  /*  
      #include <sys/types.h>
      #include <signal.h>
  
      int kill(pid_t pid, int sig);
          - 功能：给任何的进程或者进程组pid, 发送任何的信号 sig
          - 参数：
              - pid ：
                  > 0 : 将信号发送给指定的进程
                  = 0 : 将信号发送给当前的进程组
                  = -1 : 将信号发送给每一个有权限接收这个信号的进程
                  < -1 : 这个pid=某个进程组的ID取反 （-12345）
              - sig : 需要发送的信号的编号或者是宏值，0表示不发送任何信号
  
          kill(getppid(), 9);
          kill(getpid(), 9);
          
      int raise(int sig);
          - 功能：给当前进程发送信号
          - 参数：
              - sig : 要发送的信号
          - 返回值：
              - 成功 0
              - 失败 非0
          kill(getpid(), sig);   
  
      void abort(void);
          - 功能： 发送SIGABRT信号给当前的进程，杀死当前进程
          kill(getpid(), SIGABRT);
  */
  ```

- alarm 函数

  ```c++
  /*
      #include <unistd.h>
      unsigned int alarm(unsigned int seconds);
          - 功能：设置定时器（闹钟）。函数调用，开始倒计时，当倒计时为0的时候，
                  函数会给当前的进程发送一个信号：SIGALARM
          - 参数：
              seconds: 倒计时的时长，单位：秒。如果参数为0，定时器无效（不进行倒计时，不发信号）。
                      取消一个定时器，通过alarm(0)。
          - 返回值：
              - 之前没有定时器，返回0
              - 之前有定时器，返回之前的定时器剩余的时间
  
      - SIGALARM ：默认终止当前的进程，每一个进程都有且只有唯一的一个定时器。
          alarm(10);  -> 返回0
          过了1秒
          alarm(5);   -> 返回9
  
      alarm(100) -> 该函数是不阻塞的
  */
  ```

- #### setitimer 函数

  ```c++
  /*
      #include <sys/time.h>
      int setitimer(int which, const struct itimerval *new_value,
                          struct itimerval *old_value);
      
          - 功能：设置定时器（闹钟）。可以替代alarm函数。精度微妙us，可以实现周期性定时
          - 参数：
              - which : 定时器以什么时间计时
                ITIMER_REAL: 真实时间，时间到达，发送 SIGALRM   常用
                ITIMER_VIRTUAL: 用户时间，时间到达，发送 SIGVTALRM
                ITIMER_PROF: 以该进程在用户态和内核态下所消耗的时间来计算，时间到达，发送 SIGPROF
  
              - new_value: 设置定时器的属性
              
                  struct itimerval {      // 定时器的结构体
                  struct timeval it_interval;  // 每个阶段的时间，间隔时间
                  struct timeval it_value;     // 延迟多长时间执行定时器
                  };
  
                  struct timeval {        // 时间的结构体
                      time_t      tv_sec;     //  秒数     
                      suseconds_t tv_usec;    //  微秒    
                  };
  
              过10秒后，每个2秒定时一次
             
              - old_value ：记录上一次的定时的时间参数，一般不使用，指定NULL
          
          - 返回值：
              成功 0
              失败 -1 并设置错误号
  */
  ```

- #### signal 函数

  ```c++
  /*
      #include <signal.h>
      typedef void (*sighandler_t)(int);
      sighandler_t signal(int signum, sighandler_t handler);
          - 功能：设置某个信号的捕捉行为
          - 参数：
              - signum: 要捕捉的信号
              - handler: 捕捉到信号要如何处理
                  - SIG_IGN ： 忽略信号
                  - SIG_DFL ： 使用信号默认的行为
                  - 回调函数 :  这个函数是内核调用，程序员只负责写，捕捉到信号后如何去处理信号。
                  回调函数：
                      - 需要程序员实现，提前准备好的，函数的类型根据实际需求，看函数指针的定义
                      - 不是程序员调用，而是当信号产生，由内核调用
                      - 函数指针是实现回调的手段，函数实现之后，将函数名放到函数指针的位置就可以了。
  
          - 返回值：
              成功，返回上一次注册的信号处理函数的地址。第一次调用返回NULL
              失败，返回SIG_ERR，设置错误号
              
      SIGKILL SIGSTOP不能被捕捉，不能被忽略。
  */
  ```

- #### 信号集函数

  ```c++
  /*
      以下信号集相关的函数都是对自定义的信号集进行操作。
  
      int sigemptyset(sigset_t *set);
          - 功能：清空信号集中的数据,将信号集中的所有的标志位置为0
          - 参数：set,传出参数，需要操作的信号集
          - 返回值：成功返回0， 失败返回-1
  
      int sigfillset(sigset_t *set);
          - 功能：将信号集中的所有的标志位置为1
          - 参数：set,传出参数，需要操作的信号集
          - 返回值：成功返回0， 失败返回-1
  
      int sigaddset(sigset_t *set, int signum);
          - 功能：设置信号集中的某一个信号对应的标志位为1，表示阻塞这个信号
          - 参数：
              - set：传出参数，需要操作的信号集
              - signum：需要设置阻塞的那个信号
          - 返回值：成功返回0， 失败返回-1
  
      int sigdelset(sigset_t *set, int signum);
          - 功能：设置信号集中的某一个信号对应的标志位为0，表示不阻塞这个信号
          - 参数：
              - set：传出参数，需要操作的信号集
              - signum：需要设置不阻塞的那个信号
          - 返回值：成功返回0， 失败返回-1
  
      int sigismember(const sigset_t *set, int signum);
          - 功能：判断某个信号是否阻塞
          - 参数：
              - set：需要操作的信号集
              - signum：需要判断的那个信号
          - 返回值：
              1 ： signum被阻塞
              0 ： signum不阻塞
              -1 ： 失败
  
  */
  ```

- #### sigprocmask 和sigpending 函数

  ```c++
  /*
      int sigprocmask(int how, const sigset_t *set, sigset_t *oldset);
          - 功能：将自定义信号集中的数据设置到内核中（设置阻塞，解除阻塞，替换）
          - 参数：
              - how : 如何对内核阻塞信号集进行处理
                  SIG_BLOCK: 将用户设置的阻塞信号集添加到内核中，内核中原来的数据不变
                      假设内核中默认的阻塞信号集是mask， mask | set
                  SIG_UNBLOCK: 根据用户设置的数据，对内核中的数据进行解除阻塞
                      mask &= ~set
                  SIG_SETMASK:覆盖内核中原来的值
              
              - set ：已经初始化好的用户自定义的信号集
              - oldset : 保存设置之前的内核中的阻塞信号集的状态，可以是 NULL
          - 返回值：
              成功：0
              失败：-1
                  设置错误号：EFAULT、EINVAL
  
      int sigpending(sigset_t *set);
          - 功能：获取内核中的未决信号集
          - 参数：set,传出参数，保存的是内核中的未决信号集中的信息。
  */
  ```

- #### 共享内存函数

  ```c++
  /*
  int shmget(key_t key, size_t size, int shmflg);
      - 功能：创建一个新的共享内存段，或者获取一个既有的共享内存段的标识。
          新创建的内存段中的数据都会被初始化为0
      - 参数：
          - key : key_t类型是一个整形，通过这个找到或者创建一个共享内存。
                  一般使用16进制表示，非0值
          - size: 共享内存的大小
          - shmflg: 属性
              - 访问权限
              - 附加属性：创建/判断共享内存是不是存在
                  - 创建：IPC_CREAT
                  - 判断共享内存是否存在： IPC_EXCL , 需要和IPC_CREAT一起使用
                      IPC_CREAT | IPC_EXCL | 0664
          - 返回值：
              失败：-1 并设置错误号
              成功：>0 返回共享内存的引用的ID，后面操作共享内存都是通过这个值。
  
  
  void *shmat(int shmid, const void *shmaddr, int shmflg);
      - 功能：和当前的进程进行关联
      - 参数：
          - shmid : 共享内存的标识（ID）,由shmget返回值获取
          - shmaddr: 申请的共享内存的起始地址，指定NULL，内核指定
          - shmflg : 对共享内存的操作
              - 读 ： SHM_RDONLY, 必须要有读权限
              - 读写： 0
      - 返回值：
          成功：返回共享内存的首（起始）地址。  失败(void *) -1
  
  
  int shmdt(const void *shmaddr);
      - 功能：解除当前进程和共享内存的关联
      - 参数：
          shmaddr：共享内存的首地址
      - 返回值：成功 0， 失败 -1
  
  int shmctl(int shmid, int cmd, struct shmid_ds *buf);
      - 功能：对共享内存进行操作。删除共享内存，共享内存要删除才会消失，创建共享内存的进行被销毁了对共享内存是没有任何影响。
      - 参数：
          - shmid: 共享内存的ID
          - cmd : 要做的操作
              - IPC_STAT : 获取共享内存的当前的状态
              - IPC_SET : 设置共享内存的状态
              - IPC_RMID: 标记共享内存被销毁
          - buf：需要设置或者获取的共享内存的属性信息
              - IPC_STAT : buf存储数据
              - IPC_SET : buf中需要初始化数据，设置到内核中
              - IPC_RMID : 没有用，NULL
  
  key_t ftok(const char *pathname, int proj_id);
      - 功能：根据指定的路径名，和int值，生成一个共享内存的key
      - 参数：
          - pathname:指定一个存在的路径
              /home/nowcoder/Linux/a.txt
              / 
          - proj_id: int类型的值，但是这系统调用只会使用其中的1个字节
                     范围 ： 0-255  一般指定一个字符 'a'
  
  
  问题1：操作系统如何知道一块共享内存被多少个进程关联？
      - 共享内存维护了一个结构体struct shmid_ds 这个结构体中有一个成员 shm_nattch
      - shm_nattach 记录了关联的进程个数
  
  问题2：可不可以对共享内存进行多次删除 shmctl
      - 可以的
      - 因为shmctl 标记删除共享内存，不是直接删除
      - 什么时候真正删除呢?
          当和共享内存关联的进程数为0的时候，就真正被删除
      - 当共享内存的key为0的时候，表示共享内存被标记删除了
          如果一个进程和共享内存取消关联，那么这个进程就不能继续操作这个共享内存。也不能进行关联。
  
      共享内存和内存映射的区别
      1.共享内存可以直接创建，内存映射需要磁盘文件（匿名映射除外）
      2.共享内存效果更高
      3.内存
          所有的进程操作的是同一块共享内存。
          内存映射，每个进程在自己的虚拟地址空间中有一个独立的内存。
      4.数据安全
          - 进程突然退出
              共享内存还存在
              内存映射区消失
          - 运行进程的电脑死机，宕机了
              数据存在在共享内存中，没有了
              内存映射区的数据 ，由于磁盘文件中的数据还在，所以内存映射区的数据还存在。
  
      5.生命周期
          - 内存映射区：进程退出，内存映射区销毁
          - 共享内存：进程退出，共享内存还在，标记删除（所有的关联的进程数为0），或者关机
              如果一个进程退出，会自动和共享内存进行取消关联。
  */
  ```

- #### pthread_create 函数

  ```c++
  /*
      一般情况下,main函数所在的线程我们称之为主线程（main线程），其余创建的线程
      称之为子线程。
      程序中默认只有一个进程，fork()函数调用，2进行
      程序中默认只有一个线程，pthread_create()函数调用，2个线程。
  
      #include <pthread.h>
      int pthread_create(pthread_t *thread, const pthread_attr_t *attr, 
      void *(*start_routine) (void *), void *arg);
  
          - 功能：创建一个子线程
          - 参数：
              - thread：传出参数，线程创建成功后，子线程的线程ID被写到该变量中。
              - attr : 设置线程的属性，一般使用默认值，NULL
              - start_routine : 函数指针，这个函数是子线程需要处理的逻辑代码
              - arg : 给第三个参数使用，传参
          - 返回值：
              成功：0
              失败：返回错误号。这个错误号和之前errno不太一样。
              获取错误号的信息：  char * strerror(int errnum);
  
  */
  ```

- #### pthread_exit 函数

  ```c++
  /*
  
      #include <pthread.h>
      void pthread_exit(void *retval);
          功能：终止一个线程，在哪个线程中调用，就表示终止哪个线程
          参数：
              retval:需要传递一个指针，作为一个返回值，可以在pthread_join()中获取到。
  a
      pthread_t pthread_self(void);
          功能：获取当前的线程的线程ID
  
      int pthread_equal(pthread_t t1, pthread_t t2);
          功能：比较两个线程ID是否相等
          不同的操作系统，pthread_t类型的实现不一样，有的是无符号的长整型，有的
          是使用结构体去实现的。
  */
  ```

- #### 线程互斥函数 Mutex

  - ```c++
    /*
        互斥量的类型 pthread_mutex_t
        int pthread_mutex_init(pthread_mutex_t *restrict mutex, const pthread_mutexattr_t *restrict attr);
            - 初始化互斥量
            - 参数 ：
                - mutex ： 需要初始化的互斥量变量
                - attr ： 互斥量相关的属性，NULL
            - restrict : C语言的修饰符，被修饰的指针，不能由另外的一个指针进行操作。
                pthread_mutex_t *restrict mutex = xxx;
                pthread_mutex_t * mutex1 = mutex;
    
        int pthread_mutex_destroy(pthread_mutex_t *mutex);
            - 释放互斥量的资源
    
        int pthread_mutex_lock(pthread_mutex_t *mutex);
            - 加锁，阻塞的，如果有一个线程加锁了，那么其他的线程只能阻塞等待
    
        int pthread_mutex_trylock(pthread_mutex_t *mutex);
            - 尝试加锁，如果加锁失败，不会阻塞，会直接返回。
    
        int pthread_mutex_unlock(pthread_mutex_t *mutex);
            - 解锁
    */
    ```

- #### 读写锁函数

  ```c++
  /*
      读写锁的类型 pthread_rwlock_t
      int pthread_rwlock_init(pthread_rwlock_t *restrict rwlock, const pthread_rwlockattr_t *restrict attr);
      int pthread_rwlock_destroy(pthread_rwlock_t *rwlock);
      int pthread_rwlock_rdlock(pthread_rwlock_t *rwlock);
      int pthread_rwlock_tryrdlock(pthread_rwlock_t *rwlock);
      int pthread_rwlock_wrlock(pthread_rwlock_t *rwlock);
      int pthread_rwlock_trywrlock(pthread_rwlock_t *rwlock);
      int pthread_rwlock_unlock(pthread_rwlock_t *rwlock);
  
      案例：8个线程操作同一个全局变量。
      3个线程不定时写这个全局变量，5个线程不定时的读这个全局变量
  */
  ```

- #### 条件变量函数

  ```c++
  /*
      条件变量的类型 pthread_cond_t
      int pthread_cond_init(pthread_cond_t *restrict cond, const pthread_condattr_t *restrict attr);
      int pthread_cond_destroy(pthread_cond_t *cond);
      int pthread_cond_wait(pthread_cond_t *restrict cond, pthread_mutex_t *restrict mutex);
          - 等待，调用了该函数，线程会阻塞。
      int pthread_cond_timedwait(pthread_cond_t *restrict cond, pthread_mutex_t *restrict mutex, const struct timespec *restrict abstime);
          - 等待多长时间，调用了这个函数，线程会阻塞，直到指定的时间结束。
      int pthread_cond_signal(pthread_cond_t *cond);
          - 唤醒一个或者多个等待的线程
      int pthread_cond_broadcast(pthread_cond_t *cond);
          - 唤醒所有的等待的线程
  */c++
  ```

- #### select 函数

  ```c++
  // sizeof(fd_set) = 128 1024
  #include <sys/time.h>
  #include <sys/types.h>
  #include <unistd.h>
  #include <sys/select.h>
  int select(int nfds, fd_set *readfds, fd_set *writefds,
  fd_set *exceptfds, struct timeval *timeout);
  - 参数：
  - nfds : 委托内核检测的最大文件描述符的值 + 1
  - readfds : 要检测的文件描述符的读的集合，委托内核检测哪些文件描述符的读的属性
  - 一般检测读操作
  - 对应的是对方发送过来的数据，因为读是被动的接收数据，检测的就是读缓冲
  区
  - 是一个传入传出参数
  - writefds : 要检测的文件描述符的写的集合，委托内核检测哪些文件描述符的写的属性
  - 委托内核检测写缓冲区是不是还可以写数据（不满的就可以写）
  - exceptfds : 检测发生异常的文件描述符的集合
  - timeout : 设置的超时时间
  struct timeval {
  long tv_sec; /* seconds */
  long tv_usec; /* microseconds */
  };
  - NULL : 永久阻塞，直到检测到了文件描述符有变化
  - tv_sec = 0 tv_usec = 0， 不阻塞
  - tv_sec > 0 tv_usec > 0， 阻塞对应的时间
  - 返回值 :
  - -1 : 失败
  - >0(n) : 检测的集合中有n个文件描述符发生了变化
  // 将参数文件描述符fd对应的标志位设置为0
  void FD_CLR(int fd, fd_set *set);
  // 判断fd对应的标志位是0还是1， 返回值 ： fd对应的标志位的值，0，返回0， 1，返回1
  int FD_ISSET(int fd, fd_set *set);
  // 将参数文件描述符fd 对应的标志位，设置为1
  void FD_SET(int fd, fd_set *set);
  ```

- #### POLL 函数

  ```c++
  #include <poll.h>
  struct pollfd {
  int fd; /* 委托内核检测的文件描述符 */
  short events; /* 委托内核检测文件描述符的什么事件 */
  short revents; /* 文件描述符实际发生的事件 */
  };
  struct pollfd myfd;
  myfd.fd = 5;
  myfd.events = POLLIN | POLLOUT;
  int poll(struct pollfd *fds, nfds_t nfds, int timeout);
  - 参数：
  - fds : 是一个struct pollfd 结构体数组，这是一个需要检测的文件描述符的集合
  - nfds : 这个是第一个参数数组中最后一个有效元素的下标 + 1
  - timeout : 阻塞时长
  0 : 不阻塞
  -1 : 阻塞，当检测到需要检测的文件描述符有变化，解除阻塞
  >0 : 阻塞的时长
  - 返回值：
  -1 : 失败
  >0（n） : 成功,n表示检测到集合中有n个文件描述符发生变化
  ```

- #### POLL机制

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

- #### Epoll 函数

  ```c++
  #include <sys/epoll.h>
  // 创建一个新的epoll实例。在内核中创建了一个数据，这个数据中有两个比较重要的数据，一个是需要检
  测的文件描述符的信息（红黑树），还有一个是就绪列表，存放检测到数据发送改变的文件描述符信息（双向
  链表）。
  int epoll_create(int size);
  - 参数：
  size : 目前没有意义了。随便写一个数，必须大于0
  - 返回值：
  -1 : 失败
  > 0 : 文件描述符，操作epoll实例的
  
      typedef union epoll_data {
  void *ptr;
  int fd;
  uint32_t u32;
  uint64_t u64;
  } epoll_data_t;
  struct epoll_event {
  uint32_t events; /* Epoll events */
  epoll_data_t data; /* User data variable */
  };
  常见的Epoll检测事件：
  - EPOLLIN
  - EPOLLOUT
  - EPOLLERR
  // 对epoll实例进行管理：添加文件描述符信息，删除信息，修改信息
  int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
  - 参数：
  - epfd : epoll实例对应的文件描述符
  - op : 要进行什么操作
  EPOLL_CTL_ADD: 添加
  EPOLL_CTL_MOD: 修改
  EPOLL_CTL_DEL: 删除
  - fd : 要检测的文件描述符
  - event : 检测文件描述符什么事情
  // 检测函数
  int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int
  timeout);
  - 参数：
  - epfd : epoll实例对应的文件描述符
  - events : 传出参数，保存了发送了变化的文件描述符的信息
  - maxevents : 第二个参数结构体数组的大小
  - timeout : 阻塞时间
  - 0 : 不阻塞
  - -1 : 阻塞，直到检测到fd数据发生变化，解除阻塞
  - > 0 : 阻塞的时长（毫秒）
  - 返回值：
  - 成功，返回发送变化的文件描述符的个数 > 0
  - 失败 -1
  ```




# 数据库

## 索引

- B+树：https://mp.weixin.qq.com/s/w1ZFOug8-Sa7ThtMnlaUtQ

  

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

  
  
  