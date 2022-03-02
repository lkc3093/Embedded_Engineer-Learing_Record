# **ML**
1. *relu* 优点
    * 当 *loss* 过大或者过小，*sigmoid*，*tanh* 的导数接近于0，*relu* 为非饱和激活函数不存在这种现象

2. resnet优点
   * 防止梯度消失

3. 数据预处理

4. *Batch normalization* 用于将数据的分布重整，标准化；数据变成正态分布，有效减少梯度消失

5. 动态调节学习率

6. *FSMN* 与 *RNN* 不同的是，每一层都保存的是前面时刻的feature，然后进行降维，输入到下一时刻的隐藏层中
  * *CFSMN* 将前一刻的feature降维了，减少参数；并只输出记忆模块参数
  * *DFSMN* 增加了跳跃输入，类似于resnet

------



## **算法优化**
* 深度可分离卷积，每个通道单独卷积，然后多个通道逐点点积
  * *MobileNet*
  
* 算子融合，前一个计算图的结果是后一个计算图的输入，可以融合，减少数据的 *load*，*store* 的操作
  * 可以通过 *tvm* 前端来实现
  
* 通过加 *L2* 正则，将输入的权重稀疏化，节省内存
  * 稀疏矩阵可以只访问非0值部分，减少指令读取的执行
  
* 向量指令加速
  * 内联，减少函数跳转，和堆栈开辟
  * 指令并行，无数据冲突的指令流水线执行
  * VILW，相当于多发射，指令槽
  
* 异构多核
  * 完成会报中断，类似于线程同步
  
* 量化
  * 定点数，int8
  * tensorflow训练中量化
  * 非对称量化更好
  
  ------



## TVM

### FrontEnd

  支持主流的深度学习前端框架，包括TensorFlow, MXNet, PyTorch, Keras, CNTK。目前TVM可以继承到PyTorch框架中优化、训练，而不是单纯地调用CNN模型接口

### Relay

  根据具体硬件对原始计算图进行重构、张量优化、数据重排等图优化操作。源代码分析中，Relay层比较杂，干的事情比较多，既对接上层的图优化又对接硬件的调度器



![img](https://iostream.io/wp-content/uploads/2019/09/Relay.png)



![img](https://iostream.io/wp-content/uploads/2019/09/Tensorlization.png)

​																									Relay及Tensorlization示例



### BackEnd

后端支持ARM、CUDA/Metal/OpenCL及加速器VTA。

1. 生成硬件所需指令流与数据打包
2. 一个CPU与VTA的交互式运行环境：包括driver、JIT。

## RTOS
* 线程切换类似于中断中的上下文保护
* 互斥量是对共享内存(全局变量之类)的保护，只有`0 / 1` 状态
* 信号量用于多线程同步
  * 线程获取一次信号量，信号量的值就会减1，当信号量的值减到0，再有线程获取信号量时，该线程就会被挂起到信号量的等待队列中，等待其他线程释放信号量


* 由于在多处理器环境中某些资源的有限性，有时需要互斥访问(`mutual exclusion`)，这时候就需要引入锁的概念，只有获取了锁的线程才能够对资源进行访问，由于多线程的核心是CPU的时间分片，所以同一时刻只能有一个线程获取到锁。那么就面临一个问题，那么没有获取到锁的线程应该怎么办？
  * 通常有两种处理方式：一种是没有获取到锁的线程就一直循环等待判断该资源是否已经释放锁，这种锁叫做 `自旋锁`，它不用将线程阻塞起来(`NON-BLOCKING)`；还有一种处理方式就是把自己阻塞起来，等待重新调度请求，这种叫做`互斥锁`
* 管道可以用于通信
* sizeof 指针是4个字节
* 1. 结构体变量的首地址能够被其最宽基本类型成员的大小所整除；
   2. 结构体每个成员相对结构体首地址的偏移量(offset)都是成员大小的整数倍，如有需编译器会在成员之间加上填充字节(internal adding)；
   3. 结构体的总大小为结构体最宽基本类型成员大小的整数倍，如有需要编译器会在最末一个成员之后加上填充字节{trailing padding}。

------



## OPENCL

* 实时编译，因此可以适应多种异构平台
<<<<<<< HEAD

=======
>>>>>>> 1a9f187bd4d3e73977928054dcbe46a83f0ac803
* 执行模式有任务并行和数据并行
  * 数据并行指的是不同的工作组执行同一个指令不同数据，通过`get_global_id` 来嵌入使用
  
  * 任务并行，多用于不同的 **kernel** 函数，以及队列命令乱序执行，当前命令没有对之前命令的依赖关系
    * 有序执行 (**in-order execution**)：命令按其在命令队列中出现的顺序发出，并按顺序完成。队列中前一个命令完成之后，下一个命令才会开始。这会将队列中命令的执行顺序串行化
    
    * 乱序执行 (**out-of-order execution**)：命令按顺序发出，但是下一个命令执行之前不会等待前一个命令完成。程序员要通过显式的同步机制来强制顺序
    
      
  
* **work-item**（工作项）：**work-item** 与 **cuda threads** 是一样的，是最小的执行单元。每次一个Kernel开始执行，很多（程序员定义数量）的 **work-item** 就开始运行，每个都执行同样的代码
  * 每个 **work-item** 有一个 id ，这个 id 在 kernel 中是可以访问的，每个运行在 **work-item **上的 kernel 通过这个 id 来找出**work-item**需要处理的数据
  
  * **work-item** 是循环中的最小单位
  
  * **work-item** 应该是由opencl自动分配给硬件线程的
  
    
  
* 存储器模式是软件定义的，而非单纯硬件实现；可以使用`constant` 来将数据放入常量存储区，访问更快

  * **Global memory:**工作区内的所有工作节点都可以自由的读写其中的任何数据。OpenCL C语言提供了全局缓存（Global buffer）的内建函数

  * **Constant memory:** 工作区内的所有工作节点可以读取其中的任何数据但不可以对数据内容进行更改，在内核程序的执行过程中保持不变。主机端负责分配和初始化常量缓存（Constant buffer）

  * **Local memory:** 只有同一工作组中的工作节点才可以对该类内存进行读写操作。它既可以为 OpenCL 的执行分配一块私有内存空间，也可以直接将其映射到一块全局缓存（**Global buffer**）上，特点是运行速度快

  * **Private memory:** 只有当前的工作节点( **item** )能对该内存进行访问和读写操作。一个工作节点内部的私有缓存（Private buffer）对其他节点来说是不可见的

    

* **CU** 对应cuda中的 **SM**，PE是SM中的一个处理单元

  * 一个work group的最大work item个数是指一个compute unit最多能调度、分配的线程数。这个数值一般就是一个CU内所包含的PE的个数的倍数。比如，如果一个GPU有2个CU，每个CU含有8个PE，而Max work group size是512，那么说明一个CU至少可以分配供512个线程并发操作所需要的各种资源。由于一个GPU根据一条算术逻辑指令能对所有PE发射若干次作为一个“原子的”发射操作，因此，这一个对程序员而言作为“原子的”发射操作启动了多少个线程，那么我们就可以认为是该GPU的最小并行线程数。如果一款GPU的最小线程并行数是32，那么该GPU将以32个线程作为一组原子的线程组。这意味着，如果遇到分支，那么一组32个线程组中的所有线程都将介入这个分支，对于不满足条件的线程，则会等到这32个线程中其它线程都完成分支处理之后再一起执行下面的指令。

    如果我将work group size指定为64，并且在kernel程序里加一个判断，如果pid小于32做操作A，否则做操作B，那么pid为0~31的线程组会执行操作A，而pid为32到63的线程组不会受到阻塞，而会立马执行操作B。此时，两组线程将 **并发** 操作（注意，这里是并发，而不是并行。因为上面讲过，GPU一次发射32个线程的话，那么对于多个32线程组将会调度发射指令）

  * 一个 work-item 不能被拆分到多个PE上处理；同样，一个 work-group 也不能拆分到多个CU上同时处理

  * 如果我想让 group 数量小点，那work-item的数目就会很多，还能不能处理了呢？以当前这个示例是能的，但是对于多的work-item,这涉及到如何确定work-item数目的问题

    结合cuda的概念进行解释：因为实际上，一个 SM 可以允许的 block 数量，还要另外考虑到他所用到 SM 的资源：shared memory、registers 等。在 G80 中，每个 SM 有 16KB 的 shared memory 和 8192 个 register。而在同一个 SM 里的 block 和 thread，则要共享这些资源;如果资源不够多个 block 使用的话，那 CUDA 就会减少 Block 的量，来让资源够用。在这种情形下，也会因此让 SM 的 thread 数量变少，而不到最多的 768 个

    比如说如果一个 thread 要用到 16 个 register 的话(在 kernel 中宣告的变量)，那一个 SM 的 8192 个 register 实际上只能让 512 个 thread 来使用;而如果一个 thread 要用 32 个 register，那一个 SM 就只能有 256 个 thread 了～而 shared memory 由于是 thread block 共享的，因此变成是要看一个 block 要用多少的 shared memory、一个 SM 的 16KB 能分给多少个 block 了

    所以虽然说当一个 SM 里的 thread 越多时，越能隐藏 latency，但是也会让每个 thread 能使用的资源更少。因此，这点也就是在优化时要做取舍的了
    
    

* kernel函数中间可以嵌套kernel函数，设备端可以创建命令队列，极大地方便了代码编写

  > 

  ````c
  #include <iostream>
  #include <stdlib.h>
  #include <string.h>
  #include <stdio.h>
   
   
  #if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/cl.hpp>
  #else
  #include <CL/cl.h>
  #endif
   
  using namespace std;
   
  #define KERNEL(...)#__VA_ARGS__
   
  const char *kernelSourceCode = KERNEL(
                                     __kernel void hellocl(__global uint *buffer)
  {
      size_t gidx = get_global_id(0);
      size_t gidy = get_global_id(1);
      size_t lidx = get_local_id(0);
      buffer[gidx + 4 * gidy] = (1 << gidx) | (0x10 << gidy);
   
  }
                                 );
  
  int main(int argc, char const *argv[])
  {
      printf("hello OpenCL\n");
      cl_int status = 0;
      size_t deviceListSize;
   
      // 当前服务器上配置的仅有NVIDIA Tesla C2050 的GPU
      cl_platform_id platform = NULL;
      status = clGetPlatformIDs(1, &platform, NULL);
   
      if (status != CL_SUCCESS) {
          printf("ERROR: Getting Platforms.(clGetPlatformIDs)\n");
          return EXIT_FAILURE;
      }
   
      // 如果我们能找到相应平台，就使用它，否则返回NULL
      cl_context_properties cps[3] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)platform,
          0
      };
   
      cl_context_properties *cprops = (NULL == platform) ? NULL : cps;
   
   
      // 生成 context
      cl_context context = clCreateContextFromType(
                               cprops,
                               CL_DEVICE_TYPE_GPU,
                               NULL,
                               NULL,
                               &status);
      if (status != CL_SUCCESS) {
          printf("Error: Creating Context.(clCreateContexFromType)\n");
          return EXIT_FAILURE;
      }
   
      // 寻找OpenCL设备
   
      // 首先得到设备列表的长度
      status = clGetContextInfo(context,
                                CL_CONTEXT_DEVICES,
                                0,
                                NULL,
                                &deviceListSize);
      if (status != CL_SUCCESS) {
          printf("Error: Getting Context Info device list size, clGetContextInfo)\n");
          return EXIT_FAILURE;
      }
      cl_device_id *devices = (cl_device_id *)malloc(deviceListSize);
      if (devices == 0) {
          printf("Error: No devices found.\n");
          return EXIT_FAILURE;
      }
   
      // 现在得到设备列表
      status = clGetContextInfo(context,
                                CL_CONTEXT_DEVICES,
                                deviceListSize,
                                devices,
                                NULL);
      if (status != CL_SUCCESS) {
          printf("Error: Getting Context Info (device list, clGetContextInfo)\n");
          return EXIT_FAILURE;
      }
   
   
      // 装载内核程序，编译CL program ,生成CL内核实例
   
      size_t sourceSize[] = {strlen(kernelSourceCode)};
      cl_program program = clCreateProgramWithSource(context,
                           1,
                           &kernelSourceCode,
                           sourceSize,
                           &status);
      if (status != CL_SUCCESS) {
          printf("Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n");
          return EXIT_FAILURE;
      }
   
      // 为指定的设备编译CL program.
      status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
      if (status != CL_SUCCESS) {
          printf("Error: Building Program (clBuildingProgram)\n");
          return EXIT_FAILURE;
      }
   
      // 得到指定名字的内核实例的句柄
      cl_kernel kernel = clCreateKernel(program, "hellocl", &status);
      if (status != CL_SUCCESS) {
          printf("Error: Creating Kernel from program.(clCreateKernel)\n");
          return EXIT_FAILURE;
      }
   
      // 创建 OpenCL buffer 对象
      unsigned int *outbuffer = new unsigned int [4 * 4];
      memset(outbuffer, 0, 4 * 4 * 4);
      cl_mem outputBuffer = clCreateBuffer(
          context, 
          CL_MEM_ALLOC_HOST_PTR, 
          4 * 4 * 4, 
          NULL, 
          &status);
   
      if (status != CL_SUCCESS) {
          printf("Error: Create Buffer, outputBuffer. (clCreateBuffer)\n");
          return EXIT_FAILURE;
      }
   
      //  为内核程序设置参数
      status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&outputBuffer);
      if (status != CL_SUCCESS) {
          printf("Error: Setting kernel argument. (clSetKernelArg)\n");
          return EXIT_FAILURE;
      }
   
      // 创建一个OpenCL command queue
      cl_command_queue commandQueue = clCreateCommandQueue(context,
                                      devices[0],
                                      0,
                                      &status);
      if (status != CL_SUCCESS) {
          printf("Error: Create Command Queue. (clCreateCommandQueue)\n");
          return EXIT_FAILURE;
      }
   
   
      // 将一个kernel 放入 command queue
      size_t globalThreads[] = {4, 4};
      size_t localThreads[] = {2, 2};
      status = clEnqueueNDRangeKernel(commandQueue, kernel,
                                      2, NULL, globalThreads,
                                      localThreads, 0,
                                      NULL, NULL);
      if (status != CL_SUCCESS) {
          printf("Error: Enqueueing kernel\n");
          return EXIT_FAILURE;
      }
   
      // 确认 command queue 中所有命令都执行完毕
      status = clFinish(commandQueue);
      if (status != CL_SUCCESS) {
          printf("Error: Finish command queue\n");
          return EXIT_FAILURE;
      }
   
      // 将内存对象中的结果读回Host
      status = clEnqueueReadBuffer(commandQueue,
                                   outputBuffer, CL_TRUE, 0,
                                   4 * 4 * 4, outbuffer, 0, NULL, NULL);
      if (status != CL_SUCCESS) {
          printf("Error: Read buffer queue\n");
          return EXIT_FAILURE;
      }
   
      // Host端打印结果
      printf("out:\n");
      for (int i = 0; i < 16; ++i) {
          printf("%x ", outbuffer[i]);
          if ((i + 1) % 4 == 0)
              printf("\n");
      }
   
      // 资源回收
      status = clReleaseKernel(kernel);
      status = clReleaseProgram(program);
      status = clReleaseMemObject(outputBuffer);
      status = clReleaseCommandQueue(commandQueue);
      status = clReleaseContext(context);
   
      free(devices);
      delete outbuffer;
      return 0;
  }
  ````



* 管道可以实现不同内核函数的交互

  

------



## CUDA

* 一个**SM** 一个时刻只能运行一个 **warp**，但是可以有多个warp停驻在一个 **SM** 里面
  * 一旦其中一个 **warp** 发生IO操作，立刻切换到下一个 **active wrap**

* **block** 可以设置 **warp** 的数量，最好

* **CUDA** 是有 **cache** 的概念，指令流水线；最新的架构甚至有多个的指令流水线( 按指令分类 )

  * 数据从全局内存到SM（**stream-multiprocessor**）的传输，会进行cache，如果 **cache **命中了，下一次的访问的耗时将大大减少。
    每个SM都具有单独的L1 cache，所有的SM共用一个L2 cache
    在计算能力2.x之前的设备，全局内存和局部内存的访问都会在 **L1\L2 cache** 上缓存；在计算能力3.x以上的设备，全局内存的访问只在**L2 cache**上缓存

    

* 访问连续的全局内存时，**GPU** 会合并访问，一次取出多个连续数据

  * 合并访问是指所有线程访问连续的对齐的内存块，对于L1 cache，内存块大小支持32字节、64字节以及128字节，分别表示线程束中每个线程以一个字节（1*32=32）、16位（2*32=64）、32位（4*32=128）为单位读取数据。前提是，访问必须连续，并且访问的地址是以32字节对齐。（类似于SSE\AVX的向量指令，cuda中的合并访存也是向量指令）

    例子，假设每个 **thread **读取一个float变量，那么一个**warp**（32个**thread**）将会执行32*4=128字节的合并访存指令，通过一次访存操作完成所有thread的读取请求

  * **Shared Memory**

    用 __shared __修饰符修饰的变量存放在shared memory。因为 shared memory 是on-chip的，他相比 localMemory 和 global memory 来说，拥有高的多bandwidth和低很多的latency。他的使用和CPU的 L1cache 非常类似，但是他是 programmable 的

    按惯例，像这类性能这么好的memory都是有限制的，shared memory是以block为单位分配的；我们必须非常小心的使用shared memory，否则会无意识的限制了active warp的数目

  * **Constant Memory**

    常量内存同样是offchip内存，只读，拥有SM私有的 **constant cache**，因此在cache hit的情况下速度快。常量内存是全局的，对所有 Kernel 函数可见。因此声明要在Kernel函数外

    

* 访问共享内存时，可能会出现 **bank** **conflict**，局部内存是一个个 **bank** 组成

* **local memory** 是线程私有的，寄存器不够，就会将数据存入 **local memory**

* **CUDA** 中的合并访存也是向量指令

* **CUDA** 可以使用洗牌指令实现线程内的数据交互，可以通过归约操作来完成一些累加的算法计算

  * 由于**CUDA** 的**warp** 执行的是同一条指令，导致会有部分是重复运算


* 当 block 中的线程数超过 SM 的上限，会导致 kernel 不运行；目前英伟达官方将块的上限设置为1024个线程
* 通过 **stream** 可以实现多个内核计算并行，或者计算与IO并行；流与流之间是并发关系，类似于**opencl** 中的命令队列( 乱序 )

### Kernel Parallel

我们知道在一个stream中，kernel的执行顺序一定是串行的，由于算力3.5以上的设备基本都支持了kernel并行，所以为了最大程度利用好GPU的资源，我们可以利用多个流来实现kernel并行的效果，进而不断提升性能，榨干GPU的资源。

### 原理概述

首先我们要把一些任务塞stream中，stream的任务然后再次被转到硬件的任务队列里，GPU在运行的时候最后是通过队列调度器来执行。

### 深度优先

说个大白话就是，一个流的任务添加完后再添加下一个流的任务。下面举一个简单的例子，比如我们在CPU端设置了3个stream, 第一个流A->B->C, 第二个流是O->P->Q, 第三流是X->Y->Z; 假如只有一个GPU的硬件工作队列，那么伪代码如下：

```
### CPU端
stream_1->add(C);
stream_1->add(B);
stream_1->add(A);
stream_2->add(Q);
stream_2->add(P);
stream_2->add(O);
stream_3->add(Z);
stream_3->add(Y);
stream_3->add(X);
### GPU端
quene->add(C,B,A,Q,P,O,Z,Y,X)
```

stream中图示如下：

![在这里插入图片描述](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuLzFjNDc2MzQ4NTMyYjQ0ZTA5MmZlZTBiMGIwYTM3MThlLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcg==)
硬件队列图示如下：

![深度优先](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuLzZiMzlmZTYxYTA3YjQ0YjI4M2UyN2VlZWEyOTlmMTdiLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcg==)

### 广度优先

说个大白话就是，每个流轮流添加任务。伪代码如下：

```
stream_1->add(C);
stream_2->add(Q);
stream_3->add(Z);
stream_1->add(B);
stream_2->add(P);
stream_3->add(Y);
stream_1->add(A);
stream_2->add(O);
stream_3->add(X);
#quene->add(C,Q,Z,B,P,Y,A,O,X)
```

![广度优先](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuLzczNTc4M2JmYmIxZTQ1YThiMDlkMTUzZTU3NTY1Y2U4LnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcg==)

### 深度优先性能

当利用深度优先的时候，GPU的队列调度器会首先去取任务C部署在GPU上，当发现还有很多计算资源的时候，然后去取B任务，但是发现B依赖的是C, 而C还没计算完成，那么就会等C返回后才能开始部署B，这样就导致真实的GPU运行如下, 重叠部分是由于当在部署A的时候，发现还有计算资源，然后就去取Q, 发现Q没有依赖，所以就可以同时运行，但是即使还有计算资源剩余，P也不能执行，因为他所依赖的Q还没返回。

![在这里插入图片描述](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuLzg2MTQ2ZjhhMmFhYTQ2NGNhZjhlOTAwZjU2YmQ1MzUyLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcg==)

### 广度优先性能

当时使用广度优先时，根据上面的硬件队列配合上面GPU队列调度器原理可以发现如下的结果，这样就实现了真正的三个流都并行了

![在这里插入图片描述](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuL2QyMjU3YmI5NWJjMjQ3NmM4ODJhMTNmNDA3MGFjZDI2LnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcj0xMHgxMA==)

### 多硬件队列

最早起的fermi架构的GPU可能就是一个硬件队列，现在生活富裕了，一般的GPU都支持32个硬件队列，但是实际上会默认只开8个硬件队列，所以使用的时候需要自己根据需求去设置`CUDA_DEVICE_MAX_CONNECTIONS`。这里我们假设有三个硬件队列，那么上面的结果就会如下：

![在这里插入图片描述](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuLzRkZDFhYTZhZmRlNzRlYzQ5ZTgxODYyZDg4NGQ1MjAyLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcg==)

GPU队列调度器首先去队列1拿取C, 发现还有资源，然后去队列2拿取Q, 发现还有计算资源剩余，然后去队列3拿取Z,结果发现还有计算资源剩余，然后返回队1去拿取B, 但是B要依赖C, 只有C计算返回后才能拿，所以调度器就会等待，最后实现了三路并行。

![在这里插入图片描述](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuLzFiODRiYmNmNGI3NjRlMmE4ZjQwM2M1ZDFlMjM5MTk5LnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcg==)

## stream->quene

那么流中的任务是怎么映射到硬件队列呢？原理如下：根据实际的加入时间，比如首先是C-1(代表任务c, stream为1号)，映射的话首先看哪个硬件队列有1号stream, 如果有就放入该队列，没有的话就放在第一个序号为空的队列，比如1,2队列都有任务，3，4都为空，那么就把这个任务放在3号队列。
例子：加入只有两个GPU硬件队列，采用深度优先的加入方式话，那么结果如下：

![在这里插入图片描述](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuL2VhOWZkZmYxZGQ4YTRiYTNhYWNkYTViMTRmM2Q5OTdmLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcg==)

​																							例子二：如果采用广度优先的话，队列如下：

![在这里插入图片描述](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuLzYzZTE4MzU1MjZkNDQwOWJiMWUwYjhjZTU5OGYwNWQ3LnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTYjcGljX2NlbnRlcg==)

### 个人建议

搞不明白的话，最好把队列设置的大于stream, 但是开设的队列越多占用的资源也多，搞得明白的话就学着用广度优先的添加方式。

### 代码示例

```c
#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#define N 30000000
#define M 1
#define NSTREAM 4

__global__ void kernel_1(double* res)
{
    
      
    double sum = 0.0;
    for(int j =0; j<M; j++)
    {
    
        for(int i = 0; i < N; i++)
        {
        sum = sum + tan(0.1) * tan(0.1);
        sum = sum + sin(0.1) * tan(0.1);
        sum = sum + cos(0.1) * tan(0.1);
        }
    }
    *res = sum;
}
nt main(int argc, char **argv)
{
    
      
    int n_streams = NSTREAM;
    int isize = 1;
    int iblock = 1;
    int bigcase = 0;

    // get argument from command line
    if (argc > 1) n_streams = atoi(argv[1]);

    if (argc > 2) bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connectioin
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv (iname, "32", 1);
    char *ivalue =  getenv (iname);
    printf ("%s = %s\n", iname, ivalue);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name,
           n_streams);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
      
        if (deviceProp.concurrentKernels == 0)
        {
     
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                    "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {  
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
       }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    size_t nBytes = sizeof(double);
    double *d_A;
    CHECK(cudaMalloc((double**)&d_A, nBytes));

    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(
                                cudaStream_t));

    for (int i = 0 ; i < n_streams ; i++)
    {
    
      
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1)
    {
    
        iblock = 512;
        isize = 1 << 12;
    }

    // set up execution configuration
    dim3 block (iblock);
    dim3 grid  (isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // record start event
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
    
      
        kernel_1<<<grid, block, 0, streams[i]>>>(d_A);
        kernel_2<<<grid, block, 0, streams[i]>>>(d_A);
        kernel_3<<<grid, block, 0, streams[i]>>>(d_A);
        kernel_4<<<grid, block, 0, streams[i]>>>(d_A);
    }

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n",
           elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0 ; i < n_streams ; i++)
    {
    
      
        CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}
```

### 结果

![在这里插入图片描述](http://140.238.201.79/image/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZmVuZ19fc2h1YWkvYXJ0aWNsZS9kZXRhaWxzLzEyMjQ0MTAxMQ==_aHR0cHM6Ly9pbWctYmxvZy5jc2RuaW1nLmNuLzRkNDY5YWNkYWRiZDQ1NGZhYWYxZmY5OTk1NjI0OTJmLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3dhdGVybWFyayx0eXBlX2QzRjVMWHBsYm1obGFRLHNoYWRvd181MCx0ZXh0X1ExTkVUaUJBY3k1bVpXNW4sc2l6ZV8yMCxjb2xvcl9GRkZGRkYsdF83MCxnX3NlLHhfMTY=)

### 思考

- 在根据《CUDA C编程权威指南》提供的源码进行复现的时候，始终无法得到上述的结果，折腾两天后发现原来的源码中核函数没有输入和输出，这样在编译器优化的时候，直接不予计算了，所以怎么测都有问题，调整后就可以得到上述结果。
- 当每个kernel启动的时候，grid和block的值都设置的很大的时候，并行的情况也不好，因为单个kernel执行的时候已经把计算资源用的差不多了，所以有时候资源调度器直接放弃取下个任务了。



## 算力评估

* FLOPS的计算公式如下

```text
浮点运算能力 = 处理器核数 * 每周期浮点运算次数 * 处理器主频
```
