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
* 执行模式有任务并行和数据并行
  * 数据并行指的是不同的工作组执行同一个指令不同数据，通过`get_global_id` 来嵌入使用
  * 任务并行，多用于不同的 **kernel** 函数，以及队列命令乱序执行，当前命令没有对之前命令的依赖关系
    * 有序执行 (**in-order execution**)：命令按其在命令队列中出现的顺序发出，并按顺序完成。队列中前一个命令完成之后，下一个命令才会开始。这会将队列中命令的执行顺序串行化
    * 乱序执行 (**out-of-order execution**)：命令按顺序发出，但是下一个命令执行之前不会等待前一个命令完成。程序员要通过显式的同步机制来强制顺序约束
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

  * 一个work-item不能被拆分到多个PE上处理；同样，一个work-group也不能拆分到多个CU上同时处理

  * 如果我想让group数量小点，那work-item的数目就会很多，还能不能处理了呢？以当前这个示例是能的，但是对于多的work-item,这涉及到如何确定work-item数目的问题

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
    在计算能力2.x之前的设备，全局内存和局部内存的访问都会在 **L1\L2 cache** 上缓存；在计算能力3.x以上的设备，全局内存的访问只在**L2 cache**上缓存。

  

* 通过 **stream** 可以实现多个内核计算并行，或者计算与IO并行；类似于**opencl** 中的命令队列( 乱序 )

* 访问连续的全局内存时，**GPU** 会合并访问，一次取出多个连续数据

  * 合并访问是指所有线程访问连续的对齐的内存块，对于L1 cache，内存块大小支持32字节、64字节以及128字节，分别表示线程束中每个线程以一个字节（1*32=32）、16位（2*32=64）、32位（4*32=128）为单位读取数据。前提是，访问必须连续，并且访问的地址是以32字节对齐。（类似于SSE\AVX的向量指令，cuda中的合并访存也是向量指令）

    例子，假设每个 **thread **读取一个float变量，那么一个**warp**（32个**thread**）将会执行32*4=128字节的合并访存指令，通过一次访存操作完成所有thread的读取请求

  * **Shared Memory**

    用 __shared __修饰符修饰的变量存放在shared memory。因为shared memory是on-chip的，他相比localMemory和global memory来说，拥有高的多bandwidth和低很多的latency。他的使用和CPU的L1cache非常类似，但是他是programmable的

    按惯例，像这类性能这么好的memory都是有限制的，shared memory是以block为单位分配的；我们必须非常小心的使用shared memory，否则会无意识的限制了active warp的数目

  * **Constant Memory**

    常量内存同样是offchip内存，只读，拥有SM私有的 **constant cache**，因此在cache hit的情况下速度快。常量内存是全局的，对所有Kernel函数可见。因此声明要在Kernel函数外

  

* 访问局部内存时，可能会出现 **bank** **conflict**，局部内存是一个个 **bank** 组成

* **local memory** 是线程私有的，寄存器不够，就会将数据存入 **local memory**

* 

  

## 算力评估

* FLOPS的计算公式如下

```text
浮点运算能力 = 处理器核数 * 每周期浮点运算次数 * 处理器主频
```
