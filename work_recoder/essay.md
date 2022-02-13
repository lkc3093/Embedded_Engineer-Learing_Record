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

## RTOS
* 线程切换类似于中断中的上下文保护
* 互斥量是对共享内存(全局变量之类)的保护
* 管道可以用于通信
* sizeof 指针是4个字节
*  1. 结构体变量的首地址能够被其最宽基本类型成员的大小所整除；
   1. 结构体每个成员相对结构体首地址的偏移量(offset)都是成员大小的整数倍，如有需编译器会在成员之间加上填充字节(internal adding)；
   2. 结构体的总大小为结构体最宽基本类型成员大小的整数倍，如有需要编译器会在最末一个成员之后加上填充字节{trailing padding}。

## OPENCL

* 执行模式有任务并行和数据并行
  * 数据并行指的是不同的工作组执行同一个指令不同数据，通过`get_global_id` 来嵌入使用
  * 任务并行，多用于不同的 **kernel** 函数，以及队列命令乱序执行，当前命令没有对之前命令的依赖关系
    * 有序执行 (in-order execution)：命令按其在命令队列中出现的顺序发出，并按顺序完成。队列中前一个命令完成之后，下一个命令才会开始。这会将队列中命令的执行顺序串行化
    * 乱序执行 (out-of-order execution)：命令按顺序发出，但是下一个命令执行之前不会等待前一个命令完成。程序员要通过显式的同步机制来强制顺序约束
  
* **work-item**（工作项）：**work-item** 与 **cuda threads** 是一样的，是最小的执行单元。每次一个Kernel开始执行，很多（程序员定义数量）的**work-item**就开始运行，每个都执行同样的代码
  * 每个**work-item**有一个 id ，这个 id 在 kernel 中是可以访问的，每个运行在 **work-item **上的 kernel 通过这个 id 来找出**work-item**需要处理的数据
  * **work-item** 是循环中的最小单位
  * 
  
* 存储器模式是软件定义的，而非单纯硬件实现；可以使用`constant` 来将数据放入常量存储区，访问更快；

* kernel函数中间可以嵌套kernel函数，设备端可以创建命令队列，极大地方便了代码编写

  

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

  