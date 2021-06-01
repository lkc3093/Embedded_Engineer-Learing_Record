##  API熟悉
* **lamba匿名函数**
````python

a = lambda x, y: x + y # x,y相当于传入的参数,函数会返回x+y的值
print(a(1,2)) # 打印出3

````
* **构建TVM算子**
```python
A = te.placeholder((1024,), name='A') 
# 创建变量
k = te.reduce_axis((0, n), 'k')
# 创建维度
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B') 
# 定义计算细节
s = te.create_schedule(B.op)
# 创建调度器
print(tvm.lower(s, [A, B], simple_mode=True))
# 低级代码显示
```
* **生成中间IR和后端编译代码**

  * 通过IRModule的astext函数可以查看中间IR

  * 通过运行时module的get_source查看生成的代码
  
````python
import tvm
from tvm import te

M = 1024
K = 1024
N = 1024

# 创建变量和维度
k = te.reduce_axis((0, K), 'k')
A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
C = te.compute(
           (M, N),
           lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
           name='C')

# 创建调度器
s = te.create_schedule(C.op)
ir_m = tvm.lower(s, [A, B, C], simple_mode=True,name='mmult')
rt_m = tvm.build(ir_m, [A, B, C], target='c', name='mmult')

# 打印中间代码
print("tir:\n", ir_m.astext(show_meta_data=False))
# 打印后端c代码
print("source code:\n",rt_m.get_source())
````
* **后端调度优化**
````python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tvm import relay
from tvm.relay import testing
from tvm.contrib import util
import tvm

# Resnet18 mod是计算图,params是参数模块,lib是算子
resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)

# 优化级别选择
with relay.build_config(opt_level=0):
    _, resnet18_lib, _ = relay.build_module.build(resnet18_mod, "llvm", params=resnet18_params)

print(resnet18_mod.astext(show_meta_data=False))

print(resnet18_lib.get_source())
````
