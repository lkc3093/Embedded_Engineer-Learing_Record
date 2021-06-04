## 计算图优化

* **优化策略**
    * <font color=DeepSkyBlue>operator fusion</font>: 把多个独立的operator融合成一个；
    * <font color=DeepSkyBlue>constant-folding</font>: 把一些可以静态计算出来的常量提前计算出来；
    * <font color=DeepSkyBlue>static memory planning pass</font>: 预先把需要的存储空间申请下来，避免动态分配；
    * <font color=DeepSkyBlue>data layout transformations</font>: 有些特殊的计算设备可能会对不同的 data layout (i.e. NCHW, NHWC, HWCN)有不同的性能，TVM可以根据实际需求在graph层面就做好这些转换。
* **autotvm使用**
