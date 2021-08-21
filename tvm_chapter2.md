## 计算图优化

* **优化策略**
    * <font color=DeepSkyBlue>operator fusion</font>: 把多个独立的operator融合成一个；
    * <font color=DeepSkyBlue>constant-folding</font>: 把一些可以静态计算出来的常量提前计算出来；
    * <font color=DeepSkyBlue>static memory planning pass</font>: 预先把需要的存储空间申请下来，避免动态分配；
    * <font color=DeepSkyBlue>data layout transformations</font>: 有些特殊的计算设备可能会对不同的 data layout (i.e. NCHW, NHWC, HWCN)有不同的性能，TVM可以根据实际需求在graph层面就做好这些转换。

* **基本调度学习**
  * split:分割，有助于提高并行性
  * reorder:重新分布循环次序，提高cache的命中率 

* **autotvm使用**
````python
def case_autotvm_relay_centerFace():
    # InitCenterFacePy封装了pytorch的	       加载代码
    model = InitCenterFacePy()
    # tvm搜索完成后将结果保存在.log中
    log_file = "centerFace.log"
    dtype = "float32"
    # 初始化优化器，及优化选项
    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        # "n_trial": 1,
        "n_trial": 2000,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }
    print("Extract tasks centerFace...")
    mod, params, = relay_import_from_torch(model.module.cpu(), direct_to_mod_param=True)
    input_shape = [1, 3, 544, 960]
    target = tvm.target.cuda()
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_option)

    # compile kernels with history best records
    # 模型搜索完成后，进行耗时统计。
    profile_autvm_centerFace(mod, target, params, input_shape, dtype, log_file)
````
