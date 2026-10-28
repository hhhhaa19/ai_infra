# AI Infra / CUDA 学习项目

主线：手写一个 RMSNorm CUDA 算子，一步步把它接进真实的推理路径。

```
手写 kernel ──▶ PyTorch 算子（eager / torch.compile） ──▶ 整层 decoder layer ──▶ ONNX ──▶ TensorRT plugin
src/rmsnorm/    src/rmsnorm/torch/                       src/llama/              src/llama/   src/rmsnorm/tensorrt/
```

两台机器：
- **Orin**（Jetson AGX Orin，sm_87，CUDA 11.4，TensorRT 8.5，torch 2.1）：日常开发和测试。
- **Quest**（A100，sm_80，torch 2.x + Triton）：torch.compile 相关对比，通过 Slurm 跑。

## 目录

```
src/
  rmsnorm/                 ★ 主线：RMSNorm 算子
    kernels.cu             所有 kernel：循环版、own_reg、warp 版、float4 版、fused add+RMSNorm 等
    launch.cuh             选档 + 启动 kernel（只依赖 CUDA，PyTorch 扩展和 TensorRT plugin 共用）
    cuda_bench/            不经过 PyTorch 的独立 CUDA 测试
      n_bench.cu             多行、可变 n 的正确性和性能
      cub_kernel.cuh         CUB BlockReduce 版 kernel（对照用）
      cub_example.cu         CUB 版单形状示例
      cub_bench.cu           自己的 kernel vs CUB
      profile.cu             给 Nsight Compute 采集用
    torch/                 PyTorch 扩展：torch.ops.ai_infra.rms_norm / fused_add_rms_norm
      ext.cu                 张量检查 + 注册 CUDA / Meta 实现
      build_ext.py           编译并加载扩展（load_rmsnorm）
      test.py                rms_norm 正确性（各实现、各形状、该报错的输入）
      check_fused.py         fused_add_rms_norm 正确性
      check_vec4.py          float4 版正确性
      bench.py               各实现计时
      compile_vs_ext.py      和 torch.compile 生成的 kernel 对比
      sweep_block.py         线程数 × float4 扫描（Quest）
      sweep_warp.py          warp 版扫描
    tensorrt/
      plugin.cu              TensorRT plugin：ONNX 里的 RmsNorm 节点由它执行
  llama/                   整层 LLaMA 风格 decoder layer（无 RoPE / KV cache）
    decoder_layer.py         模型；impl = eager / ours / fused 切换 RMSNorm 实现
    check_fused_layer.py     整层正确性 + torch.compile 不断图
    bench_layer.py           整层计时（含手动 CUDA Graph）
    compile_layer.py         eager / torch.compile / CUDA Graph 等组合对比
    profile_layer.py         torch profiler 采集
    analyze_nsys.py          解析 nsys 结果
    export_onnx.py           导出 ONNX（ours 会导出成自定义节点 ai_infra::RmsNorm）
    trt_check.py             TensorRT engine 正确性 + 计时
  reduction/               规约练习：reduce_sum.cu、warp_reduce.cu、reduction_bench.cu
  tensorrt_intro/          最早的 ResNet → ONNX → TensorRT 练习
    export_resnet.py         导出 ResNet 的 ONNX 和参考输出
    check_onnx.py            用 onnxruntime 检查 ONNX
    build_engine.cpp         C++ 建 engine
    infer*.cpp               C++ 推理（同步 / 可分页内存 / 异步流对比）
    compare_outputs.py       对比 TensorRT 和参考输出
  basics/                  早期练习：main.cpp、vector_add.cu
scripts/
  setup_orin.sh            Orin 上建 Python 环境（配合 requirements-orin.txt）
  env_quest.sh             Quest 上加载模块和 conda 环境
  build_trt_engines.sh     用 trtexec 建 engine 并计时
```

不进 git 的目录：`build-orin/`（CMake 产物）、`build-onnx/`（ONNX 和 engine）、`reports/`（实验结果）、`data/`。

## 常用命令（Orin，在仓库根目录）

环境：
```bash
source ~/venvs/ai_infra/bin/activate && export PATH=/usr/local/cuda/bin:$PATH
```

计时前先锁频（重启后会失效），做完恢复：
```bash
sudo jetson_clocks            # 恢复：sudo jetson_clocks --restore
```

PyTorch 扩展测试（第一次会编译扩展，约 40 秒）：
```bash
python src/rmsnorm/torch/test.py
python src/rmsnorm/torch/check_fused.py
python src/rmsnorm/torch/check_vec4.py
python src/llama/check_fused_layer.py
```

C++ / CUDA 程序和 TensorRT plugin：
```bash
cmake -S . -B build-orin -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build-orin                       # 或 --target rmsnorm_trt_plugin
```

整层 → ONNX → TensorRT：
```bash
python src/llama/export_onnx.py ours           # 输出到 build-onnx/layer_ours/
scripts/build_trt_engines.sh ours              # 建 engine + trtexec 计时
TRT_PLUGINS=build-orin/librmsnorm_trt_plugin.so python src/llama/trt_check.py build-onnx/layer_ours/ours
```
把 `ours` 换成 `eager` 就是 TensorRT 自己处理 RMSNorm 的对照组（不需要 `--plugins` 和 `TRT_PLUGINS`）。

## 约定

- 所有命令从仓库根目录运行。
- 实验结果放 `reports/<主题>/<日期>_<机器>_<内容>/`，新实验用新目录，不覆盖旧结果。
- 计时前锁频、先预热；用 CUDA event 包住 N 次调用再除以 N，多轮取中位数。
