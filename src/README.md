# 源码导航

- `rmsnorm/`：当前主线。先读 `rmsNormWarp.cu`，再读 `rmsnorm_n_bench.cu`。
- `reduction/`：共享内存、warp 与多阶段规约练习。
- `vector_add/`：向量加法练习。
- `tensorrt/`：engine 构建与推理。
- `basics/`：早期 C++ / CUDA Runtime 练习。

RMSNorm 的 kernel 和包含它的测试留在同目录，相对 include 不变。
可执行文件仍在根目录的 `build-gcc/`。数据与报告路径不变，运行时仍从项目根目录启动。
旧的手动 nvcc 命令需要换成新源码路径，例如 `src/rmsnorm/rmsnorm_n_bench.cu`。
