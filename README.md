# AI Infra / CUDA 学习项目

## 当前主线：RMSNorm

按这个顺序阅读：

1. `src/rmsnorm/rmsNormWarp.cu`：你写的 kernel。`rmsNorm` 是单行版本；`rmsNormN` 每个 block 处理一行，支持正整数行宽 n。当前线程配置要求完整 warp，且 warp 数为 2 的幂。
2. `src/rmsnorm/rmsnorm_n_bench.cu`：直接包含上述文件，执行 CPU 参考校验及 128/256/512 线程性能对照。
3. `reports/rmsnorm/20261017_multiline/README.md`：本轮方法、结果和原始数据入口。

当前已确认：72 组正确性检查和 81 组性能配置全部 PASS。接下来是 Nsight Compute 分析及 CUB 对照；尚未完成，不将性能原因假设记作证据。

## 目录

| 目录 | 内容 |
|---|---|
| `src/` | 按主题分目录的源码；CMake 配置已同步更新 |
| `build-gcc/` | 可执行文件与历史 CMake 生成文件，不当作源码编辑 |
| `data/` | 输入、参考输出、ONNX、TensorRT engine |
| `reports/rmsnorm/` | RMSNorm 正确性及性能实验 |
| `reports/reduction/` | shared/warp 求和对照 |
| `reports/vector_add/` | 向量加法与预热对照 |
| `reports/tensorrt/nsight/` | TensorRT 拷贝与执行时间轴 |
| `archive/source_backups/` | 修改前的源码备份 |
| `archive/organize-*.json` | 本次移动清单与 SHA-256，可按清单恢复路径 |

## 源码导航

| 文件 | 作用 |
|---|---|
| `src/rmsnorm/rmsNormWarp.cu` | 当前手写 RMSNorm 主线 |
| `src/rmsnorm/rmsnorm_n_bench.cu` | 当前多行、可变 n 验证与性能测试 |
| `src/rmsnorm/rmsNorm.cu` + `src/rmsnorm/rmsnorm_test.cu` | 早期固定 256 维 shared-memory 版本与测试 |
| `src/rmsnorm/rmsnorm_variants.cuh` + `src/rmsnorm/rmsnorm_compare.cu` | 独立固定 256 维 shared/warp 对照，不等于当前 kernel |
| `src/rmsnorm/rmsnorm_variable_test.cu` | 助手独立单行示例，保留作参考；不替代用户源码测试 |
| `src/vector_add/vector_add.cu` | 向量加法练习 |
| `src/reduction/reduce_sum.cu`、`src/reduction/warp.cu` | 规约练习 |
| `src/reduction/reduction_bench.cu` | shared/warp 多阶段规约基准 |
| `src/tensorrt/build_engine.cpp` | TensorRT engine 构建源码 |
| `src/tensorrt/infer.cpp`、`src/tensorrt/infer_pageable.cpp`、`src/tensorrt/infer_async.cpp` | 推理与内存/流对照 |
| `src/basics/main.cpp` | 早期练习，保留 |
| 根目录 Python 脚本 | PyTorch 基线、ONNX 导出、输出比较 |
| `warmup_compare.sh` | 历史 vector_bench 预热实验脚本 |

## 运行和报告命名

从项目根目录运行；GPU 实验必须在已分配 GPU 的计算节点执行。
新的实验使用新的目录，避免覆盖已有 CSV。以下是命令模板，整理过程没有运行实验：

```bash
mkdir -p reports/rmsnorm/<新实验名>
./build-gcc/rmsnorm_n_bench reports/rmsnorm/<新实验名>/results
```

`results.csv` 是各轮统计，`results_samples.csv` 是原始样本，`.log` 是运行输出。
历史报告已按实验移动，旧的 `reports/文件名` 路径需要换成对应子目录。

## 已知工程缺口

- 已补回根目录 `CMakeLists.txt`，使用新的源码路径；只配置生成，未重新编译或运行。原 `check_trt.cpp` 缺失，历史 `check_trt` 二进制保留但不作为新配置目标。
- `warmup_compare.sh` 依赖现存的 `build-gcc/vector_bench`，远端当前没有对应 `src/vector_bench.cu`；未擅自用其他副本补回。
- RMSNorm 源文件目前有未跟踪文件；本次只整理，没有 git add、commit 或 push。
