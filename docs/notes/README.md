> 这是网站「AI Infra 学习日志」的 Markdown 版，由网站内容自动转换；网站上的交互动画在这里略去。

<a id="overview"></a>

*Learning journal · 2026*

# 从模型到 GPU 算子

这里用来复习推理优化。下面先是路线：三条线各自做到了哪一步、下一步是什么；再往下是每条主线"瓶颈在哪 → 怎么查 → 改了什么 → 快了多少"的骨架。

<a id="route"></a>

## 路线

### 算子线：手写 CUDA

- **RMSNorm 〔已完成〕**：受带宽限制的算子：warp shuffle 规约 → 编译期线程数 → x 存寄存器 → 按 n 分档 → float4，再接入 PyTorch、torch.compile 和 TensorRT plugin。 见 [RMSNorm 专题](rmsnorm.md)。
- **GEMM 〔已完成〕**：naive 408 → 寄存器分块 1744 → double buffering 2895 GFLOPS（4096³，cuBLAS 的 85%）。 见 [模块 11](m11.md)。
- **转置 〔已完成〕**：shared memory tile + padding 消 bank conflict，每线程搬 4 行后到 copy 的 97～98%（8192² 只有 80%，原因未查）。 见 [模块 12](m12.md)。
- **Softmax 〔已完成〕**：naive / block / online / warp 四版，按 n 分档派发：n ≤ 1024 用 warp 版（copy 的 106～108%），很长的行用 online 版。 见 [模块 14](m14.md)。
- **Scan 〔已完成〕**：单 block Hillis-Steele、三级 multi、warp 版（int4 + shuffle）；CUB 是 copy 的 82～85%。decoupled look-back 版没做。 见 [模块 15](m15.md)。
- **卷积 〔已完成〕**：direct → 显式 im2col + 自己的 GEMM → implicit GEMM（加预计算下标表）；resnet3x3 1.159 → 0.289 ms（cuDNN 0.126），1×1 卷积比 cuDNN 快。 见 [模块 11](m11.md)。
- **Attention / FlashAttention 〔进行中〕**：下一个算子：分块 + online softmax，不把 N×N 的分数矩阵写回显存。目前只有讲解。 见 [模块 16](m16.md)。
- **FP16 / Tensor Core GEMM 〔计划〕**：把 GEMM 换到 Tensor Core（WMMA / mma），对照 cuBLAS 的 FP16 kernel。

### 部署线：Orin + TensorRT

- **RMSNorm 接入框架 〔已完成〕**：PyTorch 扩展 → torch.compile（fused add + RMSNorm）→ CUDA Graph → TensorRT plugin。 见 [模块 7](m7.md)、[模块 8](m8.md)、[模块 9](m9.md)。
- **ACT 上 Orin 〔已完成〕**：ONNX → TensorRT FP16，b8 8.92 → 2.44 ms，和 PyTorch 的误差在 1e-3 量级；之后改做 VLA，ACT 不再往下做。 见 [模块 10](m10.md)。
- **SmolVLA 三个 engine 〔已完成〕**：视觉 A、前缀 B、去噪一步 C；LayerNorm / RMSNorm / RoPE 留 FP32，每次观测 336 → 约 95 ms，误差低于模型原生 bf16 的噪声底线。 见 [SmolVLA](m10.md#m10-smolvla)。
- **SmolVLA 提速 〔进行中〕**：C 的 10 步共约 38 ms 最大：整个循环录成 CUDA Graph、cross-attn 的 K/V 投影挪进 B 只算一次；先做逐层 profile 再定。 见 [SmolVLA](m10.md#m10-smolvla)。
- **INT8 与减少去噪步数 〔计划〕**：INT8 PTQ（敏感层回退 FP16），以及去噪步数和精度的取舍。

### 系统线：LLM 推理与通信

- **LLM 推理流程与 KV cache 〔已完成〕**：prefill 是 GEMM、受算力限制；decode 是 GEMV、受带宽限制；只缓存 K、V。 见 [模块 13](m13.md)。
- **FAST：MoE all-to-all 〔已完成〕**：服务器内用 NVLink 先匀开再跨机发送，NVLink 和网卡各用一个 stream 重叠；2 台 × 4 卡 Zipf 0.99 时比最强 NCCL 快 1.96×。后来发现的 bug 已修，修复后的数字待验证。 见 [Lina 模块](lina.md)。
- **vLLM-Omni 与 serving 〔计划〕**：PagedAttention、调度和 cache 管理，把 SmolVLA 放进 serving 框架里看 batching 下的表现。

## 各主线怎么优化的

- **RMSNorm（受带宽限制的算子）**：目标是 x 只读一次、y 只写一次。warp shuffle 规约 → 线程数改成编译期常量（循环展开，同时发出更多读请求，追平 CUB）→ 每线程元素数固定、x 存寄存器 → 按 n 分档（n ≤ 256 时一个 warp 算一行）→ float4。放进整层后，收益主要来自少发 kernel；在 torch.compile 下要做 fused add + RMSNorm，才不会挡住 Inductor 的融合；CUDA Graph 下整层追平 Inductor。见 [RMSNorm 优化主线](rmsnorm.md#mainline)。
- **GEMM（受计算限制的算子）**：核心是数据搬一次、用多次。4096³ 下：naive 408 → shared memory 分块 614 → 寄存器分块 1744 → 交错分工消除 bank conflict 2091 → float4 + As 转置 1790（反而变慢，ncu 看到占用率 33%）→ padding + 展开 kk 2366（cuBLAS 的 70%，ncu 里最大的停顿是等全局内存）→ double buffering 2895 GFLOPS（cuBLAS 的 85%，long_scoreboard 1.81 → 0.02）。见 [模块 11 优化主线](m11.md#m11-s13)。
- **转置（受带宽限制、读写方向不一致）**：读 A 连续，写 B 就要跳着写，全局内存那一半的带宽被浪费。把"跳着访问"挪到 shared memory 里做：按行读进 32×32 的 tile，再按 B 的行连续写出；竖着取 tile 的一列有 32 路 bank conflict，`tile[32][33]` 消掉。4096² 下：naive 37.3 → shared 61.9 → padding 91.1 GB/s（copy 的 89%）。之后发现 copy 自己只有 Orin 带宽的一半：每个线程只搬 1 个数，同时在路上的读请求太少。block 改成 32×8、每个线程搬 4 行后 padding 版到 184 GB/s；再给 copy 的指针加 `__restrict__`（编译器才敢把几个读一起提前发出），copy 到 188 GB/s，转置是它的 98%。见 [模块 12](m12.md)。
- **Softmax（受带宽限制、每行要两次规约）**：每行先减最大值再取指数、除以总和。优化分两个方向：**分工**（一行交给 1 个线程 / 1 个 block / 1 个 warp）和**算法**（读几遍）。一个线程一行时 warp 内地址不连续，只有 copy 的 4～11%；一个 block 一行后 n = 1024、4096 接近 copy（后两遍读命中缓存）；n = 32000 一行 125 KB 放不进缓存，三遍读 → 49%，online softmax 把最大值和分母合成一遍 → 63%；n = 128 时 256 个线程、16 次同步只为 128 个数，只有 20%；改成一个 warp 一行、整行放进寄存器、只读一遍、用 xor shuffle 规约不要同步后到 190 GB/s（copy 的 108%，峰值的 93%）。最后按 n 分档：n ≤ 1024 用 warp 版、中等用 block 版、很长用 online 版。见 [模块 14](m14.md)。
- **MoE all-to-all（FAST 复现，跨机网卡是瓶颈）**：流量不均衡时，NCCL 的用时由最忙那张 GPU 的网卡决定。FAST 先用 NVLink 在服务器内把要发出去的数据匀给 4 张卡，再让第 k 张卡只发给对面第 k 张。Quest 2 台 × 4 张 H100 上：先逐层排查跨机带宽（关掉 GPUDirect 后每张网卡 13.7 → 20.2 GB/s）→ 第一版 FAST 比最强 NCCL 慢（0.63～0.82×），在均匀流量上多出的时间就是实现开销 → 三步分开计时，发现跨机一步网卡跑不满，打包成一整块 + 每对卡 8 条通道后 Zipf 0.99 从 28.3 降到 13.0 ms → 切块流水线放在同一次 NCCL 调用里只到 1.69× → NVLink 和网卡各用一个 communicator、一个 stream，用 event 衔接，Zipf 0.99 时 **1.96×**、离理论最快约 2%（之后 2 台 × 4 卡复测只有 1.72×，查明不是机器波动，而是后来改代码引入的 bug：机内直传被并进第一块的准备步骤，网卡要等它传完才开始发，本来能重叠的两件事变成了先后；同一对机器上旧版 1.92×，修复后的数字待验证）；均匀流量仍是 0.86×。换成 4 台 × 每台 2 卡后最多只有 1.25×，补上的 BvN 分解只比简单轮流配对快 2～7%：收益几乎全部来自机内的卡互相分担，取决于每台有几张卡。见 [Lina 模块第 10 节](lina.md#lina-fast2)、[第 11 节](lina.md#lina-fast3)。
- **TensorRT 部署**：ONNX → engine；FP16 让矩阵乘换成 Tensor Core kernel（Orin，b8 8.92 → 2.44 ms）。TensorRT 会自己把 RMSNorm 的 7 个节点融成 1 个 kernel；换成自定义 plugin 后结果对，但 Myelin 的子图被切开，性能会跟着变（b8 反而快了一倍）。SmolVLA 拆成视觉 A、前缀 B、去噪一步 C 三个 engine：直接全 FP16 会算错（LayerNorm / RMSNorm 的平方超出 FP16 范围、RoPE 角度大到 240 弧度精度不够），把这些节点留 FP32、矩阵乘仍走 FP16，每次观测 336 → 约 95 ms，端到端动作误差低于模型原生 bf16 和 FP32 的差距。见 [模块 9](m9.md)、[模块 10](m10.md)、[SmolVLA](m10.md#m10-smolvla)。

设备：Jetson AGX Orin 64GB（sm_87，TensorRT 8.5.2）是主力平台；Quest A100-PCIE-40GB（sm_80）用于交叉验证。

## 复习笔记目录


**部署与工程**

- [1 · 全流程总览](m1.md)
- [2 · 环境与工程构建](m2.md)
- [3 · C++ 必备](m3.md)
- [4 · 模型导出与正确性](m4.md)

**GPU 与性能**

- [5 · TensorRT 与 CUDA](m5.md)
- [6 · 性能测量](m6.md)
- [7 · PyTorch 与端到端](m7.md)
- [8 · compile、融合与 CUDA Graph](m8.md)
- [9 · ONNX → TensorRT plugin](m9.md)

**端侧部署**

- [10 · 策略模型上机](m10.md)

**核心算子**

- [11 · 卷积与 GEMM](m11.md)
- [12 · 转置](m12.md)
- [14 · Softmax 与 online softmax](m14.md)
- [15 · Scan（前缀和）](m15.md)

**LLM 推理**

- [13 · 推理流程与 KV cache](m13.md)
- [16 · FlashAttention](m16.md)

**论文**

- [AWQ 量化](awq.md)
- [Lina 与分布式 MoE](lina.md)
- [端侧 VLA 论文](vla.md)

- [RMSNorm 专题](rmsnorm.md)

