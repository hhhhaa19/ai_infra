<a id="rmsnorm"></a>

*重点面试专题*

# RMSNorm

从一个手写 CUDA kernel 出发，做到 PyTorch 扩展、torch.compile 对照、decoder layer 端到端测量，再写成 TensorRT plugin 在 Orin 上跑通。目标是不看笔记也能连贯讲完，并能回答追问。

y[r][i] = x[r][i] · rsqrt( (1/n)·Σ

j

x[r][j]² + ε ) · w[i]

每行求平方和、规约成一个数，再逐元素缩放。计算很少、读写很多，受**显存带宽**限制。优化目标：**x 只读一次、y 只写一次**，尽量跑满带宽。

优化主线

两条部署路线

讲述稿

完整过程

分档设计

我的笔记

实测数据

踩坑与方法

面试追问

验收标准

<a id="mainline"></a>

## 优化主线：从一个 kernel 到整层追平 Inductor

每一步都按同一个格式：**遇到什么问题 → 怎么查（证据）→ 改了什么 → 结果 → 学到什么**。四个阶段，问题的层次一步步往上走：kernel 内部 → 放进 PyTorch → 放进生产路线（torch.compile / CUDA Graph）→ 进 TensorRT。

```text
准备    规约练习（A100）
  ⓪ warp 规约  shared 规约 → warp shuffle + shared → 大数组快 19～24%，后面 RMSNorm 直接沿用

阶段 A  单个 kernel（Orin + A100）
  ① own        能算对                         → 比 CUB 慢 15～36%
  ② own_t      线程数改成编译期常量            → 追平 CUB
  ③ own_reg    每线程元素数固定、x 存寄存器    → 18 个配置里 15 个最快
  ④ 按 n 分档  warp 版 / own_reg / 循环版      → 各种 n 都不吃亏

阶段 B  接进 PyTorch（Orin）
  ⑤ 扩展       pybind + 当前 stream            → 单算子比 eager 快约 4 倍，带宽 94%
  ⑥ 进整层     LLaMA 风格 decoder layer        → prefill −1.5%，decode b1 −10%（收益来自少发 kernel）

阶段 C  进生产路线（Quest A100）
  ⑦ 正式注册   TORCH_LIBRARY + Meta            → compile 认得它；修好一个假的 graph break
  ⑧ 正面对比   9 种写法整层对照                 → compile 下反而变慢：挡住了 Inductor 的融合
  ⑨ 融合       fused add + RMSNorm             → kernel 数追平，GPU 还多 26 μs
  ⑩ 按行数分档 decode 只有 1 行，每行要更多线程 → GPU 差 5 μs
  ⑪ float4     一条指令读 4 个 float           → GPU 差 1.6 μs；CUDA Graph 下整层 +0.6%，b8 打平

阶段 D  进 TensorRT（Orin，FP32）
  ⑫ 基线       eager 整层 → ONNX → engine       → Myelin 自己把 7 个节点融成 1 个 norm kernel（约 7 μs）
  ⑬ plugin     launch.cuh 去 PyTorch + 自定义节点 + plugin → 结果对（1e-6）；norm2 5.9 μs；b8 意外快一倍（Myelin 被切开）
```

### 阶段 A：把一个 kernel 写快（瓶颈在 kernel 内部）

| 步 | 问题 | 怎么查 | 改了什么 | 结果 | 学到什么 |
|---|---|---|---|---|---|
| 0 | RMSNorm 为什么要手写 | nsys 看 eager 的 kernel | — | eager 6 个 kernel，单个都接近带宽上限，但中间结果来回写，总搬运 448 MiB（只需 128 MiB） | 受带宽限制的算子，目标是 x 读一次、y 写一次 |
| ① | 先写出能用的版本 | float64 参考、输出先写 NaN、逐元素比 | own：一个 block 一行，跨步循环，warp shuffle + shared 规约 | 15 组 PASS（误差 2.38e-7），但比 CUB 慢 15～36% | 先正确，再谈快 |
| ② | CUB 为什么快 | ncu（CUB 的 eligible 反而更低，推翻"更多 warp 就绪"）→ 看 SASS：own 3 条 LDG，CUB 39～105 条 | own_t：线程数改成模板参数 | 和 CUB 差距 ≤4% | 编译期常量 → 循环展开 → 同时发出更多读请求；高占用率 / 高发射率不是目标 |
| ③ | n 除不尽时 own_t 变慢 | ncu：barrier stall 约 7 倍，活跃线程 31.6 → 27.7；n 不是 32 的倍数时 sector/请求 4 → 4.44 | own_reg：每线程固定 PER_THREAD 个、完全展开、`col < n` 屏蔽、x 存寄存器只读一次；padding 解决对齐 | 18 个配置中 15 个最快，比 own 快 15～33%，LOCAL:0 | 零头分支 + barrier 会拖住整个 block；对齐决定一次读多宽 |
| ④ | 不同 n 一个 kernel 管不过来 | 强制两种实现扫描 n=32～384，找交叉点 | n ≤ 256 一 warp 一行（xor shuffle，无 shared、无 barrier）；≤ 33792 own_reg；更大循环版 | 阈值从猜的 1024 改成实测的 256 | "能放下"不等于"更快"，边界靠扫描 |

### 规约是怎么一步步变快的

RMSNorm 每一行都要先把所有元素的平方加起来，这个规约必须让线程之间传数据。最慢的做法是所有数都经过共享内存，所以优化方向很明确：**能在寄存器里合并就不进 shared，能少一次屏障就少一次**。这里要注意，规约只是 RMSNorm 的一部分，own → own_t → own_reg 的主要收益来自读请求和展开，不是规约。

| 写法 | 线程怎么合并 | 256 线程时的屏障数 | 代价和效果 |
|---|---|---|---|
| shared 规约 | 全部数写进 shared memory，一层层两两相加 | 9 | 每层都要读写 shared，而且每层一个屏障 |
| warp shuffle + shared | warp 内用 5 轮 `__shfl_down_sync` 在寄存器里合并，只有 warp leader 把部分和写进 shared，再合并 8 个部分和 | 4 | A100 四轮中位数：100 万元素 16.527 → 13.394 μs（−18.96%），1600 万元素 131.092 → 99.471 μs（−24.12%） |
| warp-per-row | 一行只用 1 个 warp，5 轮 `__shfl_xor_sync` 后 32 个线程都有总和 | 0 | 没有 shared，也没有屏障；一行元素要放得进这个 warp，所以用于 n ≤ 256 |

为什么 warp 版更快？shared 规约要做 8 层合并，warp 内 5 轮 shuffle 只操作寄存器，剩下 3 层才进 shared，所以 shared 读写和屏障都变少了。这是机制假设，还没有用 ncu 证实。屏障数 9 和 4 由合并层数加上结果可读前的 1 次得出。

own_reg 的规约还是第二种写法：warp 内 5 轮 shuffle，`lane == 0` 写 `p[warpid]`，再按 `warpCount/2` 一层层合并，最后线程 0 算 scale，其他线程等屏障后读取。n 不大时换成第三种，就不需要这些屏障了。

这张表里的耗时数据来自规约练习，不是 RMSNorm 整体的提速，也没有单独测过 RMSNorm 里规约这一段的时间。

### 阶段 B：接进 PyTorch，放进模型（瓶颈变成 Amdahl 和 CPU）

| 步 | 问题 | 怎么查 | 改了什么 | 结果 | 学到什么 |
|---|---|---|---|---|---|
| ⑤ | Python 里怎么用 | 23 形状 + 160 强制路径 + 5 输入检查 | CUDA 扩展：TORCH_CHECK、empty_like、当前 stream、pybind；源码哈希触发重编 | 单算子 4096×8192 比 eager 快 4.1 倍，约 192 GB/s（峰值 94%）；A100 单算子和 compile 打平 | 必须用 PyTorch 当前 stream；只做了前向 |
| ⑥ | 在完整模型里有没有收益 | nsys + NVTX 按模块归 kernel；再不开工具用 CUDA event 计时 | 写 LLaMA 风格 decoder layer，RMSNorm 可切换 | RMSNorm 只占约 2%：prefill −1.5%（Amdahl 预测 −1.4%）；decode b1 −10%、b8 −2%，来自少发 10 个 kernel（每个约 15 μs CPU） | 算子快 4 倍 ≠ 模型快 4 倍；decode 时 GPU 在等 CPU；nsys 会把发射开销放大（曾测成 −23%） |

### 阶段 C：进生产路线，和编译器正面比（瓶颈在融合边界和真实形状）

| 步 | 问题 | 怎么查 | 改了什么 | 结果 | 学到什么 |
|---|---|---|---|---|---|
| ⑦ | pybind 函数 compile 看不懂 | — | TORCH_LIBRARY：schema + CUDA + Meta，统一 torch.ops | compile 能把它当正常节点；explain 报切断 → explain 跑两次确认是 forward 里延迟加载，改到 `__init__` | 注册才让 PyTorch "认识"算子；先注册、再 compile、最后调用；推断要重跑确认（切断其实没影响计时） |
| ⑧ | 生产路线下还有没有收益 | A100 整层 eager / compile / compile+CUDA Graph × 原生 / ours，kernel 数 + 计时 | — | decode b1：eager+ours −29%，但 compile+ours（−23%）比 compile（−30%）还慢 | 我们的算子是黑盒，挡住了 Inductor 的"残差相加 + norm"融合 |
| ⑨ | 把被挡住的融合补上 | 读 inductor_cache 生成的代码 | fused add + RMSNorm：h = x + r 只写一次，归一化用寄存器里的值 | kernel 数追平（14 对 14），eager+fused 已≈compile；但 GPU 时间多 26 μs | 要按数据流设计融合边界；kernel 数一样以后，比的是 kernel 本身 |
| ⑩ | 为什么 kernel 本身慢 | Inductor 用 num_warps=16（每行 512 线程）；单 kernel 扫描（CUDA Graph 计时）：1 行时 128×16 最慢 | 分档加行数：行 < 2×SM 每线程 4 个，否则 8 个 | GPU 差距 26 → 5 μs；CUDA Graph 下 +2.4% | 单独测算子用的是几千行；decode 只有 1 行、1 个 block，看延迟，要每行更多线程 |
| ⑪ | 剩下的几 μs | Triton 的 `tt.divisibility 16` → 它在用 128 位读取 | float4：一条 LDG.E.128 读 4 个 float，对齐时自动用，否则退回 | GPU 278.0 vs 276.4 μs；CUDA Graph 下 decode b1 +0.6%、b8 打平 | 指令少 = 等待轮数少；计时前要先让 GPU 升频（一次扫描因此测偏） |

### 阶段 D：进 TensorRT（瓶颈变成"编译器挑的矩阵乘 kernel"）

| 步 | 问题 | 怎么查 | 改了什么 | 结果 | 学到什么 |
|---|---|---|---|---|---|
| ⑫ | TensorRT 自己能把 RMSNorm 融合到什么程度 | trtexec + `--exportLayerInfo`（只看到 1 个 ForeignNode）→ nsys 看 engine 实际启动的 kernel | eager 整层导出 ONNX（3 种形状），FP32 + noTF32 建 engine；trt_check.py 查正确性 | 7 个节点融成 1 个 `MulMeaAddSqrDivMulMul`，b1 约 7 μs；b8 8.9 ms，慢在 3 个 sliced1x4 sgemm（各 2.4 ms） | TensorRT builder ≈ Inductor，但偏"挑"不偏"生成"；建 engine 也要锁频 |
| ⑬ | 让 TensorRT 用我们的 kernel | trtexec 日志看 plugin 查找过程；trt_check.py 对 ref；`--dumpProfile` 逐层 | 选档 / 启动搬进 launch.cuh；symbolic 导出 RmsNorm 节点 + RmsNormPlugin / Creator（1665953） | 误差约 1e-6；b1 1.33（CG 1.21）vs 1.32（1.25）ms；b8 3.98 vs 8.92 ms，但原因是 MLP 矩阵乘被切出 Myelin，不是 norm 快 | 插入黑盒会改变编译器的整体决策；同一个算子在两个编译器里各接一次，接法对称（Meta ↔ getOutputDimensions，CUDA ↔ enqueue） |

### 第 ⑬ 步实际是怎么做的

这一步要回答三个问题：TensorRT 有没有真的调用我们的 kernel？结果对不对？时间花在哪？下面按实际操作顺序走一遍，全程在 Orin 上，FP32、禁用 TF32、锁频。

| 顺序 | 运行什么 | 计算或产出什么 | 回答什么问题 |
|---|---|---|---|
| 1 | `python src/llama/export_onnx.py ours` | PyTorch 用固定输入算出 ref，再导出 ONNX。每个 RMSNorm 从 7 个节点变成 1 个 `RmsNorm`，eps 写进节点属性。产出 `layer_ours_*.onnx`、`io_*.pt`、`graph_ours.txt` | ONNX 里有没有留下自定义节点 |
| 2 | `cmake --build build-orin --target rmsnorm_trt_plugin` | `nvcc` 编译 plugin、launch.cuh 和 kernels.cu，产出 `librmsnorm_trt_plugin.so` | plugin 能不能脱离 PyTorch 编译 |
| 3 | `scripts/build_trt_engines.sh ours` 里的第一条 trtexec：`--saveEngine`、`--plugins`、`--noTF32`、`--exportLayerInfo` | TensorRT 解析 ONNX、调用 plugin 查形状和类型、做融合、实测挑 kernel，产出 `ours_*_fp32.engine` 和 `*_layers.json`；顺带跑随机输入，得到普通计时 | 日志里有 `Successfully created plugin: RmsNorm`，说明确实找到了 plugin；layers.json 说明 Myelin 被切成 17 段 |
| 4 | 同一脚本里的第二条 trtexec：`--loadEngine`、`--useCudaGraph` | 加载 engine，用 CUDA Graph 重放并计时 | 去掉 CPU 发射开销后要多久：b1 1.21 ms |
| 5 | 同一脚本里的第三条 trtexec：`--dumpProfile`、`--separateProfileRun`、`--exportProfile` | 逐层统计时间，写入 `*_profile.json` | 时间花在哪：norm2 5.9 μs、norm1 16.5 μs；b8 时 MLP 三个矩阵乘 0.87 ms，attention 段 3.05 ms |
| 6 | `TRT_PLUGINS=build-orin/librmsnorm_trt_plugin.so python src/llama/trt_check.py build-onnx/layer_ours/ours` | 用导出时的同一个 x 运行 engine，计算 `max\|out − ref\|` 和相对误差；再用 CUDA event 计时 | 结果对不对：误差约 1e-6，和 eager engine 同一量级 |
| 7 | 同样的流程跑 `eager`，放在一起比较 | 得到 eager engine 的时间和逐层信息 | 换成 plugin 后到底变了什么 |

有两点要分清。第一，trtexec 用随机输入，所以它的时间能看，但不能证明结果正确；正确性由第 6 步证明。第二，b8 从 8.92 ms 降到 3.98 ms，只看总时间会以为是 norm 变快了。第 5 步的逐层数据显示，norm 只有几 μs，真正的变化是 plugin 把 Myelin 切开后，MLP 三个矩阵乘改走普通矩阵乘层：eager engine 里它们各约 2.4 ms，ours 里三个合计 0.87 ms。

### 同一条线上的数字（decode b1，Quest A100，μs）

|  | ⑧ 只换 RMSNorm | ⑨ + 融合 | ⑩ + 按行数分档 | ⑪ + float4 | Inductor |
|---|---|---|---|---|---|
| GPU kernel 总时间（compile 下） | 298 | 303.2 | 285.5 | 278.0 | 276.4 |
| compile + CUDA Graph 实际耗时 | 223.6 | 230.7 | 216.9 | 212.4 | 211.2 |
| 单 kernel：1 行 add + RMSNorm | — | 12.9 | 5.9 | 4.4 | 2.7 |

⑧ 那列是第一次作业的 compile+ours / cg+ours；单 kernel 数来自预热后的扫描（5705706）。不同作业的 Inductor 数值在 211～212 / 275～280 之间。

### 每一阶段的瓶颈在哪

- **A**：kernel 内部——指令数、读请求并行度、分支和 barrier、对齐。工具：SASS、ncu。
- **B**：模型里的占比和 CPU 发射——Amdahl、GPU 等 CPU。工具：nsys 看 kernel 时间、CUDA event 计时。
- **C**：和编译器的融合边界、真实形状（行数）、读取宽度。工具：读生成代码、CUDA Graph 单 kernel 计时、对比 GPU 时间与实际耗时。

### 贯穿始终的方法

1. 先正确，再测快；每次改动后全部测试重跑。
2. 每次只改一个变量（先分档、再 float4），收益才分得清。
3. 先定位层次，再动手：GPU 还是 CPU？kernel 内部还是 kernel 之间？
4. 工具会改变被测对象（nsys 放大发射、profiler 拖慢 CPU、没预热的 GPU 频率低），结论用不带工具的计时确认。
5. 推断标明是推断，用实验确认；被推翻的猜测留档。

> 还没做完
>
> ：① 不用 CUDA Graph 时 compile+fused 仍慢约 34 μs，GPU 只差 1.6，几乎全在 CPU（推测是 torch.ops 调用路径，待查）；② 1 行 add + RMSNorm 单 kernel 仍比 Triton 慢 1.7 μs（候选：warp 间规约只同步 1 次）；③ 按 A100 定的行数规则要回 Orin 锁频复测；④ TensorRT 这条线（FP32 已跑通）：plugin 的 FP16、fused 版 plugin、用自己的 C++ 程序建 engine；norm1 16.5 μs 而 norm2 5.9 μs 的原因；FP32 b8 下 Myelin 为什么挑 sliced1x4 sgemm（eager engine 开 FP16 后换成 Tensor Core kernel，b8 8.92 → 2.44 ms，原因本身仍未查）。

<a id="routes"></a>

## 同一个算子怎么走进 compile 和 TensorRT

这条线从算子出发。kernel 只写一份，先注册进 PyTorch。注册后分成两条路：一条进入 `torch.compile`，全程留在 PyTorch 里运行；另一条导出 ONNX，再配上 plugin 的 `.so`，由 TensorRT 建 engine，运行时不需要 PyTorch。

```text
                 kernels.cu + launch.cuh（kernel 和选档，只依赖 CUDA）
                                 │
          ┌──────────────────────┴───────────────────────┐
          ▼                                              ▼
   ext.cu：TORCH_LIBRARY 注册                       plugin.cu：RmsNorm plugin
   ai_infra::rms_norm                              CMake → nvcc
   load() → ninja → nvcc → rmsnorm_cuda.so         librmsnorm_trt_plugin.so
          │                                              │
     ┌────┴──────────────┐                               │
     ▼                   ▼                               │
 路线一 compile       路线二 导出 ONNX                     │
 Dynamo（Meta 推形状）  tracing 真跑 CUDA 实现             │
 AOTAutograd          symbolic 翻译                      │
 Inductor 融合        ONNX 节点 ai_infra::RmsNorm         │
 Triton / cuBLAS      eps 写成属性                        │
 可选 CUDA Graph             │                           │
     ▼                       └─────────────┬─────────────┘
 在 PyTorch 里运行                          ▼
                              trtexec 加载 .so，解析 ONNX
                              creator 按名字生产 plugin
                              Myelin 在 plugin 前后分段融合
                              engine（只存 plugin 名字和 eps）
                                           ▼
                              加载 .so 和 engine，enqueue 启动 kernel
                              运行时不需要 PyTorch
```

### 第一步：写算子，并去掉框架依赖

`kernels.cu` 放 kernel，`launch.cuh` 根据 n、行数和对齐情况选版本并启动。这两个文件只依赖 CUDA，所以 PyTorch 扩展和 TensorRT plugin 都能用同一份源码。

### 第二步：注册进 PyTorch

`ext.cu` 用 `TORCH_LIBRARY` 把算子注册成 `ai_infra::rms_norm`，提供 schema、CUDA 实现和 Meta 实现。运行 Python 时，`load()` 自动完成编译和加载。注册之后，PyTorch 才认识这个算子，后面的 compile 和 ONNX 导出都从这里开始。

### 第三步：第一条路，进入 torch.compile（Quest）

1. **Dynamo**：抓 FX 图。算子不实际运行，走 Meta 实现推形状。
2. **AOTAutograd**：把 PyTorch 写法转成 aten 层的 FX 图，算子已经是注册过的节点，不会被拆开。
3. **Inductor**：融合其他算子，生成 Triton kernel，矩阵乘交给 cuBLAS。我们的算子是黑盒，只能在它前后融合。
4. **运行**：生成的 Python 文件按顺序调用各个 kernel，走到这个节点时调用我们的算子，经 dispatcher 进入 CUDA 实现。`reduce-overhead` 时再录 CUDA Graph。

因为黑盒挡住了"残差相加 + norm"的融合，我们又写了 `fused_add_rms_norm`，再加上按行数分档和 float4。Quest A100 上的结果是：decode b1 cg+fused 212.4 μs，compile_cg 211.2 μs；b8 打平。

### 第四步：第二条路，导出 ONNX（Orin）

`export_onnx.py` 先注册 symbolic，规定 `ai_infra::rms_norm` 在 ONNX 里写成 `ai_infra::RmsNorm` 节点，`eps` 写成属性。Orin 的 torch 2.1 老导出器用 tracing 真跑一遍模型，这时运行的是 CUDA 实现，所以要在 GPU 上导出。这一步只留下节点，ONNX 里没有我们的 kernel 代码。

### 第五步：同一个算子编译成 plugin

`plugin.cu` 是 TensorRT 这边的外壳：`getOutputDimensions` 对应 Meta，`enqueue` 对应 CUDA 实现，里面调用同一个 `launchRMSNorm`。CMake 调用 `nvcc` 把它编译成 `librmsnorm_trt_plugin.so`。这个 `.so` 和 PyTorch 的 `rmsnorm_cuda.so` 共用源码，但各编译一次。

### 第六步：在 TensorRT 里建 engine

`trtexec` 先加载 `.so`，`REGISTER_TENSORRT_PLUGIN` 把 creator 登记进去。Parser 解析 ONNX 时遇到 `RmsNorm`，就按名字找到 creator，创建 plugin。builder 再问 plugin 形状和类型，并优化其他节点。

**这里 plugin 本身不会被融合。**它在 TensorRT 里也是黑盒，Myelin 只能在它前后分段融合，所以 decoder layer 被切成 17 段。最后写进 engine 的只有 plugin 名字、版本和 eps，kernel 还在 `.so` 里。

### 第七步：加载并运行

运行时先加载同一个 `.so`，再反序列化 engine，plugin 用 engine 里的 eps 恢复出来。执行到这个节点时，TensorRT 调用 `enqueue`，启动和 PyTorch 同一份 kernel。Orin FP32 上，结果误差约 1e-6；decode b1 为 1.33 ms（CUDA Graph 1.21 ms），和 eager engine 的 1.32（1.25）ms 接近。

> Quest 和 Orin 的 GPU 不同，两条路的时间不能直接比。路线一和 compile_cg 比，路线二和 eager engine 比。融合版目前只进了 compile 这条路，plugin 还只有
>
> rms_norm
>
> 。在 Quest 上用
>
> dynamo=True
>
> 导出 ONNX、再到 Orin 建 engine 也是可能的做法，但 decoder layer 还没跑过。

<a id="pitch"></a>

## 2 分钟讲述稿

固定顺序：问题与公式 → 线程映射 → 规约和正确性 → 资源取舍与分档 → 对照实验与推翻的猜测 → PyTorch 接入 → 端到端与限制。

1. 问题。RMSNorm 是 LLaMA 每层都有的归一化算子，受带宽限制。PyTorch eager 写法要启动 6 个 kernel，中间结果反复写回显存。
2. 手写 kernel 和原因分析。先写"一个 block 处理一行"的版本，和 CUB 对比时发现 CUB 快很多。看 SASS 后确认原因不是 BlockReduce，而是线程数在编译期已知、循环能展开、能同时发出多个读请求。把线程数改成模板参数（own_t），速度追平 CUB。
3. 继续优化。own_t 在 n 除不尽时变慢，ncu 显示是展开后的零头分支让最后一个 warp 分叉、其他 warp 卡在 barrier。于是写 own_reg：每线程固定处理 PER_THREAD 个元素、完全展开、x 存寄存器只读一次，18 个配置中 15 个最快。
4. 按规模分档。n ≤ 256 一行一个 warp（xor shuffle，不需要 shared memory 和 barrier），n ≤ 33792 用 own_reg，更大用循环版。warp 版阈值原本 1024，扫描实测后改成 256。
5. 接入 PyTorch。写成 CUDA 扩展：检查输入、用 PyTorch 当前 stream 启动、pybind 导出。单算子比 eager 快约 4 倍，大形状达到 Orin 带宽峰值的 94%；在 A100 上和 torch.compile 生成的 Triton kernel 打平。
6. 端到端。放进 decoder layer 后 RMSNorm 只占约 2%：prefill 整层只快约 1.5%，符合 Amdahl；decode 时 GPU 有一部分时间在等 CPU 发射，融合把每层 kernel 从 34 个减到 24 个，batch 1 整层快约 10%，batch 8 只有约 2%。一开始在 nsys 下测到 20%，不开工具复测后发现是 nsys 放大了发射开销。
7. 进 torch.compile。用 TORCH_LIBRARY 注册成 `torch.ops.ai_infra.rms_norm`，补 Meta 实现，compile 能把它当成图里的正常节点。在 A100 上整层对照：decode b1 下 eager+ours −29%，但 compile+ours 反而比 compile 慢——Inductor 会把"残差相加 + RMSNorm"融合成一个 kernel，我们的算子是黑盒，挡住了这次融合。
8. 融合与按行数分档。于是写了 fused add + RMSNorm，kernel 数和 compile 一样了，GPU 时间还是多 26 μs。读 Inductor 生成的代码、再单独扫描 kernel，发现 decode 只有 1 行，我们每行 128 线程，Triton 用 512；原来的分档只看 n、是按很多行调的。加上行数维度后，CUDA Graph 下整层只比 Inductor 慢 2.4%（b1）/ 1.0%（b8）；再加 float4 向量化（一条指令读 4 个 float），GPU 时间追平（278.0 vs 276.4 μs），CUDA Graph 下 b1 只差 0.6%、b8 打平。
9. 结论。只做单个算子赢不了编译器的融合，要按真实数据流设计融合边界，并且按真实形状（尤其 decode 的小行数）分档。手写的价值在没有编译器的平台（Orin）、eager + CUDA Graph 的引擎（vLLM 式）、TensorRT plugin 和编译器做不好的复杂算子。最后把同一个 kernel 写成 TensorRT plugin（选档代码先剥离 PyTorch 依赖，两边共用），在 Orin 上结果和 PyTorch 一致；FP32 下 b8 整层还意外快了一倍，但拆开看是 plugin 把 Myelin 切开、矩阵乘换了 kernel，不是 norm 本身的功劳。下一步 FP16。

<a id="path"></a>

## 完整过程：每一步的问题、结论和证据

1. 

   A100
   
   ### 基础版与手写 own 对比 CUB
   
   基础版 15 组 PASS（最大误差 2.38e-7）。own 一个 block 处理一行、跨步循环；CUB 在 128×4096 上快 15～36%。
   
   [rmsNormWarp.cu:56 rmsNormN](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/rmsNormWarp.cu#L56) · [rmsnorm_cub.cuh:10](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/rmsnorm_cub.cuh#L10)
2. 

   A100 / Orin
   
   ### 为什么 CUB 快：编译期线程数
   
   SASS 里 own 只有 3 条 LDG，CUB 有 39～105 条。把 own 的线程数改成模板参数得到 own_t，和 CUB 差距在 4% 以内，说明与 BlockReduce 本身无关。
   
   [rmsNormWarp.cu:112 rmsNormNT](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/rmsNormWarp.cu#L112)
3. 

   Orin
   
   ### 零头分支与行不对齐
   
   展开后的零头分段让最后一个 warp 分叉，其他 warp 在 barrier 等：barrier stall 约 7 倍，活跃线程 31.6 → 27.7。n 不是 32 的倍数时每个读请求从 4 个 sector 变成约 4.44 个，padding 后恢复。
4. 

   Orin
   
   ### own_reg：x 存进寄存器
   
   模板固定 BLOCK 和 PER_THREAD、完全展开、`col < n` 屏蔽，没有零头分支；x 只读一遍。18 个配置中 15 个最快，相对 own 快 15～33%，cuobjdump 确认 LOCAL:0。
   
   [rmsNormWarp.cu:170 rmsNormReg](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/rmsNormWarp.cu#L170)
5. 

   Orin
   
   ### L2 容量悬崖，独立优化收尾
   
   工作集小于约 3.5 MiB 时数据留在 L2，有效带宽 240～253 GB/s，超过显存峰值；"n=3840 快一倍"只是那次运行恰好在 L2 里。
6. 

   Orin
   
   ### PyTorch 扩展 + warp 版 + 分档
   
   自己写了 warp 处理一行的 kernel，扩展按 n 自动选 kernel。23 个形状、160 个强制路径用例、5 个输入检查全部通过。
   
   [rmsnorm_ext.cu:37](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/torch/rmsnorm_ext.cu#L37) · [rmsNormWarp.cu:232 rmsNormWarpRow](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/rmsNormWarp.cu#L232)
7. 

   Orin / A100
   
   ### eager 为什么慢，和 torch.compile 比
   
   eager 6 个 kernel 单独看都接近带宽上限，但总共搬 448 MiB 对 128 MiB。A100 上 Inductor 生成的 Triton kernel 和我们打平：n ≤ 1024 是 persistent reduction，n ≥ 4096 是循环规约（读两遍，第二遍靠 L2）。
   
   [compile_vs_ext.py](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/torch/compile_vs_ext.py#L1)
8. 

   Orin
   
   ### decoder layer 端到端
   
   RMSNorm 占整层约 2%；不开 nsys 计时：prefill 整层快 1.5%，decode batch 1 快 10%、batch 8 快 2%，收益来自少发射 10 个 kernel（nsys 下曾测得 −20%，是工具放大了发射开销）。
   
   [decoder_layer.py:87](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/llama/decoder_layer.py#L87) · [analyze_nsys.py](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/llama/analyze_nsys.py#L1)
9. 

   Orin / A100
   
   ### TORCH_LIBRARY 注册，整层和 compile / CUDA Graphs 正面比
   
   注册 CUDA + Meta 实现，删 pybind。A100 整层 6 种写法：decode b1 eager+ours −29%、compile −30%、compile+ours −23%、compile_cg −57%；prefill 各写法差距 ≤2%。compile+ours 比 compile 慢，因为残差相加没法和我们的 RMSNorm 融合（kernel 14 → 15）。graph break 查明是扩展加载时机，修好后耗时不变。
   
   [rmsnorm_ext.cu:153 TORCH_LIBRARY](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/torch/rmsnorm_ext.cu#L153) · [compile_layer.py:25](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/llama/compile_layer.py#L25)
10. 

   Orin / A100
   
   ### fused add + RMSNorm
   
   h = x + residual 写一次，归一化直接用寄存器里的值。eager+fused 334.6 μs 已接近 compile 332.0；但 compile+fused 362.9、kernel 数同为 14，GPU kernel 时间多 26 μs，问题在 kernel 本身。Inductor 生成的代码不写 h、两遍循环、每行 512 线程。
   
   [rmsNormWarp.cu:272 fusedAddRmsNormReg](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/rmsNormWarp.cu#L272) · [rmsnorm_ext.cu:106 fused_forward](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/torch/rmsnorm_ext.cu#L106) · [decoder_layer.py:98](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/llama/decoder_layer.py#L98)
11. 

   A100
   
   ### 分档加上"行数"维度
   
   单 kernel 扫描：1 行时 128×16 是 8.25 μs、512×4 是 5.01 μs、Triton 3.24 μs；512 行时 256×8 最好。`chooseBlock(n, rows)`：行数 < 2 × SM 数每线程 4 个元素，否则 8 个。整层 GPU 差距 +25.7 → +5.4 μs，CUDA Graph 下只差 +2.4% / +1.0%。（这次扫描 norm 几列在升频前测的、偏大，预热后复测：1 行 128×16 实为 4.52 μs，结论不变。）
   
   [rmsnorm_ext.cu:10 chooseBlock(n, rows)](https://github.com/hhhhaa19/ai_infra/blob/a0cc9011c420da3db432cee729ee68d047351ee7/src/rmsnorm/torch/rmsnorm_ext.cu#L10) · [sweep_block.py](https://github.com/hhhhaa19/ai_infra/blob/a0cc9011c420da3db432cee729ee68d047351ee7/src/rmsnorm/torch/sweep_block.py)
12. 

   Orin / A100
   
   ### float4 向量化
   
   一条 LDG.E.128 读 4 个 float；n % 4 == 0 且指针对齐时自动使用，否则退回标量版。1 行 norm 2.28 μs（Triton 3.25），add+norm 4.43 μs（Triton 2.71）。整层 GPU kernel 时间 278.0 vs 276.4 μs，CUDA Graph 下 decode b1 慢 0.6%、b8 打平；不用 CUDA Graph 时仍慢约 34 μs，几乎全在 CPU。
   
   [rmsNormWarp.cu:340 rmsNormVec4](https://github.com/hhhhaa19/ai_infra/blob/a0cc9011c420da3db432cee729ee68d047351ee7/src/rmsnorm/rmsNormWarp.cu#L340) · [:404 fusedAddRmsNormVec4](https://github.com/hhhhaa19/ai_infra/blob/a0cc9011c420da3db432cee729ee68d047351ee7/src/rmsnorm/rmsNormWarp.cu#L404) · [rmsnorm_ext.cu:31 aligned16 / launchVec4](https://github.com/hhhhaa19/ai_infra/blob/a0cc9011c420da3db432cee729ee68d047351ee7/src/rmsnorm/torch/rmsnorm_ext.cu#L31)
13. 

   Orin
   
   ### ONNX → TensorRT：基线和 plugin
   
   eager 整层 engine 里 Myelin 把 RMSNorm 融成 1 个 kernel（约 7 μs）。选档 / 启动搬进不依赖 PyTorch 的 launch.cuh，ONNX 导出成 `ai_infra::RmsNorm` 节点，plugin 的 enqueue 一行调 launchRMSNorm。结果误差约 1e-6；b1 和 eager engine 持平，b8 3.98 vs 8.92 ms（来自 Myelin 被切开后矩阵乘换了 kernel）。之后整理仓库。
   
   [launch.cuh:84 launchRMSNorm](https://github.com/hhhhaa19/ai_infra/blob/992fce75dd0e040e3ae7b107b8b4ca136c15f1c8/src/rmsnorm/launch.cuh#L84) · [plugin.cu:45 enqueue](https://github.com/hhhhaa19/ai_infra/blob/992fce75dd0e040e3ae7b107b8b4ca136c15f1c8/src/rmsnorm/tensorrt/plugin.cu#L45) · [plugin.cu:89 RmsNormPluginCreator](https://github.com/hhhhaa19/ai_infra/blob/992fce75dd0e040e3ae7b107b8b4ca136c15f1c8/src/rmsnorm/tensorrt/plugin.cu#L89) · [export_onnx.py:23 symbolic](https://github.com/hhhhaa19/ai_infra/blob/992fce75dd0e040e3ae7b107b8b4ca136c15f1c8/src/llama/export_onnx.py#L23)

<a id="design"></a>

## 分档设计：一行交给多少线程

| n 的范围 | 一行交给谁 | 规约 | 模板参数 | 为什么 |
|---|---|---|---|---|
| n ≤ 256 | 1 个 warp（每 block 4 个） | `__shfl_xor_sync` 5 轮 | PER_LANE ∈ {1,2,4,8,9,16,17,32} | 行短时整个 block 会闲置；不需要 shared memory 和 barrier，可以提前 return |
| ≤ 33792 | 1 个 block（128～1024 线程） | warp shuffle → shared memory | BLOCK, PER_THREAD ∈ {1,2,4,8,9,16,17,32,33} | x 存寄存器，只读一次；n % 4 == 0 且对齐时换成 float4 版（VPT = PER_THREAD / 4 ∈ {1,2,3,4,5,8,9}，n ≤ 36864） |
| 更大 | 1 个 block（512 线程），循环 | 同上 | BLOCK = 512 | 寄存器放不下，读 x 两遍 |

### 参数怎么定

- **BLOCK**（按行数分档之前）：128～1024 中满足 BLOCK × 16 ≥ n 的最小 2 的幂。**分档后改为** BLOCK × per ≥ n，per 看行数：行数 < 2 × SM 数（A100 216 行、Orin 32 行）时 per = 4，否则 8。n=2048 时 decode（1、8 行）用 512×4，prefill（512 行）用 256×8。`sms` 用 static 变量只查一次 `multiProcessorCount`。rms_norm 和 fused_add_rms_norm 共用这条规则。
- **PER_THREAD**：档位里第一个 ≥ ⌈n / BLOCK⌉ 的值。
- **9、17、33**：接住比 2 的幂略大的 n，比如 n=4100 每线程 17 个，没有这一档就得用 32。
- **每 block 4 个 warp**：Orin 每 SM 最多 16 个 block，4 × 16 = 64 ≥ 48 个 warp 才能占满。

### 边界从哪来

- **256**：扫描实测。warp / own_reg 时间比 n=224 时 0.97、256 时 1.02、320 时 1.08；rows=16384、n=32 时 warp 版快 3.6 倍。
- **33792 = 1024 × 33**：最大一档是人为的预编译容量选择，不是硬件上限；所有 own_reg 实例 ≤ 48 个寄存器、LOCAL:0。
- switch 加宏把运行时的 n 映射到编译好的模板实例。

[chooseBlock](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/torch/rmsnorm_ext.cu#L6) · [阈值 256](https://github.com/hhhhaa19/ai_infra/blob/37b55a70ec4425921e63905bf83e543f2ef871df/src/rmsnorm/torch/rmsnorm_ext.cu#L45)

### 每一档都是同一个骨架

```text
row = …;  tid = …;  if (row >= rows) return;          // 1. 定位：哪一行、行内第几号线程
#pragma unroll
for (i = 0; i < PER; i++) {                              // 2. 读取：存进寄存器，同时累加平方
    idx = i * 线程数 + tid;
    v[i] = idx < n ? x[rowOffset + idx] : 0;
    sum += v[i] * v[i];
}
sum = 规约(sum);                                          // 3. warp 版 xor 5 轮；block 版 warp 内 + shared memory
scale = rsqrtf(sum / n + eps);                           // 4. 除以 n，不是线程数
for (…) if (idx < n) y[rowOffset + idx] = v[i] * scale * w[idx];
```

> 检查五件事：下标用 size_t；读时越界补 0、写时越界跳过；规约后谁拿到结果（down 只有 lane 0，xor 全部 lane）；除数是 n；cuobjdump 确认 LOCAL 为 0。

### 融合版：fusedAddRmsNormReg

```text
// 和 rmsNormReg 只差第一个循环；h = x + r 只写一次，归一化直接用寄存器里的 v[i]
for (int i = 0; i < PER_THREAD; ++i) {
    int col = xid + i * BLOCK;
    v[i] = col < n ? x[rowOffset + col] + r[rowOffset + col] : 0.0f;   // 读：越界给寄存器 0，不读内存
    if (col < n) h[rowOffset + col] = v[i];                             // 写：越界必须跳过
    sum += v[i] * v[i];
}
// 之后 warp 内规约 → shared 规约 → scale → y[rowOffset + col] = v[i] * scale * w[col]，和 rmsNormReg 完全一样
```

- **接口**：`fused_add_rms_norm(Tensor x, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)`，返回新张量 h、y（不原地，Meta 只要两个 empty_like）。检查三个张量 CUDA / float32 / 连续，`residual.sizes() == x.sizes()`，weight 是 [n]。只支持 n ≤ 33792（1024 × 33），没有循环版后路；对 LLaMA-3-405B 的 16384 也够。
- **接进层里**：`a = attn(norm1(x)); h, n = fused_add_rms_norm(a, x, norm2.weight, norm2.eps); return h + mlp(n)`。kernel 参数叫 x 的传的是 attn 输出、叫 r 的是原输入（加法顺序无所谓）。norm2 模块保留只为参数结构一致，`load_state_dict` 能直接加载同一组权重。
- **为什么"残差最后加"也能融合**：LLaMA 是 pre-norm，注意力子块最后的 `h = x + a` 和 MLP 子块开头的 `norm2(h)` 时间上紧挨着；MLP 最后的 `out = h + m` 后面接的是下一层的 norm1，多层叠起来也能融合（vLLM 把 residual 在层间传下去）。单层里 norm1 和最后那个 add 暂时没融合。
- **自己写 TODO 时踩的坑**：`x[...] + r` 加的是指针；`h = …` 是改指针变量本身，要写 `h[下标]`；`if (h = col < n)` 把 bool 赋给指针；`h = v[rowOffset+col]` 下标放反——h 是显存里所有行（下标 rowOffset + col），v 是这个线程自己的 PER_THREAD 个寄存器（下标 i）。条件里只放判断，赋值放在后面。

<a id="mindmap"></a>

## 我的笔记，以及后来被实测修正的地方

左边是最初思维导图里的说法，右边是现在的状态。

硬件限制：单线程寄存器 255；单 block 寄存器 65536；单 SM 寄存器 65536；单 block 线程 1024；单 SM warp 48；单 SM block 16

保留

都是 Orin（sm_87）的数字。A100：每 SM 64 个 warp、32 个 block，108 个 SM。寄存器按每线程 8 个为单位分配。

单 SM 最多 block = min(本身限制 16，总寄存器 / 每 block 寄存器，SM 线程数 / block 线程数)

保留

还要加 shared memory 的限制；occupancy 由最紧的那一项决定。

一个 row 一个 warp：每行 32 线程；PerLane = n/32 不超过 32，所以 n < 1024

修正

1024 是"能放下"的上限，不是"更快"的上限。实测 n ≥ 320 时 own_reg 更快，阈值改成 256。

每个 block 4 个 warp；block 不能跨 SM，要细粒度用满

保留

4 × 16 个 block = 64 ≥ 48 个 warp。

一个 row 一个 block：Block = N/16 向上对齐到 2 的幂；预编译档位 128、256、512

修正

档位扩展到 128～1024，own_reg 最大处理 n = 1024 × 33 = 33792。

PerThread = 16；太大时寄存器不够，同时能跑的 warp 变少，要实测

保留

BLOCK 的选法保证 PER_THREAD ≤ 16，最大档 33 只在 BLOCK = 1024 时用到。

最大限额 n > 1024 线程 × 64 个寄存器的一半就用循环版，

col += 512

，x 读两次

修正

实际上限是 33792；循环版用 rmsNormNT<512>。

Block = 512 / 1024 时 n 太小没有意义，反而 warp 要等

保留

这正是要分档的原因。

（笔记里没有）

新增

warp 版用 xor 蝶形规约，32 个 lane 都有总和，不需要广播。

PerThread = 16（只按 n 定 BLOCK）

修正

16 是"很多行、吞吐优先"时对的。decode 只有 1～8 行，GPU 只有几个 block，看延迟：每行要更多线程。现在 per 看行数：< 2 × SM 数取 4，否则 8。原来的分档解决"n 不同时怎么选"，这次补的是"行数不同时怎么选"。

<a id="data"></a>

## 实测数据

### kernel 演进（Orin，rows=512，128 线程，μs）

| 版本 | 关键改动 | n=4096 | n=4095 紧密 | n=4095 补齐 |
|---|---|---|---|---|
| own | 一个 block 一行，跨步循环 | 147.7 | 158.6 | 149.4 |
| own_t | 线程数改成编译期常量 | 120.6 | 143.5 | 129.4 |
| own_reg | 固定每线程元素数，x 存寄存器 | 109.5 | 142.0 | 110.7 |

### 扩展 vs eager（Orin，锁频，μs，含 Python 开销）

| rows × n | ours | eager | 倍数 |
|---|---|---|---|
| 4096 × 128 | 19.5 | 约 168 | CPU 开销为主 |
| 512 × 4096 | 101.1 | 370.6 | 3.7× |
| 4096 × 4096 | 702.8 | 2803.3 | 4.0× |
| 4096 × 8192 | 1395.9 | 5665.7 | 4.1×（约 192 GB/s） |

### eager 的 6 个 kernel（Orin，[4096, 4096]）

| 运算 | kernel | grid × block | μs | 带宽 |
|---|---|---|---|---|
| x.pow(2) | vectorized_elementwise<4> | 32768 × 128 | 697 | 193 GB/s |
| .mean(-1) | reduce_kernel<512,1> | 256 × (32×16) | 401 | 167 GB/s |
| + eps、rsqrt | vectorized_elementwise<4> | 8 × 128 | 4 + 3 | — |
| × r、× w（广播） | elementwise_kernel<128,2> | 65536 × 128 | 889 + 893 | 约 150 GB/s |
| **eager 合计** | 6 个 kernel，约 448 MiB |  | 2887 |  |
| **ours** | rmsNormReg<256,16>，128 MiB | 4096 × 256 | 701 | 191 GB/s |

### A100：eager vs torch.compile vs ours（μs）

| rows × n | eager | compile | compile + CUDA Graphs | ours |
|---|---|---|---|---|
| 512 × 128 | 65.8 | 48.0 | 105.7 | 8.8 |
| 4096 × 1024 | 91.8 | 49.0 | 107.6 | 23.6 |
| 4096 × 4096 | 374.0 | 103.1 | 208.0 | 107.3 |
| 4096 × 8192 | 714.9 | 200.8 | 413.8 | 205.1 |

### decoder layer（Orin，hidden 2048，FP32；占比来自 nsys 的 kernel 时间，耗时不开 nsys）

eager prefill 的整层 kernel 时间构成

| 场景 | eager wall | ours wall | 整层 | RMSNorm 占比 | 收益来源 |
|---|---|---|---|---|---|
| prefill b1 s512 | 19430 μs | 19145 μs | −1.5% | 2.0% | 少搬运（Amdahl 预测 −1.4%）；nsys 下 −1.2% |
| decode b1 | 1436 μs | 1290 μs | −10% | 2.5% | 少发射 10 个 kernel；nsys 下 −23% |
| decode b8 | 1775 μs | 1742 μs | −2% | 2.3% | kernel 变大，GPU 基本不空闲；nsys 下 −20% |

> 表中耗时是不开 nsys、用 CUDA event 测的（3 个独立进程，差异 ≤0.3%）。在 nsys 下测得 decode −23% / −20%，是 nsys 放大了每次发射的开销。教训：工具本身会改变被测对象，结论要用不带工具的计时确认。

### Quest A100 整层：eager / compile / CUDA Graphs（μs）

A100-PCIE-40GB、torch 2.14.1+cu130，hidden 2048、FP32。CUDA event 计时：每样本 N 次前向（prefill 10、decode 50）÷ N，30 样本中位数，3 轮轮换顺序；kernel 数来自 torch.profiler 单独跑的 1 次前向。Quest 无法锁频，频率未记录。第一次运行：

| 形状 | eager | eager+ours | compile | compile+ours | compile_cg | cg+ours |
|---|---|---|---|---|---|---|
| prefill b1 s512 | 3613.0 (26) | 3580.1 (16) | 3564.3 (14) | 3545.0 (15) | 3561.6 (15) | 3564.5 (16) |
| decode b1 | 487.7 (26) | 346.5 −29% | 340.8 −30.1% | 377.5 −22.6% | 210.8 −56.8% | 223.6 −54.1% |
| decode b8 | 520.6 (33) | 414.0 −20.5% | 401.1 −23.0% | 414.6 −20.3% | 387.4 −25.6% | 396.8 −23.8% |

括号里是每次前向的 kernel 数。Quest 上 eager 每层 26 个（Orin torch 2.1 下是 34 个）。decode b1 GPU kernel 合计：compile 275 μs、compile+ours 298 μs；cg 两列多一个 `multi_tensor_apply`（把输入拷到录制地址）。修好 graph break 后重跑：eager 3622 / 499 / 526 μs，decode b1 eager+ours −28.6%、compile −31.8%、compile+ours −24.5%、compile_cg −57.7%、cg+ours −55.1%，结论不变。

### 加入 fused（decode b1，μs，括号为 kernel 数）

| RMSNorm 实现 | eager 系 | compile 系 | compile_cg 系 |
|---|---|---|---|
| PyTorch 原生 | 481.4 (26) | 332.0 (14) | 211.0 (15) |
| ours | 342.1 (16) | 365.5 (15) | 223.5 (16) |
| fused | 334.6 (15) | 362.9 (14) | 230.7 (15) |

GPU kernel 合计：compile 277.5、compile+fused 303.2、eager+ours 305.1、eager+fused 307.4 μs。decode b8：compile 406.2 vs compile+fused 419.2（+3.2%），compile_cg 393.5 vs cg+fused 407.8；prefill compile 3618.2 vs compile+fused 3613.3（噪声内）。三次独立作业 compile+ours − compile 都是 +34～37 μs。

### 单 kernel 扫描（n=2048，CUDA Graph 重放 100 次 × 20 取中位数，μs）

| 行数 | 算子 | 128×16（原） | 256×8 | 512×4 | 1024×2 | Triton |
|---|---|---|---|---|---|---|
| 1 | RMSNorm | 8.25 | 5.96 | 5.01 | 5.00 | 3.24 |
| 8 | RMSNorm | 8.52 | 6.11 | 5.13 | 5.12 | 4.31 |
| 64 | RMSNorm | 9.16 | 6.69 | 5.72 | 5.86 | 4.56 |
| 512 | RMSNorm | 11.36 | 9.12 | 11.46 | 13.43 | 7.34 |
| 1 | add + RMSNorm | 12.97 | 8.04 | 5.92 | 5.30 | 3.62 |
| 8 | add + RMSNorm | 13.42 | 8.39 | 6.12 | 5.49 | 5.07 |
| 64 | add + RMSNorm | 14.19 | 8.88 | 6.62 | 6.26 | 5.41 |
| 512 | add + RMSNorm | 14.18 | 10.76 | 12.69 | 13.29 | 6.64 |

脚本用 `load_inline` 单独编译一个可指定 BLOCK 的入口，不改仓库代码。开始时 GPU 空闲 210 MHz，结束时 1410 MHz；最先测的"1 行、128×16"可能在升频中，但 8 行那组（Triton 编译后才测）8.52 μs 与之接近，判断影响不大。decode b1 两个 kernel 合计 21.2 vs Triton 6.9 μs（差 14 μs），比整层里的 26 μs 小，可能和真实层里输入刚由矩阵乘写出、缓存状态不同有关（推测）。融合版 128×16 特别慢：每线程串行 16 次读 x、16 次读 r、16 次写 h。选最好的切法仍比 Triton 慢 1.5～2 μs，候选原因：没用 128 位向量化读取；warp 间用 shared 树形规约，每层一次 `__syncthreads`（512 线程要 4 轮）。

### 按行数分档后（μs）

| 对比 | 改分档前 | 改分档后 |
|---|---|---|
| decode b1 GPU kernel：compile / compile+fused | 277.5 / 303.2（+25.7） | 280.1 / 285.5（+5.4） |
| decode b8 GPU kernel 差距 | +18.2 | +2.5 |
| decode b1：compile_cg / cg+fused | 211.0 / 230.7（+19.7） | 211.8 / 216.9（+5.1，+2.4%） |
| decode b8：compile_cg / cg+fused | 393.5 / 407.8（+14.3） | 388.5 / 392.5（+4.0，+1.0%） |
| decode b1：compile / compile+fused（实际耗时） | 332.0 / 362.9 | 361.8 / 397.1（+35） |
| prefill：compile / compile+fused | 3618.2 / 3613.3 | 3555 / 3542 |

> 不用 CUDA Graph 时，GPU 只多 5 μs、实际多 35 μs，剩下约 30 μs 在 CPU 上。推测：Inductor 调 Triton 走预先准备好的轻量 launcher、缓冲区自己提前分好；调我们的算子要走完整 torch.ops 路径（Python → dispatcher → forward 里一串 TORCH_CHECK → 两次 empty_like），一层两次。
>
> 未验证
>
> 。这次分到的节点 CPU 偏慢（eager 519.8 vs 上次 481.4 μs），把 CPU 开销放大了。CUDA Graph 会去掉这部分，生产中 decode 一般用 CUDA Graph，所以 cg 列最接近生产。

### float4 之后（计时前先让 GPU 升到 1395 MHz）

| n=2048 单 kernel（μs） | 标量 128×16 | 标量 512×4 | 标量 1024×2 | float4 256×8 | float4 512×4 | Triton |
|---|---|---|---|---|---|---|
| RMSNorm，1 行 | 4.52 | 2.75 | 2.75 | 2.29 | 2.28 | 3.25 |
| add + RMSNorm，1 行 | 12.88 | 5.91 | 5.29 | 4.73 | 4.43 | 2.71 |
| RMSNorm，8 行 | 6.43 | 3.84 | 3.82 | 3.24 | 3.21 | 3.31 |
| add + RMSNorm，8 行 | 10.12 | 4.60 | 4.11 | 3.72 | 3.48 | 3.82 |
| RMSNorm，64 行 | 6.84 | 4.24 | 4.24 | 3.61 | 3.77 | 3.39 |
| add + RMSNorm，64 行 | 10.62 | 4.95 | 4.52 | 4.09 | 4.06 | 3.51 |
| RMSNorm，512 行 | 7.48 | 7.68 | 8.89 | 5.42 | 6.20 | 4.73 |
| add + RMSNorm，512 行 | 9.15 | 8.37 | 8.64 | 5.81 | 6.58 | 6.65 |

标量 256×8：1 行 3.28 / 8.05，8 行 4.61 / 6.34，64 行 5.00 / 6.65，512 行 6.21 / 7.07（norm / add+norm）。float4 128 线程（每线程 4 个 float4）略。

| 整层（μs） | compile | compile+fused | compile_cg | cg+fused | eager+fused |
|---|---|---|---|---|---|
| decode b1 | 363.9 | 397.8 | 211.2 | 212.4（+0.6%） | 364.9 |
| decode b8 | 412.5 | 448.8 | 395.0 | 395.2（打平） | 410.4 |
| prefill b1 s512 | 3604.9 | 3603.7 | 3621.3 | 3614.9 | 3604.6 |
| decode b1 GPU kernel 总时间 | 276.4 | 278.0 | 275.5 | 279.9 | 285.2 |

> 读法：float4 对 1 行最有用（add+norm 5.91 → 4.43），行多时也普遍快 15～30%。和 Triton 比有输有赢：1、8 行的 norm 我们更快，1 行 add+norm 仍慢 1.7 μs，512 行 norm 慢、add+norm 快。整层 GPU 时间已追平；不用 CUDA Graph 时 compile+fused 多出的约 34 μs 几乎全在 CPU。
>
> 更正
>
> ：这次扫描的 norm 几列在 GPU 升频前测的，偏大（1 行 128×16：8.25 → 4.52 μs）；add+norm 几列（12.97 vs 12.88）不受影响，分档结论不变。教训：计时前要先让 GPU 跑起来，并记录开始 / 预热后 / 结束三次频率。

### Orin TensorRT（FP32 + noTF32，锁频，ms）

| 形状 | eager engine：trtexec | trt_check.py | + CUDA Graph | ours engine（plugin） | ours + CUDA Graph |
|---|---|---|---|---|---|
| decode b1 | 1.309 | 1.311 | 1.250 | 1.33 | 1.21 |
| decode b8 | 8.882 | 9.030 | 8.778 | 3.98 | — |
| prefill s512 | 14.790 | 14.789 | 14.745 | 14.99 | — |

ours 列来自之后另一次运行，那次 eager engine 是 1.32（1.25）/ 8.92 / 14.84，和左边一致。正确性：两个 engine 和 PyTorch 输出比，最大绝对误差约 1～2e-6、最大相对误差约 3～6e-4（FP32 计算顺序不同）。eager engine b1 的 kernel（nsys）：gemvx 3 个 × 342 μs + 1 个 288 μs、cutlass sgemm 104 μs、两个 norm 6.9 / 7.1 μs、silu×up 5.0、掩码 + softmax 3.2；b8：sliced1x4 sgemm 3 个 × 2436 μs、gemv2N 2 个 × 760 μs。ours engine 17 段：attention 段 b8 3.05 ms，MLP 三个矩阵乘 b8 共 0.87 ms，`/Add_1` 和两次 reshape 拷贝各约 6 μs，norm2 5.9 μs、norm1 16.5 μs（b1）。同期 PyTorch（bench_layer.py，6 种写法）只记下了 decode b8 eager 1.80 ms；其余 PyTorch 同期对比这次跳过。

### Orin TensorRT FP16 基线（eager ONNX，--fp16 --noTF32，锁频 1300.5 MHz，ms）

| 形状 | trtexec | trtexec + CUDA Graph | trt_check.py | trt_check.py + CUDA Graph | FP32 eager engine |
|---|---|---|---|---|---|
| decode b1 | 0.687 | 0.622 | 0.690 | 0.632 | 1.32 |
| decode b8 | 2.420 | 2.323 | 2.442 | 2.332 | 8.92 |
| prefill s512 | 2.370 | 2.312 | 2.314 | 2.319 | 14.84 |

这是 TensorRT 自己融合的 eager engine，RMSNorm 仍由 Myelin 生成（b8 FP16 下约 6.0 / 5.5 μs），不是我们的 plugin；plugin 只接受 kFLOAT，FP16 版还没写。误差和 kernel 名单见 [模块 9 · FP16 基线](m9.md#fp16)。

<a id="lessons"></a>

## 踩坑与方法

### 被推翻的猜测

- "CUB 让更多 warp 就绪"：ncu 显示 CUB 的 eligible 反而更低。
- "256 线程不受零头影响"：错，只是 rows=512 时被"不对齐反而更快"抵消。
- "行间距是 2 的幂导致冲突"：错，4128 和 4096 一样快。
- "n=3840 比 4096 快一倍"：那次运行数据恰好在 L2 里。
- "kernel replay 的 ncu 能看热缓存"：不能，它会打乱 L2。
- "warp 版到 1024 都更快"：320 起就更慢。
- "graph break 让 compile+ours 变慢、cg 要录两段"：切断只发生在脚本里第一次 explain，修好后耗时不变，变慢只因少了一次融合。
- "Inductor 的融合 kernel 写出 h"：读生成代码后发现它不写 h，最后用 x + attn + mlp 重算。
- "128×16 对所有行数都合适"：只在很多行时对；1 行时它在所有切法里最慢。
- "warp 版能救 1 行的情况"：方向反了，warp 版每行只有 32 线程，适合 n 小、行多。
- "1 行 128×16 norm 要 8.25 μs"：那组是在 GPU 升频前测的，预热后实为 4.52 μs。
- "插入 plugin 后 b8 快一倍是因为我们的 norm 快"：逐层看，norm 只差 1 μs 左右，快的是被切出 Myelin 的 MLP 矩阵乘。
- "engine 显示 1 层就是 1 个 kernel"：ForeignNode 内部仍启动很多个 kernel，要用 nsys 看。

### 还没解决的问题

- 不用 CUDA Graph 时 compile+fused 比 compile 多约 30 μs CPU 时间，是不是 torch.ops 调用路径（待测）。
- float4 之后，1 行 add+norm 仍比 Triton 慢 1.7 μs（4.43 vs 2.71），而 1 行 norm 我们更快：候选是 warp 间规约改成只同步 1 次，未验证。
- 单 kernel 扫描差 14 μs，整层里差 26 μs，多出来的部分从哪来（推测是缓存状态）。
- 按 A100 数据定的行数规则，在 Orin 上很多行时（变成 256×8）是快是慢，需要锁频后测。
- TensorRT FP32 + 禁 TF32 时，Myelin 为什么在 b8 给矩阵乘挑了 sliced1x4 sgemm（每个 2.4 ms，PyTorch 整层才 1.8 ms）。开 FP16 后换成 h16816gemm 系列，b8 2.44 ms，但仍慢于 PyTorch FP32 的 1.80 ms（不同时间测的）；开 TF32 没测。
- ours engine 里 norm1 16.5 μs、norm2 只要 5.9 μs，差在哪（未查）。
- 不对齐为什么让 own_t / cub 在 rows=512、256 / 512 线程时更快。
- own_reg 128 线程为什么把对齐代价放大到约 27%，256 / 512 线程只有约 2%。
- n=3840 的 L2 悬崖更早、更不稳定（猜测组冲突或物理页分配）。
- warp 版在 n 较大时变慢（候选：每线程 56 个寄存器）。

<a id="qa"></a>

## 面试追问

<a id="qa-count"></a>

<a id="q1"></a>

**问：kernel RMSNorm 和 LayerNorm 有什么区别？**

RMSNorm 不减均值、没有 bias，只用均方根缩放，只需一次规约（平方和）；LayerNorm 要算均值和方差。计算更少，在 LLM 里效果接近，所以 LLaMA 等模型都用它。

<a id="q2"></a>

**问：kernel 为什么说它受带宽限制？理论上限怎么算？**

每个元素只做几次乘加，却要读 4 字节、写 4 字节。理论最短时间 = 必须搬的字节数 ÷ 带宽。rows=4096、n=8192 时 x+y 共 256 MiB ≈ 268 MB，÷ 204.8 GB/s ≈ 1.31 ms；实测 1.396 ms，约峰值的 94%（有效字节口径，不含权重读取）。

<a id="q3"></a>

**问：kernel CUB 为什么比你最初的版本快？**

不是因为 BlockReduce。CUB 的线程数是模板参数，编译器能展开循环、一次发多个读请求、指令更少；SASS 里 own 3 条 LDG，CUB 39～105 条。把自己的线程数也改成模板参数（own_t）后速度一致。结论限定在已测配置。

<a id="q4"></a>

**问：kernel n 除不尽时为什么变慢？怎么解决？**

展开后的零头分段让最后一个 warp 分叉，其他 warp 在 `__syncthreads()` 等：barrier stall 约 7 倍，活跃线程 31.6 → 27.7。另外 n 不是 32 的倍数时行起点不对齐，sector / 请求从 4 变 4.44。own_reg 每线程固定元素数、用 `col < n` 屏蔽，没有零头分支；padding 解决对齐。

<a id="q5"></a>

**问：kernel 为什么把 x 存在寄存器里？有什么风险？**

求平方和读一遍、算输出又读一遍；存寄存器就只读一次。风险是寄存器不够溢出到 local memory（实际在显存）反而更慢，寄存器多也降低 occupancy。所以 PER_THREAD 有上限，并用 cuobjdump 确认 LOCAL:0。

<a id="q6"></a>

**问：kernel shfl_down 和 shfl_xor 有什么区别？**

down 规约完只有 lane 0 有总和，要再广播；xor 是蝶形交换，5 轮后 32 个 lane 都有总和，每个 lane 可以直接算 scale 写自己的输出。

<a id="q7"></a>

**问：kernel 为什么要按 n 分档？阈值怎么定的？**

n 小用整个 block 会闲置线程并付 shared memory 和 barrier 的代价；n 大要更多线程才读得快；寄存器放不下只能循环。阈值靠扫描：强制两种实现，rows 4096 / 16384，n 从 32 扫到 384，找时间比交叉点，结果定为 256。

<a id="q8"></a>

**问：硬件 warp 版为什么每个 block 放 4 个 warp？**

Orin 每 SM 最多 16 个 block、48 个 warp。每 block 1 个 warp 最多 16 个 warp，占用率只有三分之一；4 个 warp 时 16 个 block 就有 64 个，能填满 48 个。

<a id="q9"></a>

**问：硬件 occupancy 由什么决定？**

每 SM 能驻留的 block 数取最小值：硬件 block 上限（Orin 16、A100 32）；SM 线程上限 ÷ 每 block 线程数；65536 寄存器 ÷ 每 block 寄存器（按分配粒度）；shared memory ÷ 每 block 用量。occupancy 高不一定更快。

<a id="q10"></a>

**问：硬件 为什么有些形状的有效带宽超过显存峰值？**

数据留在了 L2。连续运行时 x+y 小于 L2（Orin 实际约 3.5 MiB，A100 40 MB），下一次直接从 L2 读。这类形状不能用来评价 kernel。A100 的大 L2 也让"读两遍"代价很小，所以 own_reg 只读一次的优势在那里体现不出来。

<a id="q11"></a>

**问：PyTorch 怎么把 CUDA kernel 接进 PyTorch？**

- TORCH_CHECK：GPU、float32、weight 形状 [n]、连续；
- empty_like 分配输出，data_ptr 取指针；
- 在 getCurrentCUDAStream() 上启动，保证顺序；
- C10_CUDA_KERNEL_LAUNCH_CHECK 检查启动错误；
- 导出：最初用 PYBIND11，后来改为 TORCH_LIBRARY 注册 CUDA + Meta 实现，调用 `torch.ops.ai_infra.rms_norm`；load() 只看列出的源文件，所以把被 include 的 kernel 文件哈希加进编译选项。

<a id="q12"></a>

**问：PyTorch eager 写法为什么慢？4 倍从哪来？**

6 个 kernel 单独看都接近带宽上限，但中间结果都要写回显存再读出，总搬运 448 MiB 对 128 MiB；广播乘法没向量化；小形状还要付 6 次调用开销。4 倍主要来自融合，和 kernel 内部优化（own → own_reg）要分开说。

<a id="q13"></a>

**问：PyTorch torch.compile 生成的 kernel 是什么样的？**

Dynamo 记录图，Inductor 融合后生成 Triton。n ≤ 1024：persistent reduction，一个 program 处理 XBLOCK 行，整行进寄存器，tl.sum 规约，思路同 own_reg。n ≥ 4096：循环规约，第一遍带 evict_last 求平方和，第二遍再读 x。A100 上和我们打平。

<a id="q14"></a>

**问：PyTorch 有了 torch.compile 还需要手写 kernel 吗？**

简单融合编译器已能做到同样水平，整层编译时还能把残差加法和下一个 norm 融合。实测：只替换 RMSNorm，compile 下反而慢（decode b1 341 → 378 μs）；写了 fused add + RMSNorm 并按行数分档后，CUDA Graph 下只差 2.4%。所以只做单个算子赢不了编译器的融合，要按真实数据流设计融合边界、按真实形状分档。手写的价值在：没有 Triton 的平台（Orin 的 torch 2.1，eager+fused 已接近 compile）、eager + CUDA Graph 的推理引擎（vLLM 式）、TensorRT plugin、编译器做不好的复杂算子（FlashAttention、量化 GEMM）。

<a id="q15"></a>

**问：PyTorch 自定义算子直接放进 torch.compile 会怎样？怎么支持训练和多卡？**

pybind 导出的函数 Dynamo 不认识，会切断计算图。要用 TORCH_LIBRARY 注册成 torch.ops.xxx 并提供只推形状的 Meta 实现，vLLM 就是这样做的；已完成，explain 确认整层 1 张图。注意扩展要在 compile 之前加载，否则加载代码本身会造成 graph break。即使不切断，自定义算子在 Inductor 眼里仍是黑盒，不能和前后融合。训练需要反向 kernel + autograd；多卡需要 device guard 和同设备检查，目前都没有。

<a id="q16"></a>

**问：端到端 算子快了 4 倍，整个模型快了多少？**

decoder layer 里 RMSNorm 只占约 2%，矩阵乘法 95% 以上。prefill 整层快约 1.5%（Amdahl 预测 1.4%，p=2%，s=3.5）；decode batch 1 快约 10%、batch 8 约 2%，原因是少发射 10 个 kernel，而不是 kernel 时间。最初在 nsys 下测到 20%，不开工具复测后更正。

<a id="q17"></a>

**问：端到端 decode 时 GPU 为什么会空闲？**

CPU 把 kernel 放进队列、GPU 按顺序执行，两者异步。decode 每个 kernel 只要几 μs，CPU 发射一个要几十 μs，GPU 做完只能等。不开 nsys 时，Orin 上 eager decode b1 每层 1436 μs，kernel 合计约 1264 μs，GPU 约 12% 时间空闲；ours 只多约 30 μs。batch 8 时 kernel 变大，几乎不空闲。判断方法：比较实际耗时和 kernel 合计。

<a id="q18"></a>

**问：端到端 CUDA Graphs 是什么？有什么限制？**

把一串 kernel 录成图，之后一次 cudaGraphLaunch 重放。kernel 不变快，只去掉 CPU 发射开销，所以 decode 收益大、prefill 几乎没用。限制：地址固定（新输入要拷进录制时的缓冲区）、形状固定（vLLM 为常见 batch 各录一张）、没有 CPU 分支。

<a id="q19"></a>

**问：端到端 decode 为什么 batch 8 的 kernel 时间只比 batch 1 多一点？**

decode 的 kernel 时间主要在读权重：一层 205 MB，按约 190 GB/s 读一遍约 1.08 ms，实测 1.26 ms。batch 8 仍只读一遍权重，8 个 token 共用，所以只到 1.73 ms。这就是推理服务要批处理的原因。

<a id="q20"></a>

**问：测量 你怎么保证性能数据可信？**

- 先验证正确性；
- Orin 锁频并记录频率；
- 预热，CUDA event 包住多次调用，多样本取中位数，多轮交替；
- 关键结论 3 个独立进程确认；
- 算有效带宽和峰值比，超过峰值说明在 L2；
- 知道工具偏差：ncu 冷缓存基础频率，nsys 放大发射开销。

<a id="q21"></a>

**问：测量 举一个你被数据推翻的猜测。**

曾看到 n=3840 比 4096 快一倍，以为和 n 有关。3 个独立进程重复扫描后只有部分运行出现，原因是那次工作集刚好留在 L2。之后把"避开接近 L2 大小的形状"作为规则。warp 版阈值 1024 → 256 也是类似例子。

<a id="q22"></a>

**问：kernel 如果继续优化，你会做什么？**

- FP16 / BF16（实际推理精度，搬运减半）；
- warp 间规约改成一次同步：部分和放进 shared，再用一个 warp shuffle；
- 查清不用 CUDA Graph 时调用自定义算子的 CPU 开销；
- 层间也融合（上一层最后的 add + 下一层 norm1）、silu * mul 融合；
- TensorRT 这条线：FP16（half 读写、float 累加）、fused 版 plugin、INT8。

已完成：残差加法 + RMSNorm 融合、TORCH_LIBRARY 注册并接入 compile 和 CUDA Graphs、按行数分档、float4 向量化、TensorRT plugin FP32。

<a id="q23"></a>

**问：PyTorch 你的算子在 torch.compile 下为什么反而变慢？后来怎么追回来的？**

- Inductor 把"残差相加 + norm2"融合成一个 Triton kernel；我们的算子是黑盒，残差只能单独一个 kernel，kernel 14 → 15，decode b1 341 → 378 μs。
- 写 fused add + RMSNorm 后 kernel 数持平，但 GPU 时间还多 26 μs，说明问题变成 kernel 本身。
- 读 inductor_cache：Triton 每行 512 线程（num_warps=16），我们 128；单 kernel 扫描证实 1 行时 128×16 最慢。分档加行数维度后，GPU 差距 26 → 5 μs，CUDA Graph 下整层只差 2.4%；再加 float4 后 GPU 时间 278.0 vs 276.4 μs，CUDA Graph 下 b1 只差 0.6%、b8 打平。
- 排除频率：同作业交替测、三次独立作业差距稳定、只多在 RMSNorm 部分（间接证据）。

<a id="q24"></a>

**问：PyTorch torch.compile 从调用到执行经过哪几步？融合发生在哪？**

torch.compile 只包一层；第一次调用时 Dynamo 用 FakeTensor 逐行读 forward 抓 FX 图和 guards（自定义算子靠 Meta 推形状，非张量代码会 graph break）→ AOTAutograd 翻译成核心 aten 算子 → Inductor lowering、scheduler（融合在这里）、codegen 出 Triton，矩阵乘交给 cuBLAS → reduce-overhead 时再录 CUDA Graph。缓存按 guards 存多份，默认上限 8。Dynamo 管图完不完整，Inductor 管 kernel 少不少，CUDA Graph 管 CPU 发射开销。

<a id="q25"></a>

**问：kernel 为什么分档不能只看 n，还要看行数？**

行数 = batch × seq，决定一次启动多少 block。decode b1 只有 1 行、整张 A100 只有 1 个 block，速度由延迟决定：128 线程每个串行读 16 个数，比不过 512 线程每个读 4 个（8.25 vs 5.01 μs）。行多时 GPU 已满，block 小一点能让一个 SM 同时放几个、更好地藏延迟（512 行时 256×8 最好）。同一模型 hidden 固定，行数每一步都在变，所以两个维度都要看。之前单算子测试都是几千行，看不出来，放进整层测 decode 才暴露。

<a id="q26"></a>

**问：端到端 eager+ours、compile、compile_cg 在 decode 下的收益分别来自哪？**

都是在减少 CPU 发 kernel 的开销。eager+ours：RMSNorm 从约 5 个 kernel 变 1 个，一层少 10 个，每个约 14 μs，−29%；compile：Inductor 自动融合，约 14 个 kernel，−30%；compile_cg：再加 CUDA Graph，整串一次重放，CPU 开销基本消失，−57%。prefill 时 GPU 被矩阵乘占满、CPU 发射和计算重叠，所有写法差距 ≤2%。

<a id="q27"></a>

**问：kernel float4 向量化是什么？有什么前提？**

把指针当 float4* 读，一条 LDG.E.128 读 16 字节（4 个 float）进寄存器：数据量不变，读指令变成四分之一，延迟受限和带宽受限时都有帮助。线程按 float4 跨步，一个 warp 一条指令读连续 512 字节，仍是合并访问。前提：地址 16 字节对齐（n % 4 == 0 才能保证每行开头对齐，切片可能不对齐），否则 misaligned address；所以 C++ 先检查，不满足就退回标量版。用 cuobjdump 看 SASS 确认有 LDG.E.128。实测 1 行 add+norm 5.91 → 4.43 μs，整层 GPU 时间追平 Inductor。Triton 从 tt.divisibility=16 知道对齐，自动就做了。

<a id="q28"></a>

**问：测量 你怎么判断时间花在 GPU 还是 CPU？举个例子。**

同时看实际耗时和 GPU kernel 总时间（profiler / nsys 的 kernel 时间可信）。float4 之后 compile+fused 和 compile：实际 397.8 vs 363.9 μs，kernel 总时间 278.0 vs 276.4 μs——GPU 只差 1.6，剩下约 32 μs 在 CPU。再用 CUDA Graph 去掉 CPU 发射：cg+fused 212.4 vs 211.2，差距几乎消失，印证了这一点。单 kernel 比较则把同一个 kernel 录进 CUDA Graph 连续 100 次，彻底排除 CPU；计时前先让 GPU 升频（有一次扫描就因为没预热，把第一组数测大了近一倍）。

<a id="q29"></a>

**问：端到端 你是怎么把自己的算子接进 TensorRT 的？**

- 先把选档和启动代码从 PyTorch 扩展里剥出来，放进只依赖 CUDA 的 launch.cuh，PyTorch 扩展和 plugin 共用（不带 PyTorch 路径单独 nvcc 编译验证）；
- 导出时写 symbolic：`ai_infra::rms_norm` → ONNX 自定义节点 RmsNorm，eps 作属性；老导出器追踪时真的跑 CUDA kernel，所以在 GPU 上导出；
- plugin（IPluginV2DynamicExt）：getOutputDimensions 对应 Meta，supportsFormatCombination 声明只要 FP32 + 连续，enqueue 里调 launchRMSNorm，serialize 存 eps；creator 是工厂，REGISTER_TENSORRT_PLUGIN 让 .so 一加载就登记；
- CMake 编成只链接 CUDA 和 TensorRT 的 .so，trtexec `--plugins=` 建 engine；用导出时存的同一个输入对 PyTorch 结果，误差约 1e-6。

<a id="q30"></a>

**问：端到端 加了 plugin 以后 b8 快了一倍，是你的 kernel 快吗？**

不是。逐层看 norm 只有几 μs，和 Myelin 自己生成的差 1 μs 左右。原来 Myelin 把整层包成一个节点，在 FP32、禁 TF32、M=8 下给 MLP 矩阵乘选了 sliced1x4 sgemm，每个 2.4 ms；plugin 是黑盒，Myelin 被切成 17 段，MLP 矩阵乘交给普通矩阵乘层后只要 0.87 ms。代价是多了独立的 Add 和 reshape 拷贝（各约 6 μs）。而且两个 engine 在 b8 都比 PyTorch eager（1.80 ms）慢，真正的问题是矩阵乘的 kernel 选择（推测，待开 TF32 / FP16 验证）。这和 compile 下"我们的算子挡住融合"是同一类现象：插入黑盒会改变编译器的整体决策。

<a id="q31"></a>

**问：kernel 标量版每个线程也读 4 个 float，为什么 float4 还有收益？**

标量版是一个线程发 4 条 `LDG.E`、分 4 次读（线程 0 读 x[0]、x[512]、x[1024]、x[1536]）；float4 是一条 `LDG.E.128` 一次读 x[0..3]。数据量一样，读指令差 4 倍；一个 warp 一条指令从 128 字节变成 512 字节。每条读指令都有发出、追踪、等数据的固定开销：1 行时速度看延迟，指令少等待轮数就少；行多时每条指令带更多数据，更容易用满带宽。Quest n=2048、1 行 add+norm：标量 512×4 5.91 μs，float4 512×4 4.43 μs。它和 sector 对齐是两件事：对齐决定这段数据跨几个 32 字节块，float4 决定读这段数据要发几条指令；float4 还要求 16 字节对齐。

<a id="q32"></a>

**问：端到端 导出那边和 plugin 那边怎么对上？写错会怎样？**

两段代码互不调用，只靠 ONNX 里两个字符串：节点名 `RmsNorm`（symbolic 写 `g.op("ai_infra::RmsNorm", …)`，plugin 的 `getPluginName()` 返回 "RmsNorm"、版本 "1"）和属性名 `eps`（导出写 `eps_f`，creator 按 "eps" 读）。加载 .so 时 creator 登记；Parser 遇到不认识的 RmsNorm 就按名字查注册表、调 `createPlugin`；engine 里只存 plugin 名字、版本和 eps。节点名写错：导出成功，建 engine 时报找不到 plugin；没加载 .so：建或加载 engine 时失败；属性名写成 epsilon：可能不报错，悄悄用默认值 1e-6，结果只差一点，所以必须和 PyTorch 输出比误差。

<a id="q33"></a>

**问：端到端 TensorRT 给矩阵乘挑了慢的 kernel，你能怎么办？**

- 先证明问题在哪：`--exportLayerInfo`、`--dumpProfile`、nsys 看实际 kernel；b8 下 PyTorch eager（cuBLAS）1.80 ms，说明硬件本身没问题；
- 验证原因：FP16 或开 TF32 重建，看 kernel 和时间变不变。用 eager ONNX 开 FP16 重建：矩阵乘换成 Tensor Core 的 h16816gemm，b8 8.92 → 2.44 ms，说明和 kernel 选择有关（TF32 没测）；
- 调整模型结构，比如改 ONNX 图、切分节点，影响 Myelin 分组；换 TensorRT 版本或平台比较；
- 某个矩阵乘特别值得时自己写 plugin；或用 ONNX Runtime CUDA EP 跑同一个 ONNX 作开源对照，隔离问题是否在 TensorRT。

之前 b8 变快恰好是 plugin 切开了 Myelin，不是主动设计，不能当成稳定方法，真实项目里要把这种依赖写进风险。

<a id="check"></a>

## 口述验收标准

### 五项

1. 不看笔记 90 秒讲完主线。
2. 5 分钟画出 Tensor → kernel 的调用链和行 / warp / block 映射。
3. 随机追问能说明原因，而不是背阈值。
4. 每个数字能说清配置、基线和证据。
5. 能主动区分事实、推断和未完成项。

### 复习方法

- 先完整讲一遍，再集中纠错，不拆成碎片问题。
- 反馈看三项：事实准确、因果有证据、结构连贯。
- 不要求背源码或全部数字；点开源码索引能找到对应位置即可。

状态：还没通过口述验收。第 1～4 点做过一轮分点口述，第 5 点和不看笔记的完整口述还没做。

### 分点口述时容易说错的地方

这次按 5 点分段讲，每讲完一点集中纠错。下表是需要改的地方，右边是改正后的说法。

① 公式写成

(sum(x*x)/n + eps) * x * w

修正

少了平方根倒数：

y = x * rsqrt(sum(x²)/n + eps) * w

。手写的原因也要讲完整：eager 6 个 kernel 单个都接近带宽上限，但中间结果反复写回显存（Orin [4096,4096]：448 MiB / 2887 μs 对 128 MiB / 701 μs），目标是 x 读一次、y 写一次。

① "torch.compile 以后可以导出 ONNX""GGUF 会对算子优化"

修正

compile 和 ONNX 导出是两条路，导出不是 compile 的下一步。GGUF 只是模型文件，优化发生在读它的引擎里（llama.cpp）。

② "编译期常量展开循环、一次发更多请求；数据存寄存器"

补全

方向对，但缺 own 的起点（比 CUB 慢 15～36%）和证据（ncu 推翻 eligible 猜测，SASS 3 条 vs 39～105 条 LDG）；寄存器要加风险（溢出到 local memory，用 cuobjdump 确认 LOCAL:0）；还漏了按 n 和按行数分档、float4。

② "线程会有无法对齐的问题"

修正

是两个问题：零头分支让最后一个 warp 分叉、其他 warp 在屏障等（barrier stall 约 7 倍，活跃线程 31.6 → 27.7）；行不对齐是 n 不是 32 的倍数时行起点不在 32 字节 sector 边界上。

② "padding 让 L1 缓存不用重复写入"

修正

padding 和 L1 无关。证据是读请求平均从 4 个 sector 变成约 4.44 个；padding 用 stride 让每行从对齐地址开始，恢复到 4 个。计算时仍除以逻辑长度 n。

③ "AOTAutograd 转化为 CUDA 中的算子"

修正

转成的是 aten 层的 FX 图，仍是 PyTorch 算子，还没有 CUDA kernel；之后 Inductor 做 lowering → scheduler（融合）→ codegen。

③ "用 nsys 抓发现 kernel 少一个，根据名字推测残差和 norm 融合"

修正

compile 这条线（Quest）的 kernel 数来自 torch.profiler；融合是读 inductor_cache 里的调用顺序确认的（

triton_red_fused__unsafe_view_add_mean_mul_pow_rsqrt_1

），不只靠名字猜。分档是这样查出来的：fused 后 kernel 数持平但 GPU 还多 26 μs → 读 .json 发现 Triton 每行 512 线程 → sweep_block.py 实测 1 行时 128×16 最慢 → 改成 chooseBlock(n, rows)。

③ "Torch.library 注册形状 meta，提前注册为 .so 所以可以直接用"

修正

几件事混在一起：.so 是编译产物，TORCH_LIBRARY 是注册方式，注册内容是 schema + CUDA 实现 + Meta 实现（Meta 是"形状怎么推"的函数，不是形状本身）。

④ "实现 plugin 接口，还是要注册 schema 以及 meta 的形状"

修正

TensorRT 没有 schema 和 Meta。对应关系是：TORCH_LIBRARY + schema ↔ ONNX 节点 + creator 按名字注册；Meta ↔

getOutputDimensions

/

getOutputDataType

；CUDA 实现 ↔

enqueue

；dispatcher ↔ 注册表按 RmsNorm 找 creator。

④ "靠 ONNX 中注册的算子名称对接；b8 变快是 TensorRT 内部换了乘法"

基本对

还要加属性名 eps；b8 变快是 plugin 把 Myelin 切成 17 段，MLP 三个矩阵乘从各约 2.4 ms 变成合计约 0.87 ms，norm 只有几 μs，不是 norm 的功劳。

### 第 3 点的面试版

最早用 PYBIND11，它只是把 C++ 函数交给 Python，PyTorch 不认识这个算子，compile 只能切断图。后来用 `TORCH_LIBRARY` 注册 schema、CUDA 实现和 Meta 实现。compile 时 Dynamo 用 Meta 推形状抓图，AOTAutograd 转成 aten FX 图，Inductor 融合并生成 Triton。但算子是黑盒，挡住了残差和 norm2 的融合，kernel 从 14 变 15，decode b1 变慢。我写了 fused add + RMSNorm，kernel 数追平，但 GPU 时间还多 26 μs。读 Inductor 生成的代码发现它每行用 512 线程，单 kernel 扫描证实 1 行时我们的 128×16 不合适，于是按行数分档，再加 float4，最后 CUDA Graph 下 decode b1 只比 Inductor 慢 0.6%，b8 打平。

> 第 5 点还没讲
>
> 。参考答案：① TensorRT b8 比 PyTorch 慢，推测 Myelin 给矩阵乘选的 kernel 不合适，开 TF32 / FP16 比较 kernel 和时间来验证；② 不用 CUDA Graph 时 compile+fused 多约 34 μs，推测是 CPU 调用路径，分段计时并比较 CUDA Graph 前后；③ ours engine 里 norm1 比 norm2 慢，原因还不知道，看逐层 profile 和 nsys。讲的时候每个问题说清"是什么、推测是什么、怎么验证"。
