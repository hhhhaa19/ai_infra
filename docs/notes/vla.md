<a id="vla"></a>

*论文*

# 端侧 VLA 与推理引擎：EagleVLA、LiteVLA-Edge 和量化 ACT

**这部分讲什么**：机器人策略模型（尤其是 VLA，视觉-语言-动作模型）放到 Jetson 上太慢，这批论文怎么解决？它们用了哪些推理引擎，和我们的 ONNX → TensorRT 路线有什么不同？这些内容来自论文和 README，**没有复现，也没有逐条核对原文**；数字以论文最新版为准。

## 1. 先看一个简单的例子：8 GB Jetson 上的量化 ACT

这篇 2026 年的预印本（arXiv 2608.03938）用 100 条遥操作示范在 RTX 3070 上训练 ACT，再部署到 Orin Nano Super 8 GB，完全没有用云端。它正好对应 FP16 → INT8 这条路线：

| 精度 | 平均延迟 | 速度 | 任务成功 |
|---|---|---|---|
| FP32，PyTorch | 114.02 ms | 8.77 Hz | 19/20 |
| FP16，TensorRT | 17.93 ms | 55.77 Hz | 18/20 |
| INT8，TensorRT | 12.65 ms | 79.0 Hz | 19/20 |

局限：只有一个任务、一种机器人，每种精度 20 次，还没有同行评审。同一研究里 Diffusion Policy 的 10 次实物测试成功 0 次，也只代表它自己的配置。有转述称这里的 INT8 只量化了 ResNet18，145 个 Transformer 层都没量化，**没在原文核对**；如果属实，学 INT8 时就要关注"哪些层真的被量化了"。

## 2. EagleVLA（原名 Jetson-PI，CoRL 2026 录用）：问题是什么

π0.5 这类 VLA 在 Jetson 上太慢，机器人来不及反应。它分三段：图像 → ViT → 语言模型（LLM）→ 动作专家（10 步去噪）→ 一段动作。Orin 50W 模式下三段约 152.3 / 631.0 / 536.8 ms，一次 1420.8 ms，约 0.7 Hz。同步推理时机器人要等下一段动作算完，动作之间会停顿。

改成异步推理（机器人一边执行当前动作，模型一边算下一段）就不用停了，但模型看到的是旧画面：t 时刻拍图开始推理，t+Δ 时刻算完时环境和机器人都已经变了，Δ 越大，动作越和环境不匹配。这就是论文说的感知和执行不同步。

## 3. EagleVLA：算法上怎么解决

新增一个约 4000 万参数（约占模型 1%）的修正模块：输入 t 时刻的 VLM 特征加这段时间已经确定要执行的动作，输出预测的 t+Δ 时刻特征和一个置信度。动作专家根据预测的未来状态生成动作。打个比方：看到球向右飞，同时知道自己在往前走，就提前判断 0.5 秒后球和手的位置。

置信度用来调度：置信度高就跳过 VLM，直接用预测状态运行动作专家；低就重新看环境、运行 VLM。论文还指出 Orin 上同时运行 VLM 和动作专家效果不好，因为两者都受带宽限制，会互相抢内存带宽，这和 RMSNorm 里学到的带宽瓶颈是同一类问题。

训练在 A100 上，用已有的 π0.5 权重继续训（不是从零）：阶段 1 训练压缩模块（Q-Former 把 VLM 输出压成 4 个 token）和动作专家，用真实的未来状态教动作专家；阶段 2 训练修正模块预测未来状态，Δ 随机取值，让同一个模型适应不同延迟。第二阶段训练步数论文表格写 25000、README 写 15000，有出入。

## 4. EagleVLA：推理引擎怎么做快（最该看的部分）

实现分两个仓库：EagleVLA 负责算法和训练（基于 OpenPI），EagleVLA-Edge 负责端侧推理（基于 llama.cpp）。模型转成 GGUF（语言模型和动作专家一个文件，视觉编码器一个文件），ViT、LLM、动作专家都放进 llama.cpp 的 GGML 计算图，常驻成一个服务，控制程序每次通过 HTTP 上传图像、状态和指令，拿回动作。推理时用三个缓冲区：`B_kv`（VLM 的 KV cache，给动作专家用）、`B_h`（VLM 压缩状态，给修正模块用）、`B_a`（已确定要执行的动作）。

| 优化 | 开关或文档 | 怎么做 | 和学过的对应 |
|---|---|---|---|
| CUDA 图复用 | PI0_PERSISTENT_CUDA_GRAPH=1 | 语言指令补到固定长度，第一次录图后反复用 | CUDA Graph 要求形状和地址固定 |
| GPU 常驻缓冲区 | pi0_decode_kv_gpu.md | embedding 和跨模型 KV 中间结果留在 GPU | 减少不必要的搬运 |
| Flow unroll | PI0_DECODE_UNROLL=10 | 10 步去噪合进一张图 | 减少 CPU 发射和调用开销 |

| 配置（v7，Jetson Orin） | ViT | LLM | 动作专家 | 总时间 | 反应时间 | 频率 |
|---|---|---|---|---|---|---|
| 原始 π0.5 | 152.3 | 631.0 | 536.8 | 1420.8 ms | 1420.8 ms | 0.70 Hz |
| 加调度 | 152.3 | 631.0 | 536.8 | 1420.8 ms | 674.9 ms | 1.48 Hz |
| 加图复用 | 71.9 | 196.9 | 143.9 | 412.9 ms | 184.0 ms | 5.43 Hz |
| 加缓冲和 unroll | 66.6 | 137.0 | 67.6 | 271.1 ms | 97.1 ms | 10.30 Hz |

v7 摘要写 14.71 倍；v1 写 8.66 倍、最后一行是 412.9 ms / 6.06 Hz，版本之间数字不同，以最新版为准。这张表的写法值得学：先拆阶段，再逐项加优化，能看出每一步的贡献；图复用收益最大，调度主要改善反应时间。这里没有用量化，作者把优化限定在系统层，说明可以和量化、剪枝叠加。

任务结果：LIBERO 仿真里异步条件下基本保住了同步推理的成功率（比如 Spatial 97.4 对同步 97.3，论文称平均比 VLASH 高 14.8%、比 RTC 高 3.9%），这些仿真实验不是在 Orin 上跑的完整流程；真机叠衣服每个子任务 10 次，Orin 上普通异步抓取 / 折叠 / 放置 6 / 0 / 5，EagleVLA 10 / 8 / 9。局限：真机样本少；模型继续变大或 batch 增大时端侧仍追不上高端 GPU（作者自己说的）。没确认的：哪个线程负责推理、哪个负责控制，Q-Former 内部结构，源码没有逐行读。

## 5. LiteVLA-Edge：只证明了"能跑起来"

流程是：相机图像 → SmolVLM-256M（LoRA 微调）→ GGUF Q4_K_M 量化 → llama.cpp CUDA 后端（42 层放 GPU）→ ROS 2 节点异步发布速度指令，平均端到端 150.5 ms，约 6.6 Hz。它没用 TensorRT，走的是 LLM 常见的 GGUF 路线。

证据很弱：没有任务成功率；量化对动作准确性的影响只有论述没有实测；硬件描述有矛盾（有的章节写 AGX Orin 64GB，另一处把 150.5 ms 写成 Orin NX 的结果）；测量次数正文说 30 次、表格写 300 次；延迟没拆分；和其他模型的比较来自不同硬件和数据；"提升 220%"看不出计算依据。反过来看，这正说明自己的实验要做到：写清硬件和功耗模式，延迟拆成采集 / 预处理 / 推理 / 后处理，同时报告成功率和置信区间，量化前后在同一任务上比较。

## 6. GGUF、ONNX Runtime、TensorRT、Inductor 分别在哪一层

GGUF 是**模型文件格式**，TensorRT 是**优化器和运行时**，不是同一层。可比的是两套组合：GGUF + llama.cpp 和 ONNX + TensorRT。最大区别是图写在哪：ONNX 文件里记录完整计算图，TensorRT 读图后自动融合、实测挑 kernel，遇到不支持的算子写 plugin；GGUF 主要存权重和元数据（层数、隐藏维度、架构名），计算图写在 llama.cpp 的 C++ 代码里，所以 π0.5 这种非标准 LLM 要改引擎代码。

|  | TensorRT | llama.cpp |
|---|---|---|
| 融合 | builder 自动融合 | 主要靠手写和后端优化 |
| kernel 选择 | 建 engine 时实测候选实现 | 由引擎和后端代码决定 |
| 量化 | FP16、INT8 等 | 常用 Q4、Q8 等权重量化 |
| 自定义算子 | ONNX 节点 + plugin | 直接改引擎 |
| 硬件 | NVIDIA GPU | CPU、NVIDIA GPU 和其他后端 |
| 产物 | 绑定 GPU 的 `.engine` | 通用的 `.gguf`，运行时再加载 |

EagleVLA 为什么选 llama.cpp，论文没专门解释；推测是它更方便控制图复用时机、GPU 缓冲区和去噪循环展开，这些都要深入引擎内部。选型建议：ACT、检测、分割走 ONNX + TensorRT；LLM、VLM 先看 GGUF + llama.cpp；π0.5 这类 VLA 没有固定答案。

**ONNX Runtime（ORT）**是微软主导的开源推理引擎：读 ONNX → 做图优化（常量折叠、部分融合）→ 把节点分给不同的 Execution Provider（CPU / CUDA / TensorRT 等后端），不支持的节点退回 CPU。它适合当对照：同一个 ONNX 比较 PyTorch、ORT、TensorRT，能分清问题出在导出还是 TensorRT 的优化。注意 CUDA EP 不支持的节点可能回到 CPU 并增加拷贝，用 ORT 先查节点是否跑在 GPU 上；我们的 RmsNorm plugin 不能直接用于 ORT，要另写 ORT 自定义算子（kernel 和 `launchRMSNorm` 可复用）；Orin 上装哪个包、支持哪些 EP 还没确认。之前 ResNet 验证里用 ORT 确认过 ONNX 的数值。

|  | Inductor | ONNX Runtime | TensorRT |
|---|---|---|---|
| 输入 | aten FX 图 | ONNX | ONNX |
| 主要工作 | 融合循环，生成 Triton | 优化图，分配节点，调用后端 | 融合，挑 kernel，生成 engine |
| kernel 从哪来 | 大量现场生成 | 主要来自 EP 现有实现 | 主要从 kernel 库实测选择 |
| 运行方式 | 在 PyTorch 里运行 | 由 ORT 运行时执行 | 加载 engine 执行 |
| 自定义算子 | 外部 kernel，黑盒 | 自定义 op，通常也是黑盒 | plugin，黑盒 |

所以 ORT 更接近"运行时 + 图优化器"，Inductor 是"编译器 + 代码生成器"；按角色对应，TensorRT 比 ORT 更接近 Inductor。三条路可以这样记：PyTorch 里自动生成 kernel 用 Inductor；ONNX 通用运行和对照用 ORT；NVIDIA 端侧深度优化用 TensorRT。其他开源框架（vLLM / SGLang、ExecuTorch、TFLite、MNN / ncnn、OpenVINO、IREE、TensorRT Model Optimizer）只是按印象分类，版本和支持情况没有逐个核实。

## 7. 这个方向的会议和阅读顺序

没有查到"机器人策略端侧推理"的公认 benchmark（几次搜索的结论，不能排除漏掉）。相邻方向较公认的有：MLPerf Inference（通用推理，Edge 场景，只查到 v4.0 / v4.1 时期的 Orin 资料）、RobotPerf（ROS 2 计算图的延迟、吞吐和功耗，是否正式发表未确认）、RoWild（ACM POMACS 2023，6 个端到端机器人应用）、MAVBench（无人机端到端负载，发表会议未查证）。有报道提到 VLA-HWBench 测了 18 个 VLA 的 7 段延迟，没找到原论文，数字先不引用。

会议按论文重点选（领域判断，没逐个核对 2027 年截止日期）：机器人学习看 CoRL、RSS；机器人综合看 ICRA、IROS、RA-L；ML 系统看 MLSys、MobiSys、SenSys；实时系统看 RTSS、RTAS；体系结构看 ASPLOS、MICRO、ISCA、HPCA。只核实到 CoRL 2026 在 11 月 9～12 日（Austin），同期有 Efficient Foundation Models for Real-Time Embodied AI workshop（11 月 12 日）。MLSys 2026 的几篇 oral 可以借方法：IntAttention（去掉 softmax 的浮点绕行，相对 FP16 最高 3.7 倍，但主要在 Armv8 CPU 上）、MixLLM（混合精度量化）、CORE（端侧 LLM 的频率和能耗）、CAGE（QAT）。

建议阅读顺序：① 量化 ACT（和要做的最接近）；② EagleVLA 最新版的延迟拆分和优化方法，再读 EagleVLA-Edge 的 `pi0_decode_graph_reuse.md`，对照 CUDA Graph 理解它怎么固定图和地址；③ Embodied Foundation Models at the Edge（综述，强调端侧部署主要是系统工程问题）；④ Characterizing VLA across XPUs（RTX 4090、Orin、Thor 上的瓶颈分析）；⑤ 学 INT8 时读 IntAttention；有空看 RoWild 学系统评估方法、Kress-Gazit 等学真机评估设计。现在不急着投稿，等项目有"量化 + 系统优化 + 真机任务结果"再和导师讨论。

[EagleVLA](https://arxiv.org/html/2607.12659) · [EagleVLA-Edge](https://github.com/PKU-SEC-Lab/EagleVLA-Edge) · [LiteVLA-Edge](https://arxiv.org/html/2603.03380) · [8 GB Jetson 上的量化 ACT](https://arxiv.org/html/2608.03938) · [The Reality Gap in Robotics](https://arxiv.org/abs/2510.20808v1) · [Kress-Gazit 等](https://arxiv.org/abs/2409.09491) · [Kadian 等](https://arxiv.org/abs/1912.06321v2)

<a id="vla-iv"></a>

## 面试要求

### 必须会

- VLA 的结构：视觉编码器（ViT）+ 语言模型 + 动作头 / 动作专家；为什么比 ACT 难部署（参数量大、自回归生成、显存和带宽）。
- 能说出端侧 VLA 的主要优化手段：量化（INT8 / INT4）、减少视觉 token、缓存、算子融合、异步推理。
- 几个推理框架各在哪一层：GGUF + llama.cpp、ONNX + ONNX Runtime、ONNX + TensorRT、torch.compile / Inductor。

### 应该会

- EagleVLA 的思路（算法上减少计算 + 推理引擎上做快，Orin 上从 0.70 Hz 到 10.30 Hz，论文数字，未复现）。
- 量化 ACT 在 8 GB Jetson 上的例子说明了什么。
- 怎么读这类论文：先看问题和指标，再看哪些是算法改动（要重训）、哪些是工程改动（不用重训）。

### 加分

- 投机解码、动作分块与推理频率的关系、云端 + 端侧的分层部署。
- 这个方向的主要会议（CoRL、RSS、ICRA、IROS，以及 MLSys 等系统会议）。

高频问题并在下面的自测里。三档的含义见复习笔记顶部说明。

## 自测

- 量化 ACT 的 FP16 和 INT8 结果能证明 INT8 更好吗？"145 个 Transformer 层没量化"如果属实，说明什么？
- 异步推理解决了什么、又带来什么新问题？EagleVLA 的修正模块输入输出是什么，置信度怎么用？
- EagleVLA 三个系统优化分别对应学过的哪个概念？哪一项收益最大？为什么拆阶段的表格比只报总时间有说服力？
- LiteVLA-Edge 的证据弱在哪里？自己的实验要怎么避免？
- GGUF 和 TensorRT 为什么不是同一层？计算图分别写在哪？
- ONNX Runtime 的 EP 是什么？用它做对照时第一件事要查什么？它和 Inductor、TensorRT 的最大区别是什么？

## 面试高频问题

- VLA 部署到端侧的难点有哪些？
- llama.cpp 和 TensorRT 有什么区别？各适合什么场景？
