#pragma once
// CUB BlockReduce 版 RMSNorm：单个 block 处理一行；FP32 输入/输出，所有行共享 weight。
// 被 rmsnorm_cub.cu（单形状示例）和 rmsnorm_cub_bench.cu（与 rmsNormN 对照）共同包含。
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cstddef>

// BLOCK_THREADS 是编译期参数，必须等于启动时的每 block 线程数。
// n 可以小于、等于、大于 BLOCK_THREADS，也可以不是其整数倍。
template<int BLOCK_THREADS>
__global__ void rmsNormCub(
    const float* x, const float* weight, float* y, int n, float eps,
    int stride = 0   // 行间距（元素个数）；0 表示紧密存放，等于 n
) {
    const int tid = threadIdx.x;
    const std::size_t rowOffset = std::size_t(blockIdx.x) * (stride > 0 ? stride : n);

    // 1. 与手写版相同：每个线程在自己的寄存器里累加平方和。
    float sum = 0.0f;
    for (int col = tid; col < n; col += BLOCK_THREADS) {
        float value = x[rowOffset + col];
        sum += value * value;
    }

    // 2. CUB 替换手写 shuffle + 跨 warp 的共享内存规约。
    // using 是类型别名；TempStorage 是该类型需要的临时存储类型。
    using Reduce = cub::BlockReduce<float, BLOCK_THREADS>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float scale;

    // 所有线程都调用；每个线程传入自己的 sum。
    // 完整总和仅在线程 0 中有效；没有启动另一个 kernel。
    float total = Reduce(temp).Sum(sum);

    // 3. 线程 0 计算本行 scale，其他线程等待它写好。
    if (tid == 0) {
        scale = rsqrtf(total / static_cast<float>(n) + eps);
    }
    __syncthreads();

    // 4. 与手写版相同：各线程写回自己的输出位置。
    for (int col = tid; col < n; col += BLOCK_THREADS) {
        y[rowOffset + col] = x[rowOffset + col] * scale * weight[col];
    }
}
