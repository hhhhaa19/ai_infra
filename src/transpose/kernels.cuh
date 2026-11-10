// 手写矩阵转置：B[N×M] = Aᵀ，A 是 M×N，全部行优先存储。
//   A[i][j] 在 A[i * N + j]，B[j][i] 在 B[j * M + i]。
// 这个文件由你来写：每一版写好 kernel 和对应的 launch_xxx，launch 里把 return false 改成启动 kernel 并 return true。
// transpose_bench.cu 负责准备数据、和 CPU 结果逐元素比对、计时；没实现的版本（返回 false）会被跳过。
// M、N 不一定是 32 的倍数，越界要处理（越界的线程不读也不写）。
// 转置没有计算，是纯带宽问题：目标是接近 copy（同样大小的直接复制）的速度。
#pragma once
#include <cuda_runtime.h>

// ───────────── 天花板：直接复制（不是转置，只用来对比速度）─────────────
// 读和写都连续，是这块数据量能达到的最快速度。
__global__ void copy_kernel(const float* A, float* B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < M && x < N) B[y * N + x] = A[y * N + x];
}

inline bool launch_copy(const float* A, float* B, int M, int N, cudaStream_t s) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    copy_kernel<<<grid, block, 0, s>>>(A, B, M, N);
    return true;
}

// ───────────── 第 1 版：朴素版 ─────────────
// 一个线程搬一个元素：x 方向对应 A 的列 j，y 方向对应 A 的行 i。
// 读 A[i][j]（同一个 warp 里 j 连续 → 读是连续的），写 B[j][i]（每个线程相隔 M 个 float → 写是跳着的）。
__global__ void transpose_naive(const float* A, float* B, int M, int N) {
    // TODO
}

inline bool launch_naive(const float* A, float* B, int M, int N, cudaStream_t s) {
    // TODO：block 32×32，grid 向上取整盖住 N 列、M 行，启动 transpose_naive<<<grid, block, 0, s>>>
    return false;
}

// ───────────── 第 2 版：shared memory 中转（故意保留 bank conflict）─────────────
// 一个 block（32×32 个线程）负责 A 的一个 32×32 小块。
// 第 1 步：按 A 的行读进 tile[ty][tx]（读全局连续）；__syncthreads()；
// 第 2 步：按 B 的行写出去，B 的这一格取 tile[tx][ty]（写全局连续）。
// 注意：A 的第 (by, bx) 块，转置后落在 B 的第 (bx, by) 块。
__global__ void transpose_shared(const float* A, float* B, int M, int N) {
    // TODO：__shared__ float tile[32][32];
}

inline bool launch_shared(const float* A, float* B, int M, int N, cudaStream_t s) {
    // TODO
    return false;
}

// ───────────── 第 3 版：第 2 版 + padding，消除 bank conflict ─────────────
// 只把 tile 改成 [32][33]，其余和第 2 版一样。
__global__ void transpose_shared_pad(const float* A, float* B, int M, int N) {
    // TODO：__shared__ float tile[32][33];
}

inline bool launch_shared_pad(const float* A, float* B, int M, int N, cudaStream_t s) {
    // TODO
    return false;
}
