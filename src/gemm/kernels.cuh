// 手写 FP32 GEMM：C[M×N] = A[M×K] · B[K×N]，全部行优先存储。
//   A[i][k] 在 A[i * K + k]，B[k][j] 在 B[k * N + j]，C[i][j] 在 C[i * N + j]。
// 这个文件由你来写：每一版写好 kernel 和对应的 launch_xxx，launch 里把 return false 改成启动 kernel 并 return true。
// gemm_bench.cu 负责准备数据、和 cuBLAS 对答案、计时；没实现的版本（返回 false）会被跳过。
// M、N、K 不一定是块大小的倍数，越界要处理（读越界补 0，写越界跳过）。
#pragma once
#include <cuda_runtime.h>

// ───────────── 第 1 版：朴素版 ─────────────
// 一个线程算一个 C[i][j]：i 用 y 方向，j 用 x 方向；循环 K 次累加 A[i][k] * B[k][j]。
__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
}

inline bool launch_naive(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
    // TODO：定 block（比如 16×16）和 grid（向上取整盖住 N 列、M 行），启动 gemm_naive<<<grid, block, 0, s>>>
    return false;
}

// ───────────── 第 2 版：shared memory 分块 ─────────────
// 一个 block（TILE×TILE 个线程）算 C 的一个 TILE×TILE 块；沿 K 每次搬一段 A、B 到 shared memory。
// 每段：一起搬 → __syncthreads() → 各自做 TILE 次乘加 → __syncthreads()。
template <int TILE>
__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO：__shared__ float As[TILE][TILE], Bs[TILE][TILE];
}

inline bool launch_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
    // TODO：constexpr int TILE = 16;（之后试 32）启动 gemm_tiled<TILE>
    return false;
}

// ───────────── 第 3 版：寄存器分块 ─────────────
// 一个 block（256 个线程）算 C 的 BM×BN 块；一个线程算 TM×TN 个元素，存在寄存器数组里。
// 每段 BK：256 个线程合作把 As[BM][BK]、Bs[BK][BN] 搬进 shared memory；
// 然后对每个 kk：从 As 读 TM 个到 a[]，从 Bs 读 TN 个到 b[]，做 TM×TN 外积累加到 c[][]。
// BM、BN、BK、TM、TN 都用编译期常量，循环才能展开、c[][] 才能放进寄存器。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_reg(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
}

inline bool launch_reg(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
    // TODO：比如 BM=BN=128, BK=8, TM=TN=8 → block 256 个线程（一维）
    return false;
}
