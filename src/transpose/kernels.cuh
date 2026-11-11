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
__global__ void copy_kernel(const float* __restrict__ A, float* __restrict__ B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * 32+ threadIdx.y;
    #pragma unroll
    for(int i = 0;i<4;++i){
        if (y + i*8 < M && x < N) B[(y+i*8) * N + x] = A[(y+i*8) * N + x];
    }
}

inline bool launch_copy(const float* A, float* B, int M, int N, cudaStream_t s) {
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    copy_kernel<<<grid, block, 0, s>>>(A, B, M, N);
    return true;
}

// ───────────── 第 1 版：朴素版 ─────────────
// 一个线程搬一个元素：x 方向对应 A 的列 j，y 方向对应 A 的行 i。
// 读 A[i][j]（同一个 warp 里 j 连续 → 读是连续的），写 B[j][i]（每个线程相隔 M 个 float → 写是跳着的）。
__global__ void transpose_naive(const float* A, float* B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;       // 每个 block 负责 32 行，线程每次跨 8 行
    #pragma unroll
    for(int i = 0;i<4;++i){
        if (y + i*8 < M && x < N) B[x * M + y+i*8] = A[(y+i*8) * N + x];
    }
}

inline bool launch_naive(const float* A, float* B, int M, int N, cudaStream_t s) {
    // TODO：block 32×32，grid 向上取整盖住 N 列、M 行，启动 transpose_naive<<<grid, block, 0, s>>>
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    transpose_naive<<<grid, block, 0, s>>>(A, B, M, N);
    return true;
}

// ───────────── 第 2 版：shared memory 中转（故意保留 bank conflict）─────────────
// 一个 block（32×32 个线程）负责 A 的一个 32×32 小块。
// 第 1 步：按 A 的行读进 tile[ty][tx]（读全局连续）；__syncthreads()；
// 第 2 步：按 B 的行写出去，B 的这一格取 tile[tx][ty]（写全局连续）。
// 注意：A 的第 (by, bx) 块，转置后落在 B 的第 (bx, by) 块。
__global__ void transpose_shared(const float* A, float* B, int M, int N) {
    __shared__ float tile[32][32];
    int tx = threadIdx.x, ty = threadIdx.y;      // tx 0~31，ty 0~7（block 32×8，每个线程搬 4 行，每次跨 8 行）
    int bx = blockIdx.x * 32;                    // 这个小块在 A 里的起始列
    int by = blockIdx.y * 32;                    // 这个小块在 A 里的起始行

    // 第 1 步：读 A 的 第 (by+ty) 行、第 (bx+tx) 列
    int ar = by + ty, ac = bx + tx;
    #pragma unroll
    for(int i = 0;i<4;++i){
        if ((ar +i*8) < M && ac < N) tile[ty+i*8][tx] = A[(ar +i*8)* N + ac];
    }

    __syncthreads();

    // 第 2 步：写 B 的 第 (bx+ty) 行、第 (by+tx) 列   ← bx、by 对调了
    int br = bx + ty, bc = by + tx;
    #pragma unroll
    for(int i = 0;i<4;++i){
        if (br + i*8 < N && bc < M) B[(br +i*8)* M + bc] = tile[tx][ty+i*8];
    }
}

inline bool launch_shared(const float* A, float* B, int M, int N, cudaStream_t s) {
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    transpose_shared<<<grid, block, 0, s>>>(A, B, M, N);
    return true;
}

// ───────────── 第 3 版：第 2 版 + padding，消除 bank conflict ─────────────
// 只把 tile 改成 [32][33]，其余和第 2 版一样。
__global__ void transpose_shared_pad(const float* A, float* B, int M, int N) {
    __shared__ float tile[32][33];               // 只有这里多 1 列，其余和第 2 版一样
    int tx = threadIdx.x, ty = threadIdx.y;      // tx 0~31，ty 0~7（block 32×8，每个线程搬 4 行，每次跨 8 行）
    int bx = blockIdx.x * 32;                    // 这个小块在 A 里的起始列
    int by = blockIdx.y * 32;                    // 这个小块在 A 里的起始行

    // 第 1 步：读 A 的 第 (by+ty) 行、第 (bx+tx) 列
    int ar = by + ty, ac = bx + tx;
    #pragma unroll
    for(int i = 0;i<4;++i){
        if (ar + i*8 < M && ac < N) tile[ty+i*8][tx] = A[(ar +i*8)* N + ac];
    }

    __syncthreads();

    // 第 2 步：写 B 的 第 (bx+ty) 行、第 (by+tx) 列   ← bx、by 对调了
    int br = bx + ty, bc = by + tx;
    #pragma unroll
    for(int i = 0;i<4;++i){
        if (br + i*8 < N && bc < M) B[(br +i*8)* M + bc] = tile[tx][ty+i*8];
    }
    
}

inline bool launch_shared_pad(const float* A, float* B, int M, int N, cudaStream_t s) {
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    transpose_shared_pad<<<grid, block, 0, s>>>(A, B, M, N);
    return true;
}
