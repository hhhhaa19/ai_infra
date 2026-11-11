// 手写 softmax：按行做。X、Y 都是 rows × n，行优先存储，X[r][j] 在 X[r * n + j]。
//   Y[r][j] = exp(X[r][j] - m_r) / d_r，  m_r = 这一行的最大值，d_r = Σ_j exp(X[r][j] - m_r)
// 减去最大值是为了数值稳定：float 的 exp 在输入超过约 88.7 时就溢出成 inf。测试数据里专门放了 +100 的行。
// 这个文件由你来写：每一版写好 kernel 和对应的 launch_xxx，launch 里把 return false 改成启动 kernel 并 return true。
// softmax_bench.cu 负责准备数据、和 CPU double 结果比对、计时；没实现的版本（返回 false）会被跳过。
// rows、n 不一定是 32 或 blockDim 的倍数，越界要处理。
// softmax 计算很少，是带宽问题：最少要读一遍 X、写一遍 Y，目标是接近 copy 的速度。
#pragma once
#include <cuda_runtime.h>
#include <cfloat>

// ───────────── 天花板：直接复制（不是 softmax，只用来对比速度）─────────────
// 读一遍、写一遍，和 softmax 的最小数据量相同。每个线程按 grid-stride 循环搬多个数。
__global__ void copy_kernel(const float* __restrict__ X, float* __restrict__ Y, size_t total) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; i < total; i += stride) Y[i] = X[i];
}

inline bool launch_copy(const float* X, float* Y, int rows, int n, cudaStream_t s) {
    size_t total = size_t(rows) * n;
    int blocks = int((total + 255) / 256);
    if (blocks > 16 * 64) blocks = 16 * 64;   // Orin 16 个 SM，每个 SM 若干个 block 就够了
    copy_kernel<<<blocks, 256, 0, s>>>(X, Y, total);
    return true;
}

// ───────────── 第 1 版：朴素版，一个线程算一整行 ─────────────
// 线程 r 负责第 r 行，三遍循环：① 求最大值 m；② 求 d = Σ exp(x - m)；③ 写 y = exp(x - m) / d。
// 问题在哪（写完测出来再想）：同一个 warp 的 32 个线程读的是 32 个不同的行，地址相隔 n 个 float。
__global__ void softmax_naive(const float* __restrict__ X, float* __restrict__ Y, int rows, int n) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;   // 全局行号
    if (r >= rows) return;                           // 多出来的线程直接退出（这一版没有 __syncthreads，可以 return）
    size_t base = size_t(r) * n;                     // 用 size_t，行数和列数大时 r * n 不会溢出
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = fmaxf(m, X[base + i]);
    float d = 0.0f;
    for (int i = 0; i < n; ++i) d += expf(X[base + i] - m);
    for (int i = 0; i < n; ++i) Y[base + i] = expf(X[base + i] - m) / d;
}

inline bool launch_naive(const float* X, float* Y, int rows, int n, cudaStream_t s) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    softmax_naive<<<blocks, threads, 0, s>>>(X, Y, rows, n);
    return true;
}

// ───────────── 第 2 版：一个 block 算一行，shared memory + warp shuffle 做规约 ─────────────
// block 有 BLOCK 个线程，线程 t 处理这一行的第 t、t+BLOCK、t+2·BLOCK … 个数（相邻线程读相邻地址）。
// ① 每个线程先在寄存器里求自己那几个数的最大值 → block 内规约成整行的最大值 m（所有线程都要拿到 m）
// ② 每个线程求自己那几个数的 Σ exp(x - m) → block 内规约成 d（所有线程都要拿到 d）
// ③ 每个线程写自己那几个数的 exp(x - m) / d
// 规约可以复用你在 reduction 里写的 warp shuffle：先 warp 内，再把各 warp 的结果放进 shared 合并。
// 这一版要读 X 三遍（①②③各一遍）、写 Y 一遍。
template <int BLOCK>
__global__ void softmax_block(const float* __restrict__ X, float* __restrict__ Y, int rows, int n) {
    int b = blockIdx.x;                  // 这个 block 负责第 b 行
    int r = threadIdx.x;                 // 线程在 block 里的编号
    size_t base = size_t(b) * n;         // 这一行的起点
    __shared__ float mb[BLOCK];
    __shared__ float sumb[BLOCK];

    // ① 最大值：线程 r 读第 r、r+BLOCK、… 个（相邻线程读相邻地址），分不到数的线程保持 -INFINITY
    float m = -INFINITY;
    for (int i = r; i < n; i += BLOCK) m = fmaxf(m, X[base + i]);
    mb[r] = m;
    __syncthreads();
    for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
        if (r < offset) mb[r] = fmaxf(mb[r], mb[r + offset]);
        __syncthreads();                 // 放在 if 外面：所有线程都要执行到
    }
    float row_max = mb[0];               // 所有线程都读 mb[0]，拿到整行的最大值

    // ② 分母：分不到数的线程保持 0
    float sum = 0.0f;
    for (int i = r; i < n; i += BLOCK) sum += expf(X[base + i] - row_max);
    sumb[r] = sum;
    __syncthreads();
    for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
        if (r < offset) sumb[r] += sumb[r + offset];
        __syncthreads();
    }
    float row_sum = sumb[0];

    // ③ 写结果
    for (int i = r; i < n; i += BLOCK) Y[base + i] = expf(X[base + i] - row_max) / row_sum;
}

inline bool launch_block(const float* X, float* Y, int rows, int n, cudaStream_t s) {
    softmax_block<256><<<rows, 256, 0, s>>>(X, Y, rows, n);   // 一个 block 负责一行
    return true;
}

// ───────────── 第 3 版：online softmax，最大值和分母一遍求出 ─────────────
// 和第 2 版一样一个 block 一行，但 ① ② 合成一遍：每个线程边读边维护 (m, d)：
//   读到新数 x：m_new = max(m, x)；d = d · exp(m − m_new) + exp(x − m_new)；m = m_new
// 两个线程的 (m1, d1)、(m2, d2) 合并：m = max(m1, m2)；d = d1 · exp(m1 − m) + d2 · exp(m2 − m)
// 规约时 m、d 要一起传（warp shuffle 两个值）。最后一遍写 Y。
// 这一版读 X 两遍、写 Y 一遍。FlashAttention 就是用这个合并公式把 softmax 拆成块来算的。
template <int BLOCK>
__global__ void softmax_online(const float* __restrict__ X, float* __restrict__ Y, int rows, int n) {
    // TODO
}

inline bool launch_online(const float* X, float* Y, int rows, int n, cudaStream_t s) {
    // TODO
    return false;
}

// ───────────── 第 4 版：一个 warp 算一行，整行放在寄存器里 ─────────────
// 适合 n 不大的情况（这里要求 n ≤ 32 · PER_LANE）。lane 读第 lane、lane+32、… 个数，存进寄存器数组 v[PER_LANE]。
// 最大值和分母都只用 warp shuffle（__shfl_xor_sync 让 32 个线程都拿到结果），不需要 shared memory 和 __syncthreads()。
// 只读 X 一遍、写 Y 一遍。注意 v[] 的下标要是编译期常量（循环次数用 PER_LANE），否则会掉进 local memory。
// n 超过上限时 launch 返回 false，这一版在测试里就会显示 not implemented / 跳过。
template <int PER_LANE>
__global__ void softmax_warp(const float* __restrict__ X, float* __restrict__ Y, int rows, int n) {
    // TODO
}

inline bool launch_warp(const float* X, float* Y, int rows, int n, cudaStream_t s) {
    // TODO：if (n > 32 * 32) return false; 一个 block 放 4 个 warp（128 线程）处理 4 行，grid = (rows + 3) / 4
    return false;
}
