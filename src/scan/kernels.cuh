// 手写 scan（前缀和），inclusive：Y[i] = X[0] + X[1] + … + X[i]。X、Y 都是长度 n 的 int 数组。
// 例：X = [3, 1, 7, 0, 4, 1, 6, 3]  →  Y = [3, 4, 11, 11, 15, 16, 22, 25]
// 这个文件由你来写：每一版写好 kernel 和对应的 launch_xxx，launch 里把 return false 改成启动 kernel 并 return true。
// scan_bench.cu 负责准备数据、和 CPU 串行结果逐个比对、计时；没实现的版本（返回 false）会被跳过。
// n 不一定是 32 或 blockDim 的倍数，越界要处理。
// work 是 bench 给的一块显存（int，大小见 kWorkInts），放中间结果用，比如每个 block 的总和、look-back 的标志位。
//   每次 launch 前它的内容是上一次留下的，需要清零的话自己 cudaMemsetAsync。
// scan 计算很少，是带宽问题：最少要读一遍 X、写一遍 Y，目标是接近 copy 的速度。
#pragma once
#include <cuda_runtime.h>

// work 的大小：n 个 int 再加 4096 个，各版本都够用（多级的 block 和、标志位都放得下）
inline size_t kWorkInts(int n) { return size_t(n) + 4096; }

// ───────────── 天花板：直接复制（不是 scan，只用来对比速度）─────────────
__global__ void copy_kernel(const int* __restrict__ X, int* __restrict__ Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) Y[i] = X[i];
}

inline bool launch_copy(const int* X, int* Y, int n, int* work, cudaStream_t s) {
    int blocks = (n + 255) / 256;
    if (blocks > 16 * 64) blocks = 16 * 64;   // Orin 16 个 SM，每个 SM 若干个 block 就够了
    copy_kernel<<<blocks, 256, 0, s>>>(X, Y, n);
    return true;
}

// ───────────── 第 1 版：一个 block，shared memory 里做 Hillis-Steele ─────────────
// 只处理 n ≤ BLOCK 的情况（一个线程一个数），n 更大就 return false。目的是先把算法写对。
// 线程 t 把 X[t] 放进 shared 的 buf[t]（越界放 0）。然后 offset = 1, 2, 4, …：
//   每个位置 t ≥ offset 的线程做 buf[t] += buf[t - offset]。
// 注意：同一步里别的线程可能正在改 buf[t - offset]，读和写要分开——
//   先把 buf[t - offset] 读到寄存器，__syncthreads()，再写 buf[t]，再 __syncthreads()。
//   （另一种办法是两块 shared 轮流用，叫 double buffer。）
// __syncthreads() 不能放在 if 里面。
// 最后线程 t 把 buf[t] 写回 Y[t]。
template <int BLOCK>
__global__ void scan_block(const int* __restrict__ X, int* __restrict__ Y, int n) {
    // TODO
    int i = threadIdx.x;
    int r = blockIdx.x * blockDim.x+i;
    __shared__ int buf [BLOCK] ;
    buf[i] = r < n ? X[r]:0;
    __syncthreads();
    for(int offset = 1;offset< n;offset *=2){
      int cur = (i >= offset) ? buf[i - offset] : 0;   // 所有线程都读（不该加的读 0）
        __syncthreads();
        buf[i] += cur;
        __syncthreads();
    }
    if (r < n) Y[r] = buf[i];
}

inline bool launch_block(const int* X, int* Y, int n, int* work, cudaStream_t s) {
    if (n > 1024) return false;                         // 一个 block 只能处理 1024 个数
    scan_block<1024><<<1, 1024, 0, s>>>(X, Y, n);
    return true;
}

// ───────────── 第 2 版：多个 block，三步走 ─────────────
// 每个 block 负责连续的一段（BLOCK 个数），分三个 kernel：
//   ① scan_local：每个 block 在自己那段里做 scan（复用第 1 版的写法），结果写进 Y；
//                 同时把这段的总和（就是这段 scan 的最后一个数）写进 sums[blockIdx.x]。
//   ② 对 sums 做 scan：sums 有 ceil(n / BLOCK) 个数。如果 ≤ BLOCK，一个 block 就够；
//                 如果更多（n = 2^26 时有 65536 个），就对 sums 再调用一次 launch_multi —— 递归。
//   ③ add_offset：第 b 个 block（b ≥ 1）给自己那段的每个数加上 sums[b - 1]（前面所有段的总和）。

// sums 放在 work 里；递归时下一层的 sums 接在这一层后面。
// 这一版读写显存的遍数：① 读 X 写 Y，③ 读 Y 写 Y，比 copy 多一读一写。
// ① 每个 block 在自己那段里做 scan，再把段的总和写进 sums[blockIdx.x]
//   基本就是你的 scan_block，只改两处：
//   (1) 用全局位置 g = blockIdx.x * BLOCK + threadIdx.x 去读 X、写 Y
//   (2) 最后由一个线程写总和：if (threadIdx.x == BLOCK - 1) sums[blockIdx.x] = buf[BLOCK - 1];
//       越界的位置补的是 0，所以 buf[BLOCK-1] 就是这段的总和，最后一个 block 不满时也成立
template <int BLOCK>
__global__ void scan_local(const int* X, int* Y, int n, int* __restrict__ sums) {   // X、Y 不加 __restrict__：递归时 X == Y
    int i = threadIdx.x;
    int r = blockDim.x* blockIdx.x+ i;
    __shared__ int buf [BLOCK] ;
    buf[i] = r < n ? X[r]:0;
    __syncthreads();
    for(int offset = 1;offset< BLOCK;offset *=2){
      int cur = (i >= offset) ? buf[i - offset] : 0;   // 所有线程都读（不该加的读 0）
        __syncthreads();
        buf[i] += cur;
        __syncthreads();
    }
    if (r < n) Y[r] = buf[i];
    if (threadIdx.x == BLOCK -1) sums[blockIdx.x] = buf[BLOCK -1];

}

// ③ 第 b 段（b ≥ 1）的每个数加上 sums[b - 1]
template <int BLOCK>
__global__ void add_offset(int* Y, int n, const int* __restrict__ sums) {
    int b = blockIdx.x;
    int g = b * BLOCK + threadIdx.x;
    if (b == 0 || g >= n) return;          // 第 0 段前面没有东西；越界的线程不写
    Y[g] += sums[b - 1];
}

inline bool launch_multi(const int* X, int* Y, int n, int* work, cudaStream_t s) {
    constexpr int BLOCK = 1024;
    int nb = (n + BLOCK - 1) / BLOCK;      // block 数，也就是 sums 的个数
    int* sums = work;  
       scan_local<BLOCK><<<nb, BLOCK, 0, s>>>(X, Y, n, sums);
    if (nb == 1) return true;
    launch_multi(sums, sums, nb, work + nb, s);
    add_offset<BLOCK><<<nb, BLOCK, 0, s>>>(Y, n, sums);
    return true;
}

// ───────────── 第 3 版：block 内的 scan 用 warp shuffle ─────────────
// 和第 2 版一样三步走，只换掉 block 内部的做法：
//   (a) warp 内用 __shfl_up_sync 做 Hillis-Steele：offset = 1, 2, 4, 8, 16，
//       x += (lane >= offset) ? 从 lane - offset 拿到的值 : 0。不需要 shared、不需要 sync。
//   (b) 每个 warp 的最后一个 lane 把 warp 的总和写进 shared 的 warp_sums[warp]，__syncthreads()。
//   (c) 第 0 个 warp 对 warp_sums 再做一次 warp scan（一个 block 最多 32 个 warp，正好一个 warp 能算）。
//   (d) __syncthreads() 后，每个线程加上前面所有 warp 的和 warp_sums[warp - 1]。
// 可选：每个线程处理多个数（先在寄存器里串行加，再参与 warp scan），一个 block 负责的段更长，sums 更少。
inline bool launch_warp(const int* X, int* Y, int n, int* work, cudaStream_t s) {
    // TODO
    return false;
}

// ───────────── 第 4 版（可选）：decoupled look-back，一个 kernel 完成 ─────────────
// 每个 block 算完自己那段的总和后，"公布"出去，然后往前看前面的 block：
//   前面的 block 要么只公布了自己那段的和（A，aggregate），要么已经知道包含前面所有段的前缀（P，prefix）。
//   往前一直累加 A，直到碰到一个 P 就停 —— 不用等前面全部算完。
// 标志位和值打包进一个 64 位数原子地写，避免读到"标志更新了值还没更新"。
// 这是 CUB 用的方法，只读 X 一遍、写 Y 一遍。
inline bool launch_lookback(const int* X, int* Y, int n, int* work, cudaStream_t s) {
    // TODO
    return false;
}
