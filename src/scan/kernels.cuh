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

// ───────────── 第 3 版：每个线程 4 个数 + warp shuffle ─────────────
// 和第 2 版一样三步走（① 局部 scan + 写段总和 → ② 递归 scan 段总和 → ③ 加偏移），只换掉 ① 的 kernel。
// 一个 block 256 个线程（8 个 warp），每个线程负责连续的 4 个数，一个 block 负责一段 1024 个数。
// 为什么每个线程拿"连续"的 4 个数，而不是 softmax 那样 lane + k*32 交错：
//   scan 要保持顺序（只用结合律，不用交换律），线程 t 的 4 个数必须整段排在线程 t+1 前面，线程内串行前缀才有意义。
//   用 int4 一次读 16 字节，一个 warp 读的是连续 512 字节，仍然是合并访问。

// warp 内的 inclusive scan：每个 lane 传入自己的 x，返回 lane 0 到自己（含）的 x 之和。
// __shfl_up_sync(mask, x, offset) 返回 lane - offset 的 x；lane < offset 时返回自己的 x（所以要判断，不能直接加）。
__device__ __forceinline__ int warp_scan_incl(int x) {
    int lane = threadIdx.x % 32;
    // TODO：offset = 1, 2, 4, 8, 16：
    //       int y = __shfl_up_sync(0xffffffff, x, offset);
    //       if (lane >= offset) x += y;
    return x;
}

template <int BLOCK, int ITEMS>
__global__ void scan_local_warp(const int* X, int* Y, int n, int* __restrict__ sums) {   // X、Y 不加 __restrict__：递归时 X == Y
    static_assert(ITEMS == 4, "这一版用 int4 读写，一个线程正好 4 个数");
    constexpr int TILE = BLOCK * ITEMS;                  // 一个 block 负责的数的个数
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int base = blockIdx.x * TILE + threadIdx.x * ITEMS;  // 本线程 4 个数中第一个的位置（一定是 4 的倍数）
    __shared__ int warp_sums[BLOCK / 32];
    int v[ITEMS];

    // 第 0 步：读 4 个数到 v[0..3]
    //   base + 3 < n：一次 int4 读：int4 t = *reinterpret_cast<const int4*>(X + base);  v[0] = t.x; … v[3] = t.w;
    //   否则（最后一个线程凑不满 4 个）：逐个读，base + k >= n 的位置补 0
    //   （X 的地址按 16 字节对齐、base 是 4 的倍数，int4 读才合法；launch_warp 已经保证了递归时也对齐）
    // TODO

    // 第 1 步：线程内串行前缀：v[1] += v[0]; v[2] += v[1]; v[3] += v[2];  本线程总和 = v[3]
    // TODO

    // 第 2 步：warp 内对"本线程总和"做 scan
    //   int incl = warp_scan_incl(v[3]);   含自己
    //   int excl = incl - v[3];            同一 warp 里排在自己前面的线程的总和
    // TODO

    // 第 3 步：warp 之间合并
    //   lane 31 把 incl（整个 warp 的总和）写进 warp_sums[warp]；__syncthreads()
    //   warp 0：读 warp_sums[lane]（lane >= BLOCK/32 的读 0），warp_scan_incl，写回 warp_sums[lane]；__syncthreads()
    //   （两个 __syncthreads() 都要放在 if 外面）
    // TODO

    // 第 4 步：本线程的偏移 = excl + (warp > 0 ? warp_sums[warp - 1] : 0)，加到 v[0..3] 上，写回 Y
    //   base + 3 < n：一次 int4 写：*reinterpret_cast<int4*>(Y + base) = make_int4(v[0], v[1], v[2], v[3]);
    //   否则逐个写，只写 base + k < n 的位置
    // TODO

    // 第 5 步：段总和写进 sums[blockIdx.x]
    //   warp_sums[BLOCK/32 - 1] 就是整个 block 的总和（越界的位置补了 0，最后一段不满时也对），让 threadIdx.x == 0 写
    // TODO
}

inline bool launch_warp(const int* X, int* Y, int n, int* work, cudaStream_t s) {
    return false;   // ← kernel 写好后删掉这一行

    constexpr int BLOCK = 256, ITEMS = 4, TILE = BLOCK * ITEMS;   // TILE = 1024
    int nb = (n + TILE - 1) / TILE;                     // 段数，也就是 sums 的个数
    int* sums = work;
    scan_local_warp<BLOCK, ITEMS><<<nb, BLOCK, 0, s>>>(X, Y, n, sums);
    if (nb == 1) return true;
    // 下一层的草稿纸起点按 4 个 int（16 字节）对齐：递归时 sums 会被当作 X、Y 用 int4 读写，必须对齐
    int* next = work + (nb + 3) / 4 * 4;
    launch_warp(sums, sums, nb, next, s);
    add_offset<TILE><<<nb, TILE, 0, s>>>(Y, n, sums);   // 复用第 2 版：段长 1024，一个 block 1024 个线程，每个线程一个数
    return true;
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
