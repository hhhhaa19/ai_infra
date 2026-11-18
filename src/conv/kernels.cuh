// 手写卷积（2D，FP32，NCHW 布局）。
//   输入  X[N, C, H, W]      N 张图，每张 C 层，每层 H×W
//   卷积核 Wt[K, C, R, S]    K 个卷积核，每个 C 层，每层 R×S
//   输出  Y[N, K, P, Q]      每张图输出 K 层，每层 P×Q
//   P = (H + 2·pad − R) / stride + 1，Q = (W + 2·pad − S) / stride + 1
//   Y[n,k,p,q] = Σ_c Σ_r Σ_s X[n, c, p·stride−pad+r, q·stride−pad+s] · Wt[k,c,r,s]
//   （h = p·stride−pad+r 或 w = q·stride−pad+s 越界的位置是补的 0，这一项跳过）
// 这个文件由你来写：每一版写好 kernel 和对应的 launch_xxx，launch 里启动 kernel 并 return true。
// conv_bench.cu 负责准备数据、和 CPU 结果比对、计时；返回 false 的版本会被跳过。
// 多维数组都按行主序摊平成一维，下标用 size_t 算，避免大形状溢出 int：
//   X[n,c,h,w]  = X[((n*C + c)*H + h)*W + w]
//   Wt[k,c,r,s] = Wt[((k*C + c)*R + r)*S + s]
//   Y[n,k,p,q]  = Y[((n*K + k)*P + p)*Q + q]
#pragma once
#include <cuda_runtime.h>
#include "../gemm/kernels.cuh"   // 第 2 版直接调用你写的 GEMM：launch_tiled、launch_reg

struct ConvShape {
    int N, C, H, W;   // 输入
    int K, R, S;      // 卷积核
    int stride, pad;
    __host__ __device__ int P() const { return (H + 2 * pad - R) / stride + 1; }
    __host__ __device__ int Q() const { return (W + 2 * pad - S) / stride + 1; }
};

// ───────────── 第 1 版：直接卷积，一个线程算一个输出数 ─────────────
// 输出一共 N·K·P·Q 个数，每个线程负责一个。线程编号 idx = blockIdx.x * blockDim.x + threadIdx.x。
// 步骤：
//   1. idx ≥ N·K·P·Q 的线程直接 return（最后一个 block 可能有多余的线程）。
//   2. 把 idx 拆回 (n, k, p, q)。Y 的摊平顺序是 n、k、p、q，q 变得最快：
//        q = idx % Q;  p = (idx / Q) % P;  k = (idx / (P*Q)) % K;  n = idx / (K*P*Q)
//   3. sum = 0，三重循环 c、r、s：
//        h = p*stride - pad + r;  w = q*stride - pad + s;
//        越界（h < 0 || h ≥ H || w < 0 || w ≥ W）就 continue；
//        sum += X[n,c,h,w] * Wt[k,c,r,s]
//   4. Y[n,k,p,q] = sum（按上面的摊平公式，其实就是 Y[idx]）。
// 想一想：同一个 warp 的 32 个线程，q 连续，它们读 X 的地址是不是连续的？读 Wt 呢？
__global__ void conv_direct(const float* __restrict__ X, const float* __restrict__ Wt,
                            float* __restrict__ Y, ConvShape sh) {
    const int C = sh.C, H = sh.H, W = sh.W, K = sh.K, R = sh.R, S = sh.S;
    const int P = sh.P(), Q = sh.Q();
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(sh.N) * K * P * Q) return;      // 多出来的线程没有格子可算
    int q = int(idx % Q);
    int p = int((idx / Q) % P);
    int k = int((idx / (size_t(P) * Q)) % K);
    int n = int(idx / (size_t(K) * P * Q));
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            int h = p * sh.stride - sh.pad + r;          // 窗口左上角 + 窗口里第 r 行
            if (h < 0 || h >= H) continue;               // 落在补的 0 上，跳过
            for (int s = 0; s < S; s++) {
                int w = q * sh.stride - sh.pad + s;
                if (w < 0 || w >= W) continue;
                sum += Wt[((size_t(k) * C + c) * R + r) * S + s] *
                       X[((size_t(n) * C + c) * H + h) * W + w];
            }
        }
    }
    Y[idx] = sum;
}

inline bool launch_direct(const float* X, const float* Wt, float* Y, const ConvShape& sh, cudaStream_t s) {
    size_t total = size_t(sh.N) * sh.K * sh.P() * sh.Q();
    int threads = 256;
    int blocks = int((total + threads - 1) / threads);
    conv_direct<<<blocks, threads, 0, s>>>(X, Wt, Y, sh);
    return true;
}

// ───────────── 第 2 版：显式 im2col + 你的 GEMM ─────────────
// 思路就是动画里那样：先把每个窗口抄成矩阵 B 的一列，再把 Wt · B 交给 src/gemm 里写好的 GEMM。
//   B 的形状 [CRS × PQ]（一张图）：
//     行号 row = (c*R + r)*S + s        —— 窗口里的第几个数，和 Wt[k,c,r,s] 摊平的顺序一样
//     列号 col = p*Q + q                —— 第几个窗口（输出位置）
//     B[row][col] = X[n, c, p*stride - pad + r, q*stride - pad + s]，越界就是 0
//     （补的 0 这次要真的写进 B：GEMM 不知道哪些是补的，它会把整列都乘一遍）
//   Y_n[K × PQ] = Wt[K × CRS] · B[CRS × PQ]
//     Wt 的内存本来就是 [K × CRS] 行主序；Y 里第 n 张图那一段（从 Y + n*K*P*Q 开始）本来就是 [K × PQ] 行主序。
//     所以 Wt 和 Y 都不用搬，只有 B 需要抄。
// N > 1 时一张一张来：每个 n 做一次 im2col、一次 GEMM。
//   （如果 N 张一起做，B 是 [CRS × NPQ]，乘出来是 [K][N][P][Q]，和 Y 的 [N][K][P][Q] 对不上，还得再搬一次。）

// 2a. im2col：一个线程抄 B 的一个数。B 一共 CRS × PQ 个数。
//   Xn 是第 n 张图的开头（launch 里传 X + n*C*H*W），所以这里不用管 n。
//   步骤：
//     1. idx ≥ CRS*PQ 的线程 return。
//     2. row = idx / PQ，col = idx % PQ。
//     3. 拆 row：s = row % S;  r = (row / S) % R;  c = row / (R*S)
//        拆 col：q = col % Q;  p = col / Q
//     4. h = p*stride - pad + r;  w = q*stride - pad + s
//     5. B[idx] = 越界 ? 0.0f : Xn[(c*H + h)*W + w]
// 想一想：同一个 warp 里 idx 连续 → col 连续 → q 连续。写 B 是连续的；读 X 呢？stride 16 时呢？
__global__ void im2col_kernel(const float* __restrict__ Xn, float* __restrict__ B, ConvShape sh) {
    const int C = sh.C, H = sh.H, W = sh.W, R = sh.R, S = sh.S;
    const int P = sh.P(), Q = sh.Q();
    const int rowline = P*Q;
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(C)*R*S*P*Q) return;      // 多出来的线程没有格子可算（先转 size_t 再乘，免得 int 溢出）
    int row = idx /rowline;
    int col = idx %rowline;
    int c = row /(R*S);
    int r = (row /S) %R;
    int s = row %S;
    int q = col % Q;
    int p = col /Q;
    int h = p*sh.stride - sh.pad + r;
    int w = q*sh.stride - sh.pad + s;
    B[idx] = (h < 0 || h >= H || w < 0 || w >= W) ? 0.0f : Xn[(c*H + h)*W + w]; 
}

// 2b. B 放在哪：申请一块显存，以后的调用接着用，不够大才换一块更大的。
//   bench 对每个形状先跑一次查对错（不计时），再预热、计时，所以 cudaMalloc 只发生在不计时的那一次。
//   这块显存到程序结束才释放，测试程序里可以接受。
inline float* im2col_workspace(size_t floats) {
    static float* buf = nullptr;
    static size_t cap = 0;
    if (floats > cap) {
        if (buf) cudaFree(buf);
        cudaMalloc(&buf, floats * sizeof(float));
        cap = floats;
    }
    return buf;
}

// 2c. launch：对每张图，im2col 一次，GEMM 一次。
//   你的 GEMM 是 C[M×N] = A[M×K] · B[K×N]，这里的 M、N、K 和卷积的 N、K 不是一个意思，别搞混：
//     GEMM 的 M = ?（Wt 有几行）  GEMM 的 N = ?（B 有几列）  GEMM 的 K = ?（Wt 的列数 = B 的行数）
//     A = ?   B = ?   C = ?（第 n 张图的输出从哪里开始）
//   GEMM 部分换成哪一版由参数 gemm 决定：launch_tiled（16×16 小块）或 launch_reg（128×128 大块）。
//   答案：M = K（Wt 有 K 行），N = PQ（B 有 PQ 列），K = CRS；A = Wt，B = B，C = Y + n*K*PQ。
using GemmLaunch = bool (*)(const float*, const float*, float*, int, int, int, cudaStream_t);
inline bool im2col_then_gemm(const float* X, const float* Wt, float* Y, const ConvShape& sh, cudaStream_t s,
                             GemmLaunch gemm) {
    int P = sh.P(), Q = sh.Q();  int CRS = sh.C*sh.R*sh.S;  int PQ = P*Q;
    float* B = im2col_workspace(size_t(CRS) * PQ);
    for (int n = 0; n < sh.N; ++n) {
        im2col_kernel<<<(CRS*PQ + 255) / 256, 256, 0, s>>>(X + size_t(n)*sh.C*sh.H*sh.W, B, sh);
        gemm(Wt, B, Y + size_t(n)*sh.K*PQ, sh.K, PQ, CRS, s);
    }
    return true;
}
inline bool launch_im2col(const float* X, const float* Wt, float* Y, const ConvShape& sh, cudaStream_t s) {
    return im2col_then_gemm(X, Wt, Y, sh, s, launch_tiled);
}
inline bool launch_im2col_reg(const float* X, const float* Wt, float* Y, const ConvShape& sh, cudaStream_t s) {
    return im2col_then_gemm(X, Wt, Y, sh, s, launch_reg);
}

// ───────────── 第 3 版：implicit GEMM ─────────────
// 要解决的问题：第 2 版先用 im2col_kernel 把 B 写进显存（resnet3x3 是 7.2 MB，X 的 9 倍），
//   GEMM 再把它读回来。B 只是中间产物，这一写一读是白花的。
// 改法：拿 gemm_tiled 抄一份，只改"把 B 搬进 shared memory"那一行：
//   以前 Bs[i][j] = B[brow][col]，现在 Bs[i][j] = X[c][h][w]，c、h、w 用 brow、col 现算
//   （公式和 im2col_kernel 里一模一样）。im2col_kernel 不用了，显存里不再有 B。
// 还是一张图一张图来：Xn = 第 n 张图的 X，Yn = 第 n 张图的 Y。
// GEMM 的三个尺寸（换个名字，免得和卷积的 K、N 搞混）：
//   GM = sh.K（Wt 的行数）   GN = P*Q（B 的列数）   GK = C*R*S（Wt 的列数 = B 的行数）
template <int TILE>
__global__ void conv_igemm_tiled(const float* __restrict__ Xn, const float* __restrict__ Wt,
                                 float* __restrict__ Yn, ConvShape sh) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int C = sh.C, H = sh.H, W = sh.W, R = sh.R, S = sh.S;
    const int P = sh.P(), Q = sh.Q();
    const int GM = sh.K, GN = P * Q, GK = C * R * S;
    int i = threadIdx.y, j = threadIdx.x;
    int row = blockIdx.y * TILE + i;      // C 的第几行 = 第几个卷积核 k
    int col = blockIdx.x * TILE + j;      // C 的第几列 = 第几个输出位置 p*Q + q
    float sum = 0.0f;
    for (int t = 0; t < (GK + TILE - 1) / TILE; ++t) {
        // 搬 A（Wt）：和 gemm_tiled 一样，Wt 本来就是 [GM × GK] 行主序
        As[i][j] = (row < GM && t * TILE + j < GK) ? Wt[size_t(row) * GK + t * TILE + j] : 0.0f;

        // 搬 B：这一行是第 3 版唯一要动脑子的地方。
        //   这个线程负责 B 的第 brow 行、第 col 列：brow = t * TILE + i
        //   1. brow ≥ GK 或 col ≥ GN → 0（矩阵本身越界）
        //   2. 拆 brow → c, r, s；拆 col → p, q（和 im2col_kernel 一样）
        //   3. h = p*stride - pad + r;  w = q*stride - pad + s
        //   4. h、w 越界 → 0（补的 0）；否则 Xn[(c*H + h)*W + w]
        //   记法：row 只和卷积核 k 有关，col 只和输出位置 pq 有关，brow 是 crs（每轮走 TILE 的那一维）。
        int brow = t * TILE + i;
        int p = col / Q;
        int q = col % Q;
        int c = brow / (R * S);
        int r = (brow / S) % R;
        int s = brow % S;
        int h = p * sh.stride - sh.pad + r;
        int w = q * sh.stride - sh.pad + s;
        // brow、col 越界时上面的 c、h、w 没意义，但条件为假就不会去读
        Bs[i][j] = (brow < GK && col < GN && h >= 0 && h < H && w >= 0 && w < W)
                   ? Xn[(c * H + h) * W + w] : 0.0f;
        __syncthreads();
        // 乘加：和 gemm_tiled 一样，一个字不用改
        for (int kk = 0; kk < TILE; ++kk) sum += As[i][kk] * Bs[kk][j];
        __syncthreads();
    }
    // 写回：Yn 本来就是 [GM × GN] 行主序，和 gemm_tiled 一样
    if (row < GM && col < GN) Yn[size_t(row) * GN + col] = sum;
}

inline bool launch_igemm(const float* X, const float* Wt, float* Y, const ConvShape& sh, cudaStream_t s) {
    // 和 launch_tiled 一样定 block、grid，再对每张图启动一次
    constexpr int TILE = 16;
    const int GM = sh.K, GN = sh.P() * sh.Q();
    dim3 block(TILE, TILE);
    dim3 grid((GN + TILE - 1) / TILE, (GM + TILE - 1) / TILE);   // x 方向盖住 GN 列，y 方向盖住 GM 行
    for (int n = 0; n < sh.N; ++n)
        conv_igemm_tiled<TILE><<<grid, block, 0, s>>>(X + size_t(n) * sh.C * sh.H * sh.W, Wt,
                                                      Y + size_t(n) * sh.K * GN, sh);
    return true;
}
