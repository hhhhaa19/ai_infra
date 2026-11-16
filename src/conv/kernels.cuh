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

// ───────────── 第 2 版：implicit GEMM（之后再写）─────────────
// 把卷积看成矩阵乘 Y[K × NPQ] = Wt[K × CRS] · B[CRS × NPQ]，
// B 不真的存下来，搬 tile 进 shared memory 的时候现算地址。
inline bool launch_igemm(const float* X, const float* Wt, float* Y, const ConvShape& sh, cudaStream_t s) {
    return false;
}
