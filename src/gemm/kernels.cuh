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
    int j = blockDim.x* blockIdx.x + threadIdx.x;
    int i = blockDim.y* blockIdx.y + threadIdx.y;
    if (i>=M ||j>= N) return ;
    float sum = 0.0f;
    for (int s = 0 ;s < K;s++){
        sum += A[i*K+s]* B[s*N+j];
    }
    C [i*N+j] = sum;
}

inline bool launch_naive(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
    // TODO：定 block（比如 16×16）和 grid（向上取整盖住 N 列、M 行），启动 gemm_naive<<<grid, block, 0, s>>>
    dim3 block (16,16);
    dim3 grid ((N+15)/16,(M+15)/16);
    gemm_naive<<<grid, block, 0, s>>>(A, B, C, M, N, K);
    return true;
}

// ───────────── 第 2 版：shared memory 分块 ─────────────
// 一个 block（TILE×TILE 个线程）算 C 的一个 TILE×TILE 块；沿 K 每次搬一段 A、B 到 shared memory。
// 每段：一起搬 → __syncthreads() → 各自做 TILE 次乘加 → __syncthreads()。
template <int TILE>
__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO：__shared__ float As[TILE][TILE], Bs[TILE][TILE];
    __shared__  float As[TILE][TILE];
    __shared__  float Bs[TILE][TILE];
    int bi = blockIdx.y;
    int bj = blockIdx.x;
    int i = threadIdx.y;
    int j = threadIdx.x;
    float sum = 0.0f;
    int row = bi*TILE+i;
    int col = bj*TILE+j;
    for (int k = 0; k< (K+TILE-1) / TILE;++k){
        As [i][j] = k*TILE+j < K && row <M? A[row*K+k*TILE+j]:0;
        Bs [i][j] = k*TILE+i < K && col < N ? B[(k*TILE+i)*N+col]:0;
        __syncthreads();
        for (int s= 0;s<TILE;++s){
            sum += As[i][s] * Bs[s][j]; 
        }
        __syncthreads();
    }
    if (row < M && col <N)C[row*N+col] = sum;
}

inline bool launch_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
    // TODO：constexpr int TILE = 16;（之后试 32）启动 gemm_tiled<TILE>
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled<TILE><<<grid, block, 0, s>>>(A, B, C, M, N, K);
    return true;
}

// ───────────── 第 3 版：寄存器分块 ─────────────
// 一个 block（256 个线程）算 C 的 BM×BN 块；一个线程算 TM×TN 个元素，存在寄存器数组里。
// 每段 BK：256 个线程合作把 As[BM][BK]、Bs[BK][BN] 搬进 shared memory；
// 然后对每个 kk：从 As 读 TM 个到 a[]，从 Bs 读 TN 个到 b[]，做 TM×TN 外积累加到 c[][]。
// BM、BN、BK、TM、TN 都用编译期常量，循环才能展开、c[][] 才能放进寄存器。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_reg(const float* A, const float* B, float* C, int M, int N, int K) {
    // 双缓冲：As/Bs 各两份。算 As[cur] 的同时，下一块先读进寄存器 ra/rb，算完再写进另一份。
    constexpr int NT = (BM / TM) * (BN / TN);   // block 线程数（编译期常量，循环才能展开）
    constexpr int LA = BM * BK / NT;            // 每个线程搬几个 A
    constexpr int LB = BK * BN / NT;            // 每个线程搬几个 B
    int bj = blockIdx.x;
    int bi = blockIdx.y;
    int i = threadIdx.y;
    int j = threadIdx.x;
    //在block中位置
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 在输出中的位置
    int row = BM * bi;
    int col = bj * BN;
    float c[TM][TN] = {0};
    float a[TM];
    float b[TN];
    float ra[LA];
    float rb[LB];
    __shared__ float As[2][BK][BM + 4];
    __shared__ float Bs[2][BK][BN];
    int cur = 0;

    // 准备阶段：第 0 块直接放进 buf 0
    #pragma unroll
    for (int r = 0; r < LA; ++r) {
        int idx = tid + r * NT;
        int local_i = idx / BK;
        int local_j = idx % BK;
        int ar = row + local_i;
        int ak = local_j;
        As[cur][local_j][local_i] = (ar < M && ak < K) ? A[ar * K + ak] : 0.0f;
    }
    #pragma unroll
    for (int r = 0; r < LB; ++r) {
        int idx = tid + r * NT;
        int local_i = idx / BN;
        int local_j = idx % BN;
        int bk = local_i;
        int bc = col + local_j;
        Bs[cur][local_i][local_j] = (bk < K && bc < N) ? B[bk * N + bc] : 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < K; t += BK) {
        bool has_next = t + BK < K;
        // ① 发出下一块的读请求，只进寄存器，不等结果
        if (has_next) {
            #pragma unroll
            for (int r = 0; r < LA; ++r) {
                int idx = tid + r * NT;
                int local_i = idx / BK;
                int local_j = idx % BK;
                int ar = row + local_i;
                int ak = t + BK + local_j;
                ra[r] = (ar < M && ak < K) ? A[ar * K + ak] : 0.0f;
            }
            #pragma unroll
            for (int r = 0; r < LB; ++r) {
                int idx = tid + r * NT;
                int local_i = idx / BN;
                int local_j = idx % BN;
                int bk = t + BK + local_i;
                int bc = col + local_j;
                rb[r] = (bk < K && bc < N) ? B[bk * N + bc] : 0.0f;
            }
        }
        // ② 用当前 buffer 计算，盖住 ① 的延迟
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int q = 0; q < TM / 4; ++q) {
                float4 v = *reinterpret_cast<const float4*>(&As[cur][kk][q * (BM / TM) * 4 + i * 4]);
                a[q * 4 + 0] = v.x;  a[q * 4 + 1] = v.y;  a[q * 4 + 2] = v.z;  a[q * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int tj = 0; tj < TN / 4; ++tj) {
                float4 v = *reinterpret_cast<const float4*>(&Bs[cur][kk][tj * (BN / TN) * 4 + j * 4]);
                b[tj * 4 + 0] = v.x;  b[tj * 4 + 1] = v.y;  b[tj * 4 + 2] = v.z;  b[tj * 4 + 3] = v.w;
            }
            #pragma unroll
            for (int ti = 0; ti < TM; ++ti) {
                #pragma unroll
                for (int tj = 0; tj < TN; ++tj) {
                    c[ti][tj] += a[ti] * b[tj];
                }
            }
        }
        // ③ 换到另一份 buffer，把寄存器里的下一块写进去（越界的位置读的时候已经是 0）
        cur ^= 1;
        if (has_next) {
            #pragma unroll
            for (int r = 0; r < LA; ++r) {
                int idx = tid + r * NT;
                As[cur][idx % BK][idx / BK] = ra[r];
            }
            #pragma unroll
            for (int r = 0; r < LB; ++r) {
                int idx = tid + r * NT;
                Bs[cur][idx / BN][idx % BN] = rb[r];
            }
        }
        // ④ 每轮只同步一次：下一块写完才能算；写的那份上一轮已经算完并同步过
        __syncthreads();
    }
    for (int ti = 0; ti < TM; ++ti) {
        for (int tj = 0; tj < TN; ++tj) {
            int r = row + (ti / 4) * (BM / TM) * 4 + i * 4 + ti % 4;
            int c_col = col + (tj / 4) * (BN / TN) * 4 + j * 4 + tj % 4;
            if (r < M && c_col < N) {
                C[r * N + c_col] = c[ti][tj];
            }
        }
    }
}

inline bool launch_reg(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
    // TODO：比如 BM=BN=128, BK=8, TM=TN=8 → block 256 个线程（一维）
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block(BN/TN, BM/TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_reg<BM, BN, BK, TM, TN><<<grid, block, 0, s>>>(A, B, C, M, N, K);
    return true;
}
