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
    // TODO
    int bj = blockIdx.x;
    int bi = blockIdx.y;
    int i = threadIdx.y;
    int j = threadIdx.x;
    //在block中位置
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    // 在输出中的位置
    int row = BM * bi;
    int col = bj* BN;
    float c[TM][TN] ={0};
    float a[TM];
    float b[TN];
    __shared__ float As [BM][BK];
    __shared__ float Bs [BK][BN];
    for (int t = 0; t < K; t += BK) {
        for(int idx = tid ;idx < BM*BK ;idx +=num_threads){
            int local_i = idx/BK;
            int local_j = idx%BK;
            int ar = row + local_i;
            int ak = t + local_j;
            As[local_i][local_j] =
                (ar < M && ak < K) ? A[ar * K + ak] : 0.0f;
        }
        for(int idx = tid ;idx < BK*BN ;idx +=num_threads){
            int local_i = idx / BN;
            int local_j = idx % BN;
            int bk = t + local_i;
            int bc = col + local_j;
            Bs[local_i][local_j] =
                (bk < K && bc < N) ? B[bk * N + bc] : 0.0f;
        }
        __syncthreads();
        for(int kk = 0;kk<BK;++kk){
            for (int ti =0;ti< TM ;++ti){
                a[ti] = As[ti+i*TM][kk];
            }
            for (int tj =0;tj <TN ;++tj){
                b[tj] = Bs[kk][tj+j*TN];
            }

            for (int ti =0;ti< TM ;++ti){
                for (int tj =0;tj <TN ;++tj){
                    c[ti][tj] += a[ti] *b[tj];
                }
            }
        }
        __syncthreads();
    }
    for (int ti =0;ti< TM ;++ti){
        for (int tj =0;tj <TN;++tj){
            int r  = row +i*TM+ ti;
            int c_col = col + j*TN +tj;
            if (r < M && c_col < N) {
                C[r *N+ c_col] = c[ti][tj];
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
