// 手写 attention（FP32）。这个文件由你来写主体；attention_bench.cu 负责准备数据、和 CPU double 结果比对、计时。
//
// 形状（AttnShape）：
//   Q、O : [B][Hq][N][d]      K、V : [B][Hkv][N][d]      全部行优先、连续存储
//   GQA：Hq 个 Q 头共用 Hkv 个 K/V 头，Q 头 h 用第 h / (Hq/Hkv) 个 K/V 头。
//        SmolVLA 是 15 / 5：每 3 个 Q 头共用一组 K、V。Hq == Hkv 就是普通多头。
//   causal = true：第 i 行只看 j ≤ i，j > i 的分数当作 -inf（softmax 后是 0）。
//
// 每个 (b, h) 互相独立，计算三步：
//   S  = scale · Qh · Kᵀ      [N×d]·[d×N] = [N×N]，scale = 1/√d
//   P  = softmax(S)           按行，每行 N 个数
//   Oh = P · V                [N×N]·[N×d] = [N×d]
// 计算量 4·N²·d FLOP（两次 GEMM 各 2·N²·d）；causal 只算一半。
//
// 每一版写好后，把 launch 里的 return false 改成启动 kernel 并 return true。
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "../gemm/kernels.cuh"      // launch_tiled、launch_reg（P·V 直接用）
#include "../softmax/kernels.cuh"   // launch_warp（n ≤ 1024）、launch_online（任意 n）

struct AttnShape {
    const char* name;
    int B, Hq, Hkv, N, d;
    bool causal;
    int group() const { return Hq / Hkv; }                    // 几个 Q 头共用一组 K/V
    float scale() const { return 1.0f / std::sqrt(float(d)); }
};

// ───────────── 第 1 版：朴素版，三步三个 kernel，S 存在显存里 ─────────────
// 像卷积一次处理一张图一样，这里一次处理一个 (b, h)，循环 B·Hq 次，每次三个 launch：
//   ① launch_gemm_nt：S = scale · Qh · Kᵀ
//   ② softmax：S 的 N 行各做一次（in-place，X 和 Y 都传 S 就行）
//   ③ launch_reg：Oh = P · V   （M = N，N = d，K = N）
// S 的工作区：N×N 个 float，所有 (b, h) 轮流用同一块。

// ① 的 kernel：C[i][j] = scale · Σ_k A[i][k] · Bt[j][k]      A: M×K，Bt: N×K，C: M×N
// 照着你的 gemm_tiled 写，只有搬 B 那一行不同：
//   gemm_tiled 里要的是 B[k][j]；这里没有 B，只有 Bt（就是 K 矩阵，N×d 行优先），
//   B[k][j] 就是 Bt[j][k]。想一想 Bs[i][j] 那一行要怎么改（手算第 ④ 题）。
// 写回 C 的时候乘 scale；causal 且列号 > 行号时写 -INFINITY。
template <int TILE>
__global__ void gemm_nt(const float* __restrict__ A, const float* __restrict__ Bt, float* __restrict__ C,
                        int M, int N, int K, float scale, bool causal) {
    // TODO
}

inline bool launch_gemm_nt(const float* A, const float* Bt, float* C, int M, int N, int K,
                           float scale, bool causal, cudaStream_t s) {
    // TODO：和 launch_tiled 一样，TILE = 16
    return false;
}

// S 的工作区：只增不减，第一次用到多大就申请多大（和卷积的 im2col_workspace 一样）
inline float* attn_workspace(size_t count) {
    static float* buf = nullptr;
    static size_t cap = 0;
    if (count > cap) {
        if (buf) cudaFree(buf);
        cudaMalloc(&buf, count * sizeof(float));
        cap = count;
    }
    return buf;
}

inline bool launch_attn_naive(const float* Q, const float* K, const float* V, float* O,
                              const AttnShape& sh, cudaStream_t s) {
    // TODO：
    //   float* S = attn_workspace(size_t(sh.N) * sh.N);
    //   for b, for h：
    //     kv = h / sh.group()
    //     Qh、Oh 的起点：第 (b·Hq + h) 个 N×d 块；Kh、Vh 的起点：第 (b·Hkv + kv) 个 N×d 块（偏移用 size_t）
    //     ① launch_gemm_nt(Qh, Kh, S, N, N, d, scale, causal, s)
    //     ② softmax：N ≤ 1024 用 launch_warp，否则 launch_online
    //     ③ launch_reg(S, Vh, Oh, N, d, N, s)
    return false;
}

// ───────────── 第 2 版：FlashAttention v1（11-26 再写）─────────────
// 按块读 K、V，S 只在片上存在，用 online softmax 把每块的结果合并进 O。
inline bool launch_attn_flash(const float* Q, const float* K, const float* V, float* O,
                              const AttnShape& sh, cudaStream_t s) {
    return false;
}
