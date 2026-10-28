#pragma once
// RMSNorm 的选档和启动：只依赖 CUDA Runtime，不依赖 PyTorch。
// PyTorch 扩展（torch/rmsnorm_ext.cu）和 TensorRT plugin 共用这一份：
// 它们各自负责把自己的张量拆成指针 + 形状，再调这里的 launchRMSNorm / launchFusedAddRMSNorm。
// 注意：这里 include 了 rmsNormWarp.cu（里面有非模板 kernel），每个 .so 只能有一个 .cu 包含本文件。
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include "rmsNormWarp.cu"

// own_reg 的线程数：按每个线程处理几个元素（per）来定，per 看行数。
// Quest A100 实测（20261027_quest_sweep_block，n=2048）：
//   行少（1~64 行）只有几个 block，GPU 吃不满，速度由延迟决定 → 每行多线程（512x4）最快；
//   行多（512 行）→ block 小一点、一个 SM 同时放几个更好（256x8）。原来固定 per=16（128x16）在两种情况都最慢。
static int chooseBlock(int n, int rows) {
    static const int sms = [] {                    // 只看第一次用到的 GPU
        int dev = 0, count = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev);
        return count;
    }();
    const int per = rows < 2 * sms ? 4 : 8;
    int b = 128;
    while (b < 1024 && b * per < n) b *= 2;
    return b;
}

// 把运行时算出的线程数 b 变成编译期常量，交给 f：f 收到 std::integral_constant<int, B>，
// 里面用 decltype(blk)::value 取出 B 当模板参数。四个 launchXxx<BLOCK> 共用这一个 switch。
template <class F>
static bool withBlock(int b, F f) {
    switch (b) {
        case 128:  return f(std::integral_constant<int, 128>{});
        case 256:  return f(std::integral_constant<int, 256>{});
        case 512:  return f(std::integral_constant<int, 512>{});
        case 1024: return f(std::integral_constant<int, 1024>{});
    }
    return false;
}

template <int BLOCK>
static bool launchReg (int rows, int n, const float* x, const float* w, float* y,
    float eps,cudaStream_t stream) {
    const int need = (n+BLOCK -1)/BLOCK;
    const size_t shm = (BLOCK/32) * sizeof(float);
#define REG_CASE(P) if (need <= P) {rmsNormReg<BLOCK,P><<<rows,BLOCK,shm,stream>>>(x,w,y,n,eps);return true;}
    REG_CASE(1) REG_CASE(2) REG_CASE(4) REG_CASE(8) REG_CASE(9)
    REG_CASE(16) REG_CASE(17) REG_CASE(32) REG_CASE(33)
#undef REG_CASE
    return false;
}


// float4 版能不能用：n 是 4 的倍数（每行开头才对齐），所有指针 16 字节对齐（切片等偏移过的张量可能不满足）
static bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// 和 launchReg 一样选模板参数，只是 need 以 float4 为单位：线程数相同时，VPT = PER_THREAD / 4
template <int BLOCK>
static bool launchVec4(int rows, int n, const float* x, const float* w, float* y,
                       float eps, cudaStream_t stream) {
    const int need = (n / 4 + BLOCK - 1) / BLOCK;      // 每个线程要处理几个 float4
    const size_t shm = (BLOCK / 32) * sizeof(float);
#define VEC_CASE(V) if (need <= V) { rmsNormVec4<BLOCK, V><<<rows, BLOCK, shm, stream>>>(x, w, y, n, eps); return true; }
    VEC_CASE(1) VEC_CASE(2) VEC_CASE(3) VEC_CASE(4) VEC_CASE(5) VEC_CASE(8) VEC_CASE(9)
#undef VEC_CASE
    return false;
}

template <int WARPS>
static bool launchWarpRow(int rows, int n, const float* x, const float* w, float* y,
                          float eps, cudaStream_t stream) {
    const int need = (n + 31) / 32;                    // 每个线程至少要处理几个
    const int blocks = (rows + WARPS - 1) / WARPS;     // 每个 block 处理 WARPS 行
#define WARP_CASE(P) if (need <= P) { rmsNormWarpRow<P, WARPS><<<blocks, WARPS * 32, 0, stream>>>(x, w, y, rows, n, eps); return true; }
    WARP_CASE(1) WARP_CASE(2) WARP_CASE(4) WARP_CASE(8) WARP_CASE(9)
    WARP_CASE(16) WARP_CASE(17) WARP_CASE(32)
#undef WARP_CASE
    return false;
}

// impl：0 自动分档；1 强制 warp 版；2 强制 own_reg；3 强制循环版；4 强制 float4 版（1~4 只用于测试和对比）
// 返回 nullptr 表示已经启动；否则返回错误信息（参数不支持，什么都没启动）。
// kernel 启动本身的错误（cudaGetLastError）由调用方检查。
static const char* launchRMSNorm(int rows, int n, const float* x, const float* w, float* y,
                                 float eps, cudaStream_t s, int impl = 0) {
    if (rows == 0 || n == 0) return nullptr;
    if (impl == 1)
        return launchWarpRow<4>(rows, n, x, w, y, eps, s) ? nullptr : "warp version supports n <= 1024";
    // Orin 实测（20261024_orin_bench_ext）：n <= 224 warp 版更快，256 打平，>= 320 own_reg 更快
    if (impl == 0 && n <= 256 && launchWarpRow<4>(rows, n, x, w, y, eps, s)) return nullptr;
    const int b = chooseBlock(n, rows);
    // own_reg 范围内：n % 4 == 0 且对齐时用 float4 版（线程数一样，只是每个线程一次读 4 个）
    const bool vec4 = n % 4 == 0 && aligned16(x) && aligned16(w) && aligned16(y);
    if (impl == 4 && !vec4) return "float4 version needs n % 4 == 0 and 16-byte aligned pointers";
    if ((impl == 0 && vec4) || impl == 4) {
        if (withBlock(b, [&](auto blk) { return launchVec4<decltype(blk)::value>(rows, n, x, w, y, eps, s); }))
            return nullptr;
        if (impl == 4) return "float4 version supports n <= 36864";
    }
    if (impl != 3) {
        if (withBlock(b, [&](auto blk) { return launchReg<decltype(blk)::value>(rows, n, x, w, y, eps, s); }))
            return nullptr;
        if (impl == 2) return "own_reg supports n <= 33792";
    }
    rmsNormNT<512><<<rows,512,(512/32)* sizeof(float),s>>>(x,w,y,n,eps);
    return nullptr;
}

// ===== fused add + RMSNorm：h = x + residual；y = rmsnorm(h) * weight =====
// n % 4 == 0 且对齐时用 float4 版，否则用 own_reg（n <= 33792）；h 只写一次，归一化直接用寄存器里的值。

// 和 launchReg 一样按每个线程要处理几个元素（need）选模板参数，只是换成融合版 kernel
template <int BLOCK>
static bool launchFusedReg(int rows, int n, const float* x, const float* r, const float* w,
                           float* h, float* y, float eps, cudaStream_t stream) {
    const int need = (n + BLOCK - 1) / BLOCK;
    const size_t shm = (BLOCK / 32) * sizeof(float);
#define FUSED_CASE(P) if (need <= P) { fusedAddRmsNormReg<BLOCK, P><<<rows, BLOCK, shm, stream>>>(x, r, w, h, y, n, eps); return true; }
    FUSED_CASE(1) FUSED_CASE(2) FUSED_CASE(4) FUSED_CASE(8) FUSED_CASE(9)
    FUSED_CASE(16) FUSED_CASE(17) FUSED_CASE(32) FUSED_CASE(33)
#undef FUSED_CASE
    return false;
}

template <int BLOCK>
static bool launchFusedVec4(int rows, int n, const float* x, const float* r, const float* w,
                            float* h, float* y, float eps, cudaStream_t stream) {
    const int need = (n / 4 + BLOCK - 1) / BLOCK;
    const size_t shm = (BLOCK / 32) * sizeof(float);
#define FUSED_VEC_CASE(V) if (need <= V) { fusedAddRmsNormVec4<BLOCK, V><<<rows, BLOCK, shm, stream>>>(x, r, w, h, y, n, eps); return true; }
    FUSED_VEC_CASE(1) FUSED_VEC_CASE(2) FUSED_VEC_CASE(3) FUSED_VEC_CASE(4) FUSED_VEC_CASE(5) FUSED_VEC_CASE(8) FUSED_VEC_CASE(9)
#undef FUSED_VEC_CASE
    return false;
}

// 返回值约定同 launchRMSNorm
static const char* launchFusedAddRMSNorm(int rows, int n, const float* x, const float* r, const float* w,
                                         float* h, float* y, float eps, cudaStream_t s) {
    if (rows == 0 || n == 0) return nullptr;
    const int b = chooseBlock(n, rows);
    const bool vec4 = n % 4 == 0 && aligned16(x) && aligned16(r) && aligned16(w) && aligned16(h) && aligned16(y);
    if (vec4 && withBlock(b, [&](auto blk) { return launchFusedVec4<decltype(blk)::value>(rows, n, x, r, w, h, y, eps, s); }))
        return nullptr;
    if (withBlock(b, [&](auto blk) { return launchFusedReg<decltype(blk)::value>(rows, n, x, r, w, h, y, eps, s); }))
        return nullptr;
    return "fused_add_rms_norm supports n <= 33792 (n <= 36864 when n % 4 == 0)";
}
