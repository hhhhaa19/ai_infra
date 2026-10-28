#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "../rmsNormWarp.cu"

// own_reg 的线程数：按每个线程处理几个元素（per）来定，per 看行数。
// Quest A100 实测（20261027_quest_sweep_block，n=2048）：
//   行少（1~64 行）只有几个 block，GPU 吃不满，速度由延迟决定 → 每行多线程（512x4）最快；
//   行多（512 行）→ block 小一点、一个 SM 同时放几个更好（256x8）。原来固定 per=16（128x16）在两种情况都最慢。
static int chooseBlock(int n, int rows) {
    static const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;  // 只看第一次用到的 GPU
    const int per = rows < 2 * sms ? 4 : 8;
    int b = 128;
    while (b < 1024 && b * per < n) b *= 2;
    return b;
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
static void launchRMSNorm(int rows, int n, const float* x, const float* w, float* y,
                          float eps, cudaStream_t s, int impl = 0){
    if (rows == 0 || n == 0) return;
    if (impl == 1) {
        TORCH_CHECK(launchWarpRow<4>(rows, n, x, w, y, eps, s), "warp version supports n <= 1024");
        return;
    }
    // Orin 实测（20261024_orin_bench_ext）：n <= 224 warp 版更快，256 打平，>= 320 own_reg 更快
    if (impl == 0 && n <= 256 && launchWarpRow<4>(rows, n, x, w, y, eps, s)) return;
    bool ok = false;
    // own_reg 范围内：n % 4 == 0 且对齐时用 float4 版（线程数一样，只是每个线程一次读 4 个）
    const bool vec4 = n % 4 == 0 && aligned16(x) && aligned16(w) && aligned16(y);
    TORCH_CHECK(impl != 4 || vec4, "float4 version needs n % 4 == 0 and 16-byte aligned pointers");
    if ((impl == 0 && vec4) || impl == 4) switch (chooseBlock(n, rows)) {
        case 128:  ok = launchVec4<128>(rows, n, x, w, y, eps, s);  break;
        case 256:  ok = launchVec4<256>(rows, n, x, w, y, eps, s);  break;
        case 512:  ok = launchVec4<512>(rows, n, x, w, y, eps, s);  break;
        case 1024: ok = launchVec4<1024>(rows, n, x, w, y, eps, s); break;
    }
    TORCH_CHECK(ok || impl != 4, "float4 version supports n <= 36864");
    if (!ok && impl != 3) switch (chooseBlock(n, rows)) {
        case 128:  ok = launchReg<128>(rows, n, x, w, y, eps, s);  break;
        case 256:  ok = launchReg<256>(rows, n, x, w, y, eps, s);  break;
        case 512:  ok = launchReg<512>(rows, n, x, w, y, eps, s);  break;
        case 1024: ok = launchReg<1024>(rows, n, x, w, y, eps, s); break;
    }
    TORCH_CHECK(ok || impl != 2, "own_reg supports n <= 33792");
    if (!ok)
        rmsNormNT<512><<<rows,512,(512/32)* sizeof(float),s>>>(x,w,y,n,eps);
}
// impl：0 自动分档（默认）；1/2/3/4 强制 warp 版 / own_reg / 循环版 / float4 版，只给测试和对比用
torch::Tensor forward(torch::Tensor x, torch::Tensor weight, double eps, int64_t impl) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    // 数据类型：kernel 目前只支持 float32
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    // 形状：x 至少一维；weight 是一维，长度等于 x 的最后一维
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
                "weight must have shape [n], where n = x.size(-1)");
    // 内存布局：先只支持连续存放
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (x.numel() == 0) return torch::empty_like(x);   // 空张量：避免下面除以 0
    const int n = x.size(-1);
    const int rows = x.numel() / n;               // 前面的维度全部展平成行
    auto y = torch::empty_like(x);                // 在 GPU 上分配输出，形状和 x 一样
    launchRMSNorm(rows, n,
                  x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
                  static_cast<float>(eps), at::cuda::getCurrentCUDAStream(), static_cast<int>(impl));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); 
    
    return y;
}
// 注册成 PyTorch 正式算子：torch.ops.ai_infra.rms_norm(x, weight, eps, impl=0)
// ③ Meta 实现：只根据输入推出输出形状，不计算（torch.compile 推形状时用）
torch::Tensor forward_meta(torch::Tensor x, torch::Tensor weight, double eps, int64_t impl) {
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
                "weight must have shape [n], where n = x.size(-1)");
    return torch::empty_like(x);
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

std::tuple<torch::Tensor, torch::Tensor> fused_forward(
        torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda(), "x, residual, weight must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && residual.scalar_type() == torch::kFloat32 &&
                weight.scalar_type() == torch::kFloat32, "x, residual, weight must be float32");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual must have the same shape as x");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
                "weight must have shape [n], where n = x.size(-1)");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
                "x, residual, weight must be contiguous");

    auto h = torch::empty_like(x);
    auto y = torch::empty_like(x);
    if (x.numel() == 0) return {h, y};
    const int n = x.size(-1);
    const int rows = x.numel() / n;
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    const float* xp = x.data_ptr<float>();
    const float* rp = residual.data_ptr<float>();
    const float* wp = weight.data_ptr<float>();
    float* hp = h.data_ptr<float>();
    float* yp = y.data_ptr<float>();
    const float e = static_cast<float>(eps);

    bool ok = false;
    const bool vec4 = n % 4 == 0 && aligned16(xp) && aligned16(rp) && aligned16(wp) && aligned16(hp) && aligned16(yp);
    if (vec4) switch (chooseBlock(n, rows)) {
        case 128:  ok = launchFusedVec4<128>(rows, n, xp, rp, wp, hp, yp, e, s);  break;
        case 256:  ok = launchFusedVec4<256>(rows, n, xp, rp, wp, hp, yp, e, s);  break;
        case 512:  ok = launchFusedVec4<512>(rows, n, xp, rp, wp, hp, yp, e, s);  break;
        case 1024: ok = launchFusedVec4<1024>(rows, n, xp, rp, wp, hp, yp, e, s); break;
    }
    if (!ok) switch (chooseBlock(n, rows)) {
        case 128:  ok = launchFusedReg<128>(rows, n, xp, rp, wp, hp, yp, e, s);  break;
        case 256:  ok = launchFusedReg<256>(rows, n, xp, rp, wp, hp, yp, e, s);  break;
        case 512:  ok = launchFusedReg<512>(rows, n, xp, rp, wp, hp, yp, e, s);  break;
        case 1024: ok = launchFusedReg<1024>(rows, n, xp, rp, wp, hp, yp, e, s); break;
    }
    TORCH_CHECK(ok, "fused_add_rms_norm supports n <= 33792 (n <= 36864 when n % 4 == 0)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h, y};
}

// Meta 实现：两个输出都和 x 同形状
std::tuple<torch::Tensor, torch::Tensor> fused_forward_meta(
        torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual must have the same shape as x");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
                "weight must have shape [n], where n = x.size(-1)");
    return {torch::empty_like(x), torch::empty_like(x)};
}

// ① 声明：在 ai_infra 命名空间下登记 rms_norm 的签名
TORCH_LIBRARY(ai_infra, m) {
    m.def("rms_norm(Tensor x, Tensor weight, float eps, int impl=0) -> Tensor");
    m.def("fused_add_rms_norm(Tensor x, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)");
}

// ② 输入在 GPU 上时，dispatcher 调用 forward
TORCH_LIBRARY_IMPL(ai_infra, CUDA, m) {
    m.impl("rms_norm", &forward);
    m.impl("fused_add_rms_norm", &fused_forward);
}

// ③ 输入是 Meta 张量时，dispatcher 调用 forward_meta
TORCH_LIBRARY_IMPL(ai_infra, Meta, m) {
    m.impl("rms_norm", &forward_meta);
    m.impl("fused_add_rms_norm", &fused_forward_meta);
}
