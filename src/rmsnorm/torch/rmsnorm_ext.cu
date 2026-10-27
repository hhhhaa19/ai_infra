#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "../rmsNormWarp.cu"

static int chooseBlock(int n) {
    int b =128;
    while (b<1024 && b* 16 <n) b*=2;
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

// impl：0 自动分档；1 强制 warp 版；2 强制 own_reg；3 强制循环版（后三种只用于测试和对比）
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
    if (impl != 3) switch (chooseBlock(n)) {
        case 128:  ok = launchReg<128>(rows, n, x, w, y, eps, s);  break;
        case 256:  ok = launchReg<256>(rows, n, x, w, y, eps, s);  break;
        case 512:  ok = launchReg<512>(rows, n, x, w, y, eps, s);  break;
        case 1024: ok = launchReg<1024>(rows, n, x, w, y, eps, s); break;
    }
    TORCH_CHECK(ok || impl != 2, "own_reg supports n <= 33792");
    if (!ok)
        rmsNormNT<512><<<rows,512,(512/32)* sizeof(float),s>>>(x,w,y,n,eps);
}
// impl：0 自动分档（默认）；1/2/3 强制 warp 版 / own_reg / 循环版，只给测试和对比用
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
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RMSNorm forward (CUDA); impl: 0 auto, 1 warp, 2 own_reg, 3 loop",
          py::arg("x"), py::arg("weight"), py::arg("eps"), py::arg("impl") = 0);
}
// 注册成 PyTorch 正式算子：torch.ops.ai_infra.rms_norm(x, weight, eps, impl=0)
// ③ Meta 实现：只根据输入推出输出形状，不计算（torch.compile 推形状时用）
torch::Tensor forward_meta(torch::Tensor x, torch::Tensor weight, double eps, int64_t impl) {
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
                "weight must have shape [n], where n = x.size(-1)");
    return torch::empty_like(x);
}

// ① 声明：在 ai_infra 命名空间下登记 rms_norm 的签名
TORCH_LIBRARY(ai_infra, m) {
    m.def("rms_norm(Tensor x, Tensor weight, float eps, int impl=0) -> Tensor");
}

// ② 输入在 GPU 上时，dispatcher 调用 forward
TORCH_LIBRARY_IMPL(ai_infra, CUDA, m) {
    m.impl("rms_norm", &forward);
}

// ③ 输入是 Meta 张量时，dispatcher 调用 forward_meta
TORCH_LIBRARY_IMPL(ai_infra, Meta, m) {
    m.impl("rms_norm", &forward_meta);
}
