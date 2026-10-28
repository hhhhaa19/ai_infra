#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "../rmsnorm_launch.cuh"   // 选档和启动（不依赖 PyTorch，TensorRT plugin 也用它）

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
    const char* err = launchRMSNorm(rows, n,
                  x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
                  static_cast<float>(eps), at::cuda::getCurrentCUDAStream(), static_cast<int>(impl));
    TORCH_CHECK(err == nullptr, err);
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
    const char* err = launchFusedAddRMSNorm(rows, n,
        x.data_ptr<float>(), residual.data_ptr<float>(), weight.data_ptr<float>(),
        h.data_ptr<float>(), y.data_ptr<float>(), static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == nullptr, err);
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
