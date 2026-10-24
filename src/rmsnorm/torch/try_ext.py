import torch
from torch.utils.cpp_extension import load

rmsnorm_cuda = load(
    name="rmsnorm_cuda",
    sources=["src/rmsnorm/torch/rmsnorm_ext.cu"],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

def ref(x, w, eps):   # 用 float64 算参考结果，避免参考本身有误差
    x64, w64 = x.double(), w.double()
    return (x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * w64).float()

shapes = [(1, 17), (3, 256), (5, 1003), (7, 2048), (2, 4095), (3, 4096), (2, 4097),
          (4, 8192), (2, 16384), (2, 30000), (1, 40000),   # 最后一个走循环版本
          (2, 3, 4096), (0, 4096)]                        # 三维输入、空张量
torch.manual_seed(0)
for shape in shapes:
    x = torch.randn(*shape, device="cuda")
    w = torch.randn(shape[-1], device="cuda")
    y = rmsnorm_cuda.forward(x, w, 1e-5)
    torch.testing.assert_close(y, ref(x, w, 1e-5), rtol=2e-5, atol=2e-6)
    print("PASS", shape)
print("ALL PASS")