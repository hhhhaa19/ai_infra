# 正确性测试：自动分档 + 强制每条路径，和 float64 的 PyTorch 参考对比；再测输入检查。
# 用法（在仓库根目录）：python src/rmsnorm/torch/test_ext.py
import torch
from build_ext import load_rmsnorm

rmsnorm_cuda = load_rmsnorm()

def ref(x, w, eps):
    x64, w64 = x.double(), w.double()
    return (x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * w64).float()

def check(y, x, w, eps, tag):
    torch.testing.assert_close(y, ref(x, w, eps), rtol=2e-5, atol=2e-6, msg=lambda m: f"{tag}: {m}")

torch.manual_seed(0)
eps = 1e-5
IMPL = {0: "auto", 1: "warp", 2: "own_reg", 3: "loop"}

# 1. 自动分档：覆盖 warp 版（n<=1024，rows 不是 4 的倍数）、各档 own_reg、循环版、多维、空张量
shapes = [(1, 1), (5, 17), (7, 256), (6, 1003), (3, 1024), (9, 1025),
          (1, 17), (3, 256), (5, 1003), (7, 2048), (2, 4095), (3, 4096), (2, 4097),
          (4, 8192), (2, 16384), (2, 30000), (2, 33792), (1, 33793), (1, 40000),
          (2, 3, 4096), (4, 5, 300), (0, 4096), (3, 0)]
for shape in shapes:
    x = torch.randn(*shape, device="cuda"); w = torch.randn(shape[-1], device="cuda")
    check(rmsnorm_cuda.forward(x, w, eps), x, w, eps, f"auto {shape}")
print(f"auto: {len(shapes)} shapes PASS")

# 2. 强制每条路径：同一个 n 三种写法都要对
count = 0
for rows in (1, 3, 4, 5, 64):
    for n in (1, 17, 32, 33, 255, 256, 257, 1000, 1023, 1024):
        x = torch.randn(rows, n, device="cuda"); w = torch.randn(n, device="cuda")
        for impl in (1, 2, 3):
            check(rmsnorm_cuda.forward_impl(x, w, eps, impl), x, w, eps, f"{IMPL[impl]} ({rows},{n})")
            count += 1
for n in (1025, 4095, 4096, 4097, 33792):
    x = torch.randn(3, n, device="cuda"); w = torch.randn(n, device="cuda")
    for impl in (2, 3):
        check(rmsnorm_cuda.forward_impl(x, w, eps, impl), x, w, eps, f"{IMPL[impl]} (3,{n})"); count += 1
print(f"forced paths: {count} cases PASS")

# 3. 输入检查：每个都应该报 RuntimeError
x = torch.randn(4, 8, device="cuda"); w = torch.ones(8, device="cuda")
bad = [("CPU x", lambda: rmsnorm_cuda.forward(x.cpu(), w, eps)),
       ("float16 x", lambda: rmsnorm_cuda.forward(x.half(), w, eps)),
       ("wrong weight length", lambda: rmsnorm_cuda.forward(x, w[:4], eps)),
       ("non-contiguous x", lambda: rmsnorm_cuda.forward(x.t(), w[:4], eps)),
       ("warp forced, n>1024", lambda: rmsnorm_cuda.forward_impl(torch.randn(2, 2048, device="cuda"), torch.ones(2048, device="cuda"), eps, 1))]
for name, f in bad:
    try:
        f(); raise SystemExit(f"{name}: expected an error")
    except RuntimeError as e:
        print(f"check ok: {name} -> {str(e).splitlines()[0]}")
print("ALL PASS")
