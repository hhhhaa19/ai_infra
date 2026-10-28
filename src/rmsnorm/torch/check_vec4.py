# float4 版的正确性：各种形状对照 PyTorch、不对齐 / n % 4 != 0 时退回普通版、强制 float4 时的报错。
# 用法（在仓库根目录）：python src/rmsnorm/torch/check_vec4.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from build_ext import load_rmsnorm

rms = load_rmsnorm(); fused = torch.ops.ai_infra.fused_add_rms_norm
torch.manual_seed(0); eps = 1e-5
ref = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
rel = lambda a, b: ((a - b).abs() / (b.abs() + 1e-3)).max().item()

def misaligned(rows, n):
    # 连续，但起始地址偏 4 字节：float4 版不能用，要退回普通版
    buf = torch.randn(rows * n + 1, device="cuda")
    t = buf[1:].view(rows, n)
    assert t.is_contiguous() and t.data_ptr() % 16 != 0
    return t

worst = 0.0
shapes = [(1, 260), (3, 512), (1, 896), (1, 2048), (8, 2048), (64, 2048), (512, 2048),
          (2, 4096), (1, 5000), (300, 4096), (1, 16384), (1, 33792), (1, 36864)]
for rows, n in shapes:
    x = torch.randn(rows, n, device="cuda"); r = torch.randn_like(x); w = torch.randn(n, device="cuda")
    worst = max(worst, rel(rms(x, w, eps, 4), ref(x, w)))      # 强制 float4
    worst = max(worst, rel(rms(x, w, eps, 0), ref(x, w)))      # 自动
    if n <= 33792 or n % 4 == 0:
        h, y = fused(x, r, w, eps)
        assert torch.equal(h, x + r), (rows, n)
        worst = max(worst, rel(y, ref(x + r, w)))
assert worst < 1e-4, worst
print(f"float4 + auto ok on {len(shapes)} shapes, worst rel err {worst:.1e}")

# 不对齐：自动档退回普通版，结果正确；强制 float4 要报错
for rows, n in [(1, 2048), (8, 2048), (3, 896)]:
    x = misaligned(rows, n); r = misaligned(rows, n); w = torch.randn(n, device="cuda")
    assert rel(rms(x, w, eps, 0), ref(x, w)) < 1e-4
    h, y = fused(x, r, w, eps)
    assert torch.equal(h, x + r) and rel(y, ref(x + r, w)) < 1e-4
print("misaligned inputs fall back ok")
for name, x, w in [("misaligned", misaligned(1, 2048), torch.randn(2048, device="cuda")),
                   ("n % 4 != 0", torch.randn(1, 2050, device="cuda"), torch.randn(2050, device="cuda")),
                   ("n too large", torch.randn(1, 36868, device="cuda"), torch.randn(36868, device="cuda"))]:
    try:
        rms(x, w, eps, 4); torch.cuda.synchronize(); print("NOT RAISED", name)
    except RuntimeError as e:
        print("raised ok:", name, "|", str(e).splitlines()[0])
# n 不是 4 的倍数：自动档和融合版仍然正确（走普通版）
x = torch.randn(4, 2050, device="cuda"); r = torch.randn_like(x); w = torch.randn(2050, device="cuda")
assert rel(rms(x, w, eps, 0), ref(x, w)) < 1e-4
h, y = fused(x, r, w, eps); assert torch.equal(h, x + r) and rel(y, ref(x + r, w)) < 1e-4
print("n % 4 != 0 falls back ok")
