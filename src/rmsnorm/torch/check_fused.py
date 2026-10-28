# fused_add_rms_norm 的正确性：CUDA 结果对照 PyTorch、3 维输入、Meta 推形状、该报错的输入。
# 用法（在仓库根目录）：python src/rmsnorm/torch/check_fused.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import torch
from build_ext import load_rmsnorm

t0 = time.time(); load_rmsnorm(); print(f"build/load {time.time() - t0:.1f}s")
fused = torch.ops.ai_infra.fused_add_rms_norm
torch.manual_seed(0)
eps = 1e-5
worst = 0.0
for rows, n in [(1, 1), (3, 31), (5, 128), (7, 200), (4, 256), (2, 896), (1, 2048), (8, 2048),
                (512, 2048), (3, 4096), (2, 5000), (2, 8192), (1, 16384), (1, 33792)]:
    x = torch.randn(rows, n, device="cuda"); r = torch.randn(rows, n, device="cuda")
    w = torch.randn(n, device="cuda")
    h, y = fused(x, r, w, eps)
    h_ref = x + r
    y_ref = h_ref * torch.rsqrt(h_ref.pow(2).mean(-1, keepdim=True) + eps) * w
    assert torch.equal(h, h_ref), (rows, n, "h")
    err = ((y - y_ref).abs() / (y_ref.abs() + 1e-3)).max().item()
    worst = max(worst, err)
    assert err < 1e-4, (rows, n, err)
print(f"cuda ok, worst rel err {worst:.1e}")

# 3 维输入（和 decoder layer 一样的 [b, s, hidden]）
x = torch.randn(2, 3, 2048, device="cuda"); r = torch.randn_like(x); w = torch.randn(2048, device="cuda")
h, y = fused(x, r, w, eps)
assert h.shape == x.shape and y.shape == x.shape and torch.equal(h, x + r)
print("3-d ok")

# Meta：只推形状
hm, ym = fused(torch.empty(2, 3, 2048, device="meta"), torch.empty(2, 3, 2048, device="meta"),
               torch.empty(2048, device="meta"), eps)
print("meta ok", tuple(hm.shape), tuple(ym.shape), hm.device)

# 应该报错的输入
for name, args in [("n too large", (torch.randn(1, 33793, device="cuda"), torch.randn(1, 33793, device="cuda"),
                                    torch.randn(33793, device="cuda"))),
                   ("residual shape", (torch.randn(2, 2048, device="cuda"), torch.randn(1, 2048, device="cuda"),
                                       torch.randn(2048, device="cuda")))]:
    try:
        fused(*args, eps); torch.cuda.synchronize(); print("NOT RAISED", name)
    except RuntimeError as e:
        print("raised ok:", name, "|", str(e).splitlines()[0])

# rms_norm 没被影响
x = torch.randn(4, 2048, device="cuda"); w = torch.randn(2048, device="cuda")
ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
assert (torch.ops.ai_infra.rms_norm(x, w, eps) - ref).abs().max().item() < 1e-4
print("rms_norm still ok")
