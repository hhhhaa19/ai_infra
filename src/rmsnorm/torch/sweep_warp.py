# 用法（在仓库根目录）：python src/rmsnorm/torch/sweep_warp.py
# warp 版 vs own_reg：rows 足够多（GPU 是瓶颈）时，细扫 n，找分界线
import statistics, sys, time, torch
sys.path.insert(0, "src/rmsnorm/torch")
from build_ext import load_rmsnorm
rms_norm = load_rmsnorm(); eps = 1e-5
def t(f, x, w, iters=50, samples=30):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); out = []
    for _ in range(samples):
        a.record()
        for _ in range(iters): f(x, w)
        b.record(); b.synchronize(); out.append(a.elapsed_time(b) * 1000 / iters)
    return statistics.median(out)
fs = {"warp": lambda x, w: rms_norm(x, w, eps, 1), "own_reg": lambda x, w: rms_norm(x, w, eps, 2)}
print("rows    n   warp_us  own_reg_us  warp/own_reg")
for rows in (4096, 16384):
    for n in (32, 64, 96, 128, 160, 192, 224, 256, 320, 384):
        x = torch.randn(rows, n, device="cuda"); w = torch.randn(n, device="cuda")
        end = time.time() + 0.5
        while time.time() < end:
            for f in fs.values():
                for _ in range(20): f(x, w)
            torch.cuda.synchronize()
        r = {k: [] for k in fs}
        for rnd in range(3):
            for k in (list(fs) if rnd % 2 == 0 else list(fs)[::-1]): r[k].append(t(fs[k], x, w))
        med = {k: statistics.median(v) for k, v in r.items()}
        print(f"{rows:5d} {n:4d} {med['warp']:9.2f} {med['own_reg']:10.2f}   {med['warp']/med['own_reg']:.2f}", flush=True)
