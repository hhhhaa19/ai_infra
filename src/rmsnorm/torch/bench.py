# 单算子性能：我们的扩展（自动分档 / 强制各写法）vs PyTorch eager 写法。
# 用法：python src/rmsnorm/torch/bench.py [输出 csv]
# 计时：每种形状先预热；每个样本用 CUDA event 包住 50 次连续调用再除以 50；
# 每个配置 30 个样本 × 3 轮，各实现在每轮内交替执行，取三轮中位数的中位数。
# 计时包含 Python 调用和 kernel 启动的开销（这也是在 PyTorch 里实际用到的成本）。
import csv, statistics, sys, time
import torch
from build_ext import load_rmsnorm

rms_norm = load_rmsnorm()
eps = 1e-5

def eager(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

def impls_for(n):
    fs = {"ours": lambda x, w: rms_norm(x, w, eps), "eager": eager}
    if n <= 1024:
        fs["warp"] = lambda x, w: rms_norm(x, w, eps, 1)
    if n <= 33792:
        fs["own_reg"] = lambda x, w: rms_norm(x, w, eps, 2)
    return fs

def time_us(f, x, w, iters=50, samples=30):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(samples):
        start.record()
        for _ in range(iters):
            f(x, w)
        stop.record(); stop.synchronize()
        out.append(start.elapsed_time(stop) * 1000 / iters)
    return statistics.median(out)

rows_list = [64, 512, 4096]
n_list = [128, 256, 512, 768, 1024, 1536, 2048, 4096, 8192]
results = []
print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
for rows in rows_list:
    for n in n_list:
        x = torch.randn(rows, n, device="cuda"); w = torch.randn(n, device="cuda")
        fs = impls_for(n)
        until = time.time() + 1.0                      # 每种形状预热至少 1 秒
        while time.time() < until:
            for f in fs.values():
                for _ in range(20): f(x, w)
            torch.cuda.synchronize()
        rounds = {k: [] for k in fs}
        names = list(fs)
        for r in range(3):
            for k in names[r % len(names):] + names[:r % len(names)]:   # 每轮换起点
                rounds[k].append(time_us(fs[k], x, w))
        med = {k: statistics.median(v) for k, v in rounds.items()}
        spread = max((max(v) - min(v)) / min(v) for v in rounds.values()) * 100
        results.append((rows, n, med, spread))
        print(f"rows={rows:5d} n={n:5d} " + " ".join(f"{k}={v:8.2f}us" for k, v in med.items())
              + f"  ours/eager={med['ours']/med['eager']:.2f}  spread={spread:.1f}%", flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else "bench_ext.csv"
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["rows", "n", "impl", "us", "max_round_spread_pct"])
    for rows, n, med, spread in results:
        for k, v in med.items():
            wr.writerow([rows, n, k, f"{v:.3f}", f"{spread:.1f}"])
print("saved", out)
