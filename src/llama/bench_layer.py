# 整层 decoder layer 计时（不开 nsys / profiler）：eager vs ours。
# 用法（在仓库根目录）：python src/llama/bench_layer.py [输出 csv]
# 计时：每个样本用 CUDA event 包住 N 次连续前向（prefill 10 次，decode 50 次）再除以 N；
# 30 个样本 × 3 轮，两种实现每轮交替先后，取三轮中位数的中位数。
# event 记录的是 GPU 流上的时间区间：GPU 等 CPU 发射造成的空闲也算在里面，所以它反映的是实际每层耗时。
import csv, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder_layer import DecoderLayer

SHAPES = {"prefill b1 s512": (1, 512, 10), "decode b1": (1, 1, 50), "decode b8": (8, 1, 50)}

def time_us(layer, x, iters, samples=30):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(samples):
        start.record()
        for _ in range(iters):
            layer(x)
        stop.record(); stop.synchronize()
        out.append(start.elapsed_time(stop) * 1000 / iters)
    return statistics.median(out)

torch.manual_seed(0)
base = DecoderLayer(impl="eager").cuda()
with torch.no_grad():
    base.norm1.weight.normal_(); base.norm2.weight.normal_()
layers = {}
for impl in ("eager", "ours"):
    layers[impl] = DecoderLayer(impl=impl).cuda(); layers[impl].load_state_dict(base.state_dict())

rows = []
print(f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)
with torch.inference_mode():
    for tag, (b, s, iters) in SHAPES.items():
        x = torch.randn(b, s, 2048, device="cuda")
        diff = (layers["eager"](x) - layers["ours"](x)).abs().max().item()
        until = time.time() + 2.0                      # 每个形状预热至少 2 秒
        while time.time() < until:
            for l in layers.values():
                for _ in range(5): l(x)
            torch.cuda.synchronize()
        rounds = {k: [] for k in layers}
        for r in range(3):
            for k in (["eager", "ours"] if r % 2 == 0 else ["ours", "eager"]):
                rounds[k].append(time_us(layers[k], x, iters))
        med = {k: statistics.median(v) for k, v in rounds.items()}
        spread = max((max(v) - min(v)) / min(v) for v in rounds.values()) * 100
        change = (med["ours"] - med["eager"]) / med["eager"] * 100
        rows.append([tag, f"{med['eager']:.1f}", f"{med['ours']:.1f}", f"{change:.1f}", f"{spread:.1f}", f"{diff:.2e}"])
        print(f"{tag:16s} eager={med['eager']:8.1f}us ours={med['ours']:8.1f}us change={change:+6.1f}%  spread={spread:.1f}%  maxdiff={diff:.1e}", flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else "bench_layer.csv"
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["shape", "eager_us", "ours_us", "change_pct", "max_round_spread_pct", "max_abs_diff"]); wr.writerows(rows)
print("saved", out)
