# 整层 decoder layer 计时（不开 nsys / profiler）：eager / ours / fused，各自再加手动 CUDA Graph（vLLM 式用法）。
# 用法（在仓库根目录）：python src/llama/bench_layer.py [输出 csv]
# 计时：每个样本用 CUDA event 包住 N 次连续前向（prefill 10 次，decode 50 次）再除以 N；
# 30 个样本 × 3 轮，每轮换起点（6 种写法、每轮挪 2 个），取三轮中位数的中位数。
# event 记录的是 GPU 流上的时间区间：GPU 等 CPU 发射造成的空闲也算在里面，所以它反映的是实际每层耗时。
# CUDA Graph 版：torch.cuda.graph 录一次前向，之后每次 g.replay()，输入用录制时的固定张量。
import csv, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder_layer import DecoderLayer

SHAPES = {"prefill b1 s512": (1, 512, 10), "decode b1": (1, 1, 50), "decode b8": (8, 1, 50)}
IMPLS = ("eager", "ours", "fused")

def time_us(fn, iters, samples=30):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(samples):
        start.record()
        for _ in range(iters):
            fn()
        stop.record(); stop.synchronize()
        out.append(start.elapsed_time(stop) * 1000 / iters)
    return statistics.median(out)

def graphed(layer, x):
    # 先在旁路 stream 上跑几次（分配好缓冲区），再录一次前向；返回 (重放函数, 录制时的输出张量)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): layer(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = layer(x)
    return g.replay, y

torch.manual_seed(0)
base = DecoderLayer(impl="eager").cuda()
with torch.no_grad():
    base.norm1.weight.normal_(); base.norm2.weight.normal_()
layers = {}
for impl in IMPLS:
    layers[impl] = DecoderLayer(impl=impl).cuda().eval(); layers[impl].load_state_dict(base.state_dict())

rows = []
print(f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)
with torch.inference_mode():
    for tag, (b, s, iters) in SHAPES.items():
        x = torch.randn(b, s, 2048, device="cuda")
        ref = layers["eager"](x).clone()
        fns = {}
        for impl in IMPLS:
            fns[impl] = (lambda l=layers[impl]: l(x))
            replay, y = graphed(layers[impl], x)
            replay(); torch.cuda.synchronize()
            fns[impl + "+cg"] = replay
            for k, out in ((impl, layers[impl](x)), (impl + "+cg", y)):
                err = (out - ref).abs().max().item()
                assert err < 1e-3, (tag, k, err)
        until = time.time() + 2.0                      # 每个形状预热至少 2 秒
        while time.time() < until:
            for f in fns.values():
                for _ in range(5): f()
            torch.cuda.synchronize()
        names = list(fns)
        rounds = {k: [] for k in names}
        for r in range(3):
            for k in names[2 * r:] + names[:2 * r]:    # 每轮换起点
                rounds[k].append(time_us(fns[k], iters))
        med = {k: statistics.median(v) for k, v in rounds.items()}
        spread = max((max(v) - min(v)) / min(v) for v in rounds.values()) * 100
        print(f"== {tag} (spread {spread:.1f}%)", flush=True)
        for k in names:
            change = (med[k] - med["eager"]) / med["eager"] * 100
            rows.append([tag, k, f"{med[k]:.1f}", f"{change:.1f}"])
            print(f"   {k:10s} {med[k]:9.1f} us  vs eager {change:+6.1f}%", flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else "bench_layer.csv"
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["shape", "variant", "us", "vs_eager_pct"]); wr.writerows(rows)
print("saved", out)
