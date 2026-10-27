# 整层 decoder layer：eager / torch.compile / compile + CUDA Graphs，RMSNorm 分别用 eager 写法和我们的算子。
# 只能在有 Triton 的环境跑（Quest A100）：
#   TORCHINDUCTOR_CACHE_DIR=<报告目录>/inductor_cache python src/llama/compile_layer.py <报告目录>
# 输出：bench.csv（耗时）、kernels.csv（每次前向启动了哪些 kernel，torch.profiler）、graph_breaks.txt（Dynamo 是否切断计算图）。
# 计时与 bench_layer.py 相同：CUDA event 包住 N 次前向（prefill 10、decode 50）÷ N，30 样本 × 3 轮轮换顺序，取中位数。
import collections, csv, json, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder_layer import DecoderLayer

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
torch._dynamo.config.cache_size_limit = 64          # 每个形状单独编译（dynamic=False）
SHAPES = {"prefill b1 s512": (1, 512, 10), "decode b1": (1, 1, 50), "decode b8": (8, 1, 50)}

torch.manual_seed(0)
base = DecoderLayer(impl="eager").cuda()
with torch.no_grad():
    base.norm1.weight.normal_(); base.norm2.weight.normal_()

def make(impl):
    layer = DecoderLayer(impl=impl).cuda(); layer.load_state_dict(base.state_dict()); return layer

# 每种写法用自己的一份层（同一组权重），避免不同 compile 模式共享缓存
variants = {
    "eager":           make("eager"),
    "eager+ours":      make("ours"),
    "compile":         torch.compile(make("eager"), dynamic=False),
    "compile+ours":    torch.compile(make("ours"), dynamic=False),
    "compile_cg":      torch.compile(make("eager"), dynamic=False, mode="reduce-overhead"),
    "compile_cg+ours": torch.compile(make("ours"), dynamic=False, mode="reduce-overhead"),
}

def time_us(f, x, iters, samples=30):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(samples):
        start.record()
        for _ in range(iters):
            f(x)
        stop.record(); stop.synchronize()
        out.append(start.elapsed_time(stop) * 1000 / iters)
    return statistics.median(out)

def kernels_of(f, x):
    # 一次前向里启动的所有 kernel（CUDA Graphs 重放时 profiler 也能看到图里的 kernel）
    path = os.path.join(out_dir, "trace_tmp.json")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        f(x); torch.cuda.synchronize()
    p.export_chrome_trace(path)
    data = json.load(open(path)); os.remove(path)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("cat") == "kernel"]

prop = torch.cuda.get_device_properties(0)
print(prop.name, "| SMs", prop.multi_processor_count, "| torch", torch.__version__, flush=True)

# 1. 计算图是否被切断：只看 RMSNorm 两种写法在 prefill 形状下
with open(os.path.join(out_dir, "graph_breaks.txt"), "w") as fh:
    for impl in ("eager", "ours"):
        exp = torch._dynamo.explain(make(impl))(torch.randn(1, 512, 2048, device="cuda"))
        line = f"impl={impl}: graphs={exp.graph_count} graph_breaks={exp.graph_break_count} ops={exp.op_count}"
        print(line, flush=True); fh.write(line + "\n")
        for r in exp.break_reasons:
            fh.write(f"    break: {r.reason}\n")

bench, kernel_rows = [], []
with torch.inference_mode():
    for tag, (b, s, iters) in SHAPES.items():
        x = torch.randn(b, s, 2048, device="cuda")
        ref = variants["eager"](x).clone()
        t0 = time.time()
        for k, f in variants.items():                # 前几次调用触发编译 / CUDA Graph 录制，同时检查结果
            for _ in range(3): y = f(x)
            err = (y - ref).abs().max().item()
            assert err < 1e-3, (k, tag, err)
        print(f"{tag}: compiled + checked in {time.time() - t0:.1f}s", flush=True)
        for k, f in variants.items():
            ks = kernels_of(f, x)
            names = collections.Counter(e["name"][:90] for e in ks)
            kernel_rows.append([tag, k, len(ks), f"{sum(e.get('dur', 0) for e in ks):.1f}",
                                " | ".join(f"{c}x {n}" for n, c in names.most_common())])
        until = time.time() + 2.0
        while time.time() < until:
            for f in variants.values():
                for _ in range(5): f(x)
            torch.cuda.synchronize()
        names = list(variants)
        rounds = {k: [] for k in names}
        for r in range(3):
            order = names[2 * r:] + names[:2 * r]     # 每轮换起点
            for k in order:
                rounds[k].append(time_us(variants[k], x, iters))
        med = {k: statistics.median(v) for k, v in rounds.items()}
        spread = max((max(v) - min(v)) / min(v) for v in rounds.values()) * 100
        for k in names:
            nk = next(r[2] for r in kernel_rows if r[0] == tag and r[1] == k)
            bench.append([tag, k, f"{med[k]:.1f}", f"{(med[k] / med['eager'] - 1) * 100:.1f}", nk])
        print(f"== {tag} (spread {spread:.1f}%)", flush=True)
        for k in names:
            nk = next(r[2] for r in kernel_rows if r[0] == tag and r[1] == k)
            print(f"   {k:16s} {med[k]:9.1f} us  vs eager {(med[k] / med['eager'] - 1) * 100:+6.1f}%  kernels={nk}", flush=True)

with open(os.path.join(out_dir, "bench.csv"), "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["shape", "variant", "us", "vs_eager_pct", "kernels_per_forward"]); wr.writerows(bench)
with open(os.path.join(out_dir, "kernels.csv"), "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["shape", "variant", "kernel_count", "kernel_time_us", "kernels"]); wr.writerows(kernel_rows)
print("saved", out_dir)
