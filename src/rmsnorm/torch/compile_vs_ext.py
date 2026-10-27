# eager vs torch.compile（Inductor 生成 Triton）vs 我们的扩展，在 Quest A100 上比较。
# 用法：TORCHINDUCTOR_CACHE_DIR=<报告目录>/inductor_cache python src/rmsnorm/torch/compile_vs_ext.py <报告目录>
# 输出：kernels.csv（每种实现启动了哪些 kernel、grid/block/寄存器，来自 torch.profiler）、
#       bench.csv（耗时与有效带宽）；Inductor 生成的 Triton 源码留在 inductor_cache 里。
# 计时方法与 bench_ext.py 相同：CUDA event 包住 50 次调用，30 个样本 × 3 轮交替，取中位数。
import csv, json, os, statistics, sys, time
import torch
from build_ext import load_rmsnorm

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
torch._dynamo.config.cache_size_limit = 64   # 每个形状单独编译（dynamic=False），避免超过默认重编译上限
rms_norm = load_rmsnorm()
eps = 1e-5

def eager(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

impls = {
    "eager": eager,
    "compile": torch.compile(eager, dynamic=False),
    "compile_cg": torch.compile(eager, dynamic=False, mode="reduce-overhead"),   # 再加 CUDA Graphs
    "ours": lambda x, w: rms_norm(x, w, eps),
}

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

def kernels_of(f, x, w, tag):
    # 用 profiler 抓一次调用里的所有 kernel；chrome trace 里带 grid/block/寄存器数
    path = os.path.join(out_dir, f"trace_{tag}.json")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        f(x, w); torch.cuda.synchronize()
    p.export_chrome_trace(path)
    events = json.load(open(path))["traceEvents"]
    os.remove(path)
    return [e for e in events if e.get("cat") == "kernel"]

prop = torch.cuda.get_device_properties(0)
print(prop.name, "| SMs", prop.multi_processor_count, "| torch", torch.__version__, flush=True)
rows_list = [512, 4096]
n_list = [128, 1024, 4096, 8192]
bench, kernel_rows = [], []
for rows in rows_list:
    for n in n_list:
        x = torch.randn(rows, n, device="cuda"); w = torch.randn(n, device="cuda")
        ref = eager(x.double(), w.double()).float()
        t0 = time.time()
        for k, f in impls.items():                    # 第一次调用触发编译，同时检查结果
            for _ in range(3): y = f(x, w)            # reduce-overhead 前几次是录制 CUDA Graph
            err = (y - ref).abs().max().item()
            assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), (k, rows, n, err)
        print(f"rows={rows} n={n}: compiled + checked in {time.time() - t0:.1f}s", flush=True)
        for k, f in impls.items():
            for e in kernels_of(f, x, w, k):
                a = e.get("args", {})
                kernel_rows.append([rows, n, k, e["name"][:120], a.get("grid"), a.get("block"),
                                    a.get("registers per thread"), a.get("shared memory"), e.get("dur")])
        until = time.time() + 1.0
        while time.time() < until:
            for f in impls.values():
                for _ in range(20): f(x, w)
            torch.cuda.synchronize()
        names = list(impls)
        rounds = {k: [] for k in names}
        for r in range(3):
            for k in names[r:] + names[:r]:
                rounds[k].append(time_us(impls[k], x, w))
        med = {k: statistics.median(v) for k, v in rounds.items()}
        gbps = {k: (2 * rows * n + n) * 4 / (v * 1e-6) / 1e9 for k, v in med.items()}
        for k in names:
            bench.append([rows, n, k, f"{med[k]:.2f}", f"{gbps[k]:.0f}"])
        print(f"rows={rows:5d} n={n:5d} " + " ".join(f"{k}={med[k]:7.2f}us({gbps[k]:.0f}GB/s)" for k in names), flush=True)

with open(os.path.join(out_dir, "bench.csv"), "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["rows", "n", "impl", "us", "effective_GBps"]); wr.writerows(bench)
with open(os.path.join(out_dir, "kernels.csv"), "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["rows", "n", "impl", "kernel", "grid", "block", "regs", "smem", "dur_us"])
    wr.writerows(kernel_rows)
print("saved", out_dir)
