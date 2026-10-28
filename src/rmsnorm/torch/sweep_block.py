# n = 2048 时，rmsNormReg / fusedAddRmsNormReg（每次读 1 个 float）和 float4 版 rmsNormVec4 / fusedAddRmsNormVec4
# 换不同的 BLOCK × PER_THREAD 切法，和 Triton（torch.compile）比单个 kernel 的 GPU 时间。
# 用法（Quest，有 Triton）：python src/rmsnorm/torch/sweep_block.py <输出目录>
# 计时：同一个 kernel 连续 100 次录进一个 CUDA Graph，重放 20 次取中位数 ÷ 100，不含 CPU 发射开销。
# 注意：连续重复读同一块数据，数据在 L2 里；我们和 Triton 条件相同。
import os, sys, csv, statistics, subprocess, pathlib
import torch
from torch.utils.cpp_extension import load_inline

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
kernel_file = pathlib.Path(__file__).resolve().parents[1] / "kernels.cu"

cuda_src = f'#include "{kernel_file}"\n' + r"""
#include <ATen/cuda/CUDAContext.h>
// 只给 n = 2048 用：BLOCK × P = 2048
void run(torch::Tensor x, torch::Tensor r, torch::Tensor w, torch::Tensor h, torch::Tensor y,
         int64_t block, bool fused, bool vec4, double eps) {
    TORCH_CHECK(x.size(-1) == 2048, "sweep only supports n = 2048");
    const int n = 2048, rows = x.numel() / n;
    const float e = static_cast<float>(eps);
    auto s = at::cuda::getCurrentCUDAStream();
    const float *xp = x.data_ptr<float>(), *rp = r.data_ptr<float>(), *wp = w.data_ptr<float>();
    float *hp = h.data_ptr<float>(), *yp = y.data_ptr<float>();
#define CASE(B, P) case B: \
        if (fused) fusedAddRmsNormReg<B, P><<<rows, B, (B / 32) * sizeof(float), s>>>(xp, rp, wp, hp, yp, n, e); \
        else       rmsNormReg<B, P><<<rows, B, (B / 32) * sizeof(float), s>>>(xp, wp, yp, n, e); \
        break;
// float4 版：V = P / 4 个 float4
#define VCASE(B, V) case B: \
        if (fused) fusedAddRmsNormVec4<B, V><<<rows, B, (B / 32) * sizeof(float), s>>>(xp, rp, wp, hp, yp, n, e); \
        else       rmsNormVec4<B, V><<<rows, B, (B / 32) * sizeof(float), s>>>(xp, wp, yp, n, e); \
        break;
    if (vec4) switch (block) { VCASE(128, 4) VCASE(256, 2) VCASE(512, 1) default: TORCH_CHECK(false, "bad block"); }
    else      switch (block) { CASE(128, 16) CASE(256, 8) CASE(512, 4) CASE(1024, 2) default: TORCH_CHECK(false, "bad block"); }
#undef CASE
#undef VCASE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
ext = load_inline(name="sweep_block_ext", cpp_sources="void run(torch::Tensor x, torch::Tensor r, torch::Tensor w, torch::Tensor h, torch::Tensor y, int64_t block, bool fused, bool vec4, double eps);",
                  cuda_sources=cuda_src, functions=["run"], extra_cuda_cflags=["-O2"], verbose=False)

def clocks():
    q = "clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()

def time_graph(fn, iters=100, reps=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()                      # 预热（Triton 在这里完成编译）
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return statistics.median(ts)

def norm_ref(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

def add_norm_ref(x, r, w, eps=1e-5):
    h = x + r
    return h, norm_ref(h, w, eps)

prop = torch.cuda.get_device_properties(0)
print(prop.name, "| SMs", prop.multi_processor_count, "| torch", torch.__version__, flush=True)
print("clocks at start (sm, max sm, mem, temp, power):", clocks(), flush=True)
_a = torch.randn(4096, 4096, device="cuda")
for _ in range(200): _a @ _a                       # 先让 GPU 忙一会儿，把频率拉满再计时
torch.cuda.synchronize(); del _a
print("clocks after warm-up:", clocks(), flush=True)

torch.manual_seed(0)
n, eps = 2048, 1e-5
rows_list = [1, 8, 64, 512]
blocks = [128, 256, 512, 1024]
vec4_blocks = [128, 256, 512]          # n=2048 是 512 个 float4，1024 线程会有一半空闲，不测
results = []
for rows in rows_list:
    x = torch.randn(rows, n, device="cuda"); r = torch.randn_like(x); w = torch.randn(n, device="cuda")
    h = torch.empty_like(x); y = torch.empty_like(x)
    h_ref, y_ref = add_norm_ref(x, r, w, eps)
    row = {}
    for fused in (False, True):
        kind = "add+norm" if fused else "norm"
        for vec4, bs in ((False, blocks), (True, vec4_blocks)):
            for b in bs:
                ext.run(x, r, w, h, y, b, fused, vec4, eps); torch.cuda.synchronize()
                ref = y_ref if fused else norm_ref(x, w, eps)
                assert (y - ref).abs().max().item() < 1e-4, (kind, rows, b, vec4)
                if fused: assert torch.equal(h, h_ref)
                t = time_graph(lambda: ext.run(x, r, w, h, y, b, fused, vec4, eps))
                name = f"{'vec4' if vec4 else 'ours'} {b}x{n // b}"
                results.append([kind, rows, name, f"{t:.2f}"]); row[(kind, vec4, b)] = t
        tri = torch.compile(add_norm_ref if fused else norm_ref, dynamic=False)
        args = (x, r, w, eps) if fused else (x, w, eps)
        got = tri(*args)
        assert ((got[1] if fused else got) - (y_ref if fused else norm_ref(x, w, eps))).abs().max().item() < 1e-4
        t = time_graph(lambda: tri(*args))
        results.append([kind, rows, "triton", f"{t:.2f}"]); row[(kind, "triton")] = t
    for kind in ("norm", "add+norm"):
        print(f"rows={rows:4d} {kind:8s} | scalar: " + " ".join(f"{b}x{n // b}={row[(kind, False, b)]:6.2f}" for b in blocks) +
              " | float4: " + " ".join(f"{b}x{n // b}={row[(kind, True, b)]:6.2f}" for b in vec4_blocks) +
              f" | triton={row[(kind, 'triton')]:6.2f} us", flush=True)

print("clocks at end   (sm, max sm, mem, temp, power):", clocks(), flush=True)
with open(os.path.join(out_dir, "sweep_block.csv"), "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["kind", "rows", "impl", "us_per_kernel"]); wr.writerows(results)
print("saved", out_dir)
