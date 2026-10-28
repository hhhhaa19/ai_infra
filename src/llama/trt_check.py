# 用 TensorRT Python 接口加载 engine：① 和 PyTorch 输出逐元素比对；② 用和 bench_layer.py 相同的方法计时
#   （CUDA event 包住 N 次 execute_async_v3 ÷ N，30 样本 × 3 轮取中位数），也测一遍录成 CUDA Graph 的版本。
# 用法（在仓库根目录）：python src/llama/trt_check.py <engine 前缀，如 build-onnx/layer_eager/eager> [数据目录]
#   会依次加载 <前缀>_<形状>_fp32.engine，参考数据来自 export_onnx.py 保存的 io_<形状>.pt。
#   engine 里有 plugin 时，用 TRT_PLUGINS=a.so[:b.so] 指定 plugin 库，反序列化前先加载（加载时自动登记到注册表）。
import ctypes, os, statistics, sys, time
import torch
import tensorrt as trt

prefix = sys.argv[1]
data_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(prefix)
precision = os.environ.get("TRT_PRECISION", "fp32")
SHAPES = {"decode_b1": 50, "decode_b8": 50, "prefill_s512": 10}
for lib in filter(None, os.environ.get("TRT_PLUGINS", "").split(":")):
    ctypes.CDLL(lib)
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

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

for tag, iters in SHAPES.items():
    path = f"{prefix}_{tag}_{precision}.engine"
    with open(path, "rb") as fh:
        engine = runtime.deserialize_cuda_engine(fh.read())
    ctx = engine.create_execution_context()
    io = torch.load(os.path.join(data_dir, f"io_{tag}.pt"))
    x = io["x"].cuda().contiguous()
    out = torch.empty_like(io["out"]).cuda()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for n in names:
        is_in = engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        t = x if is_in else out
        assert tuple(engine.get_tensor_shape(n)) == tuple(t.shape), (n, engine.get_tensor_shape(n), t.shape)
        ctx.set_tensor_address(n, t.data_ptr())
    stream = torch.cuda.current_stream()
    run = lambda: ctx.execute_async_v3(stream.cuda_stream)
    run(); torch.cuda.synchronize()
    ref = io["out"].cuda()
    err = (out - ref).abs().max().item()
    rel = ((out - ref).abs() / (ref.abs() + 1e-3)).max().item()

    # 录成 CUDA Graph（同一 stream 上 capture execute_async_v3）
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    s.wait_stream(stream)
    with torch.cuda.stream(s):
        ctx.execute_async_v3(s.cuda_stream)            # 先跑一次，让 TensorRT 完成首次运行的准备
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            ctx.execute_async_v3(s.cuda_stream)
    stream.wait_stream(s)
    g.replay(); torch.cuda.synchronize()
    err_cg = (out - ref).abs().max().item()

    fns = {"trt": run, "trt+cg": g.replay}
    until = time.time() + 2.0
    while time.time() < until:
        for f in fns.values():
            for _ in range(5): f()
        torch.cuda.synchronize()
    rounds = {k: [] for k in fns}
    for r in range(3):
        for k in (["trt", "trt+cg"] if r % 2 == 0 else ["trt+cg", "trt"]):
            rounds[k].append(time_us(fns[k], iters))
    med = {k: statistics.median(v) for k, v in rounds.items()}
    print(f"{tag:13s} max_abs_err={err:.2e} max_rel_err={rel:.2e} (cg {err_cg:.2e}) | "
          f"trt {med['trt']:9.1f} us  trt+cg {med['trt+cg']:9.1f} us", flush=True)
