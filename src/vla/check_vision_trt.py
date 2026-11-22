# 用 TensorRT Python 接口加载引擎 A（SigLIP + connector），和 export_vision.py 存下的 PyTorch 输出逐组比对（只看精度，计时用 trtexec）。
# 写法沿用 src/act/check_trt.py：torch 张量当显存，set_tensor_address + execute_async_v3。
# 用法（在 Orin 仓库根目录，venv ~/venvs/ai_infra）：
#   python src/vla/check_vision_trt.py [engine 目录，默认 build-onnx/vla] [精度…，默认 fp32 fp16]
#   加载 vision_<精度>.engine，参考数据是同目录的 vision_io.pt（pixels [2,3,3,512,512]，img_emb [2,3,64,960]）。
# 指标：max_abs、rel_l2（‖误差‖/‖参考‖，主指标）、cos；输出范围很大（约 ±2000），
#   所以另给"按 token 的 rel_l2 最大值"——看是不是个别 token 坏掉而被整体平均掩盖。
import os, sys
import torch
import tensorrt as trt

d = sys.argv[1] if len(sys.argv) > 1 else "build-onnx/vla"
precisions = sys.argv[2:] or ["fp32", "fp16"]
io = torch.load(os.path.join(d, "vision_io.pt"))
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

for precision in precisions:
    path = os.path.join(d, f"vision_{precision}.engine")
    if not os.path.exists(path):
        print(f"{precision}: 没有 {path}，跳过"); continue
    with open(path, "rb") as fh:
        engine = runtime.deserialize_cuda_engine(fh.read())
    ctx = engine.create_execution_context()
    pixels = torch.empty(io["pixels"].shape[1:], device="cuda")
    img_emb = torch.empty(io["img_emb"].shape[1:], device="cuda")
    bufs = {"pixels": pixels, "img_emb": img_emb}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        assert tuple(engine.get_tensor_shape(n)) == tuple(bufs[n].shape), (n, engine.get_tensor_shape(n))
        ctx.set_tensor_address(n, bufs[n].data_ptr())
    stream = torch.cuda.current_stream()

    for s in range(io["pixels"].shape[0]):
        pixels.copy_(io["pixels"][s])
        ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        out, ref = img_emb.cpu(), io["img_emb"][s]
        diff = out - ref
        rel = (diff.norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        tok_rel = (diff.norm(dim=-1) / ref.norm(dim=-1)).max().item()  # 3×64 个 token 里最差的
        print(f"{precision} sample {s}: max_abs {diff.abs().max():.2e}  rel_l2 {rel:.2e}  "
              f"cos {cos:.6f}  worst-token rel_l2 {tok_rel:.2e}  finite {torch.isfinite(out).all().item()}",
              flush=True)
print("img_emb range:", f"[{io['img_emb'].min():.1f}, {io['img_emb'].max():.1f}]",
      " std:", f"{io['img_emb'].std():.1f}")
