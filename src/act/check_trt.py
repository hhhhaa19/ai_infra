# 用 TensorRT Python 接口加载 ACT 的 engine，和 export_act.py 存下的 PyTorch 输出逐组比对（只看精度，计时用 trtexec）。
# 写法沿用 src/llama/trt_check.py：torch 张量当显存，set_tensor_address + execute_async_v3。
# 用法（在 Orin 仓库根目录，venv ~/venvs/ai_infra）：
#   python src/act/check_trt.py [engine 目录，默认 build-onnx/act]
#   依次加载 act_aloha_fp32.engine、act_aloha_fp16.engine，参考数据是同目录的 io_samples.pt。
# 指标：max_abs（最大绝对误差）、rel_l2（‖误差‖/‖参考‖，主指标）、cos（方向是否一致）；
#   另外按 14 个关节分别给出最大绝对误差。动作已反归一化，单位和数据集一致（关节角为弧度）。
import os, sys
import torch
import tensorrt as trt

d = sys.argv[1] if len(sys.argv) > 1 else "build-onnx/act"
io = torch.load(os.path.join(d, "io_samples.pt"))
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

for precision in ("fp32", "fp16"):
    with open(os.path.join(d, f"act_aloha_{precision}.engine"), "rb") as fh:
        engine = runtime.deserialize_cuda_engine(fh.read())
    ctx = engine.create_execution_context()
    image = torch.empty(io["image"].shape[1:], device="cuda")
    state = torch.empty(io["state"].shape[1:], device="cuda")
    action = torch.empty(io["action"].shape[1:], device="cuda")
    bufs = {"image": image, "state": state, "action": action}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        assert tuple(engine.get_tensor_shape(n)) == tuple(bufs[n].shape), (n, engine.get_tensor_shape(n))
        ctx.set_tensor_address(n, bufs[n].data_ptr())
    stream = torch.cuda.current_stream()

    per_joint = torch.zeros(io["action"].shape[-1])
    for s in range(io["image"].shape[0]):
        image.copy_(io["image"][s]); state.copy_(io["state"][s])
        ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        out, ref = action.cpu(), io["action"][s]
        diff = out - ref
        rel = (diff.norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        per_joint = torch.maximum(per_joint, diff.abs().reshape(-1, diff.shape[-1]).max(0).values)
        print(f"{precision} sample {s}: max_abs {diff.abs().max():.2e}  rel_l2 {rel:.2e}  cos {cos:.6f}", flush=True)
    print(f"{precision} per-joint max_abs:", " ".join(f"{v:.1e}" for v in per_joint.tolist()), flush=True)
print("action range:", f"[{io['action'].min():.3f}, {io['action'].max():.3f}]",
      " per-joint std:", " ".join(f"{v:.2f}" for v in io["action"].reshape(-1, io["action"].shape[-1]).std(0).tolist()))
