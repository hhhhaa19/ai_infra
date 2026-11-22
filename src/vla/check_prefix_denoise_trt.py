# 用 TensorRT Python 接口加载引擎 B（前缀 → K/V）和引擎 C（一步去噪），和 Quest 上原版 PyTorch 的结果比对（只看精度）。
# 写法同 src/vla/check_vision_trt.py。用法（在 Orin 仓库根目录，venv ~/venvs/ai_infra）：
#   python src/vla/check_prefix_denoise_trt.py [engine 目录，默认 build-onnx/vla] [精度后缀…，默认 fp32 fp16]
#   引擎文件 prefix_<后缀>.engine、denoise_<后缀>.engine；参考数据 prefix_io.pt、denoise_io.pt。
# 三项检查：
#   B   ：K/V cache 和原版比（只比真 token 位置；补齐位置本来就不同，引擎 C 会遮掉）
#   C   ：每一步都喂原版的 x_t 和 K/V，只测 C 自己的 v_t 误差
#   链路：B 的输出 → C × 10（Euler 在这里用 torch 做）→ 动作，和原版 10 步后的动作比，这是端到端误差
import os, sys
import torch
import tensorrt as trt

d = sys.argv[1] if len(sys.argv) > 1 else "build-onnx/vla"
suffixes = sys.argv[2:] or ["fp32", "fp16"]
pio = torch.load(os.path.join(d, "prefix_io.pt"))
dio = torch.load(os.path.join(d, "denoise_io.pt"))
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
stream = torch.cuda.current_stream()


class Engine:
    def __init__(self, path, bufs):
        with open(path, "rb") as fh:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        self.ctx = self.engine.create_execution_context()
        self.bufs = bufs
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            assert tuple(self.engine.get_tensor_shape(n)) == tuple(bufs[n].shape), (n, self.engine.get_tensor_shape(n))
            self.ctx.set_tensor_address(n, bufs[n].data_ptr())

    def __call__(self, **inputs):
        for k, v in inputs.items():
            self.bufs[k].copy_(v)
        self.ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()


def stats(name, out, ref):
    out, ref = out.cpu().float(), ref.float()
    diff = out - ref
    rel = (diff.norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    return (f"{name}: max_abs {diff.abs().max():.2e}  rel_l2 {rel:.2e}  cos {cos:.6f}  "
            f"finite {torch.isfinite(out).all().item()}")


cuda = lambda shape, dtype=torch.float32: torch.empty(shape, dtype=dtype, device="cuda")
for sfx in suffixes:
    pb, db = os.path.join(d, f"prefix_{sfx}.engine"), os.path.join(d, f"denoise_{sfx}.engine")
    if not (os.path.exists(pb) and os.path.exists(db)):
        print(f"{sfx}: 缺引擎文件，跳过"); continue
    kv_shape = tuple(pio["k_cache"].shape[1:])                                 # [16, 5, 241, 64]
    k_buf, v_buf = cuda(kv_shape), cuda(kv_shape)
    lang_mask = cuda((1, 48))
    B = Engine(pb, {"img_emb": cuda(tuple(pio["img_emb"].shape[1:])), "lang_tokens": cuda((1, 48), torch.int32),
                    "lang_mask": lang_mask, "state": cuda((1, 32)), "k_cache": k_buf, "v_cache": v_buf})
    x_buf, v_t = cuda((1, 50, 32)), cuda((1, 50, 32))
    C = Engine(db, {"x_t": x_buf, "t_emb": cuda((1, 720)), "k_cache": k_buf, "v_cache": v_buf,
                    "lang_mask": lang_mask, "v_t": v_t})
    n_steps = int(dio["num_steps"])
    for s in range(pio["k_cache"].shape[0]):
        valid = pio["valid"][s]
        # B
        B(img_emb=pio["img_emb"][s], lang_tokens=pio["lang_tokens"][s:s + 1], lang_mask=pio["lang_mask"][s:s + 1],
          state=pio["state"][s:s + 1])
        print(f"[{sfx}] sample {s}  B  " + stats("k", k_buf[:, :, valid], pio["k_cache"][s][:, :, valid]), flush=True)
        print(f"[{sfx}] sample {s}  B  " + stats("v", v_buf[:, :, valid], pio["v_cache"][s][:, :, valid]), flush=True)
        # 链路：K/V 用引擎 B 刚算出的，C 跑 10 步
        x = dio["noise"][s].cuda()
        for i in range(n_steps):
            C(x_t=x, t_emb=dio["t_emb"][i])
            x = x + (-1.0 / n_steps) * v_t
        print(f"[{sfx}] sample {s}  B→C×{n_steps} " + stats("actions", x, dio["actions"][s])
              + f"  real-6-dims max_abs {(x.cpu() - dio['actions'][s])[..., :6].abs().max():.2e}", flush=True)
        # C 单独：K/V 换成原版的，每步喂原版 x_t
        k_buf.copy_(dio["k_cache"][s]); v_buf.copy_(dio["v_cache"][s])
        outs = []
        for i in range(n_steps):
            C(x_t=dio["x_steps"][s][i], t_emb=dio["t_emb"][i])
            outs.append(v_t.cpu().clone())
        print(f"[{sfx}] sample {s}  C  " + stats("v_t (10 steps)", torch.stack(outs), dio["v_steps"][s]), flush=True)
print("actions range:", f"[{dio['actions'].min():.2f}, {dio['actions'].max():.2f}]",
      " real-6-dims std:", f"{dio['actions'][..., :6].std():.2f}")
