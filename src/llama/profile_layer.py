# RMSNorm 在整层 decoder layer 里占多少 GPU 时间：本脚本只负责跑模型并打 NVTX 标签，统计交给 analyze_nsys.py。
# Orin 的 Jetson 版 torch 2.1 的 torch.profiler 抓不到 GPU kernel，所以用 nsys：
#   nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o <报告目录>/layer python src/llama/profile_layer.py
#   nsys export --type sqlite -o <报告目录>/layer.sqlite <报告目录>/layer.nsys-rep
#   python src/llama/analyze_nsys.py <报告目录>/layer.sqlite
# 标签：外层 "<impl>|<形状>" 标出每组 ITERS 次前向；内层 norm1/attn/norm2/mlp 标出子模块；残差加法不在内层标签里。
import os, sys
import torch
from torch.cuda import nvtx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder_layer import DecoderLayer

SHAPES = {"prefill b1 s512": (1, 512), "decode b1": (1, 1), "decode b8": (8, 1)}
ITERS = 20
PARTS = ("norm1", "attn", "norm2", "mlp")

def label(layer):
    # 不改模型代码：把每个子模块的 forward 包一层 NVTX range
    for name in PARTS:
        mod = getattr(layer, name)
        def wrapped(*args, _f=mod.forward, _name=name):
            nvtx.range_push(_name)
            try:
                return _f(*args)
            finally:
                nvtx.range_pop()
        mod.forward = wrapped

torch.manual_seed(0)
base = DecoderLayer(impl="eager").cuda()
with torch.no_grad():
    base.norm1.weight.normal_(); base.norm2.weight.normal_()
print(f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} | ITERS={ITERS}")
for impl in ("eager", "ours"):
    layer = DecoderLayer(impl=impl).cuda(); layer.load_state_dict(base.state_dict()); label(layer)
    for tag, (b, s) in SHAPES.items():
        x = torch.randn(b, s, 2048, device="cuda")
        with torch.inference_mode():
            for _ in range(10): layer(x)                 # 预热（含扩展第一次加载），不在外层标签里
            torch.cuda.synchronize()
            nvtx.range_push(f"{impl}|{tag}")
            for _ in range(ITERS): layer(x)
            torch.cuda.synchronize()
            nvtx.range_pop()
        print("done", impl, tag, flush=True)
