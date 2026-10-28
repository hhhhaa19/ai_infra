# 整层对照：ours / fused 和 eager 输出一致，fused 在 torch.compile 下不断图（graphs=1 breaks=0）。
# 用法（在仓库根目录）：python src/llama/check_fused_layer.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch._dynamo as dynamo
import decoder_layer as dl
torch.manual_seed(0)
base = dl.DecoderLayer(impl="eager").cuda().eval()
with torch.no_grad():
    base.norm1.weight.normal_(); base.norm2.weight.normal_()
layers = {}
for impl in ("ours", "fused"):
    m = dl.DecoderLayer(impl=impl).cuda().eval(); m.load_state_dict(base.state_dict()); layers[impl] = m
for shape in [(1, 1, 2048), (8, 1, 2048), (1, 512, 2048)]:
    x = torch.randn(*shape, device="cuda")
    with torch.no_grad():
        ref = base(x)
        errs = {k: (m(x) - ref).abs().max().item() for k, m in layers.items()}
        dynamo.reset()
        _, _, graphs, ops, reasons = dynamo.explain(layers["fused"], x)[:5]
    print(f"{shape}: max_err ours={errs['ours']:.1e} fused={errs['fused']:.1e} | "
          f"fused graphs={len(graphs)} breaks={len(reasons)} ops={[len(o) for o in ops]}")
