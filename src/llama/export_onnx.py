# 把 decoder layer 导出成 ONNX，给 TensorRT 用（Orin，torch 2.1 的 TorchScript 追踪导出）。
# 用法（在仓库根目录）：python src/llama/export_onnx.py [impl] [输出目录]（需要 onnx 包：torch 2.1 写文件时要用）
#   impl 默认 eager；输出默认 build-onnx/layer_<impl>/（之后用 scripts/build_trt_engines.sh 建 engine），每种形状一个固定形状的 .onnx，
#   同时保存输入和 PyTorch 输出（.pt）作为 TensorRT 正确性检查的参考。
# 权重和 bench_layer.py / compile_layer.py 一样：seed 0，两个 norm 的 weight 用 normal_()。
import collections, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder_layer import DecoderLayer
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

impl = sys.argv[1] if len(sys.argv) > 1 else "eager"
out_dir = sys.argv[2] if len(sys.argv) > 2 else f"build-onnx/layer_{impl}"
os.makedirs(out_dir, exist_ok=True)
SHAPES = {"decode_b1": (1, 1), "decode_b8": (8, 1), "prefill_s512": (1, 512)}
OPSET = 17   # TensorRT 8.5 的 ONNX 解析器比较保守，不用 torch 允许的最高版本 18

# 导出器遇到我们的算子 ai_infra::rms_norm 时，调用这个函数把它翻译成 ONNX 节点 ai_infra::RmsNorm。
# TensorRT 解析时不认识这个节点，就按节点名 RmsNorm 去 plugin 注册表里找，节点属性（eps）原样交给 plugin。
# parse_args：每个参数怎么解析，"v" 图里的张量，"f" 常量 float，"i" 常量 int。impl 只给测试用，不导出。
@parse_args("v", "v", "f", "i")
def rms_norm_symbolic(g, x, weight, eps, impl):
    out = g.op("ai_infra::RmsNorm", x, weight, eps_f=eps)   # 属性写成 名字_类型，_f 表示 float
    out.setType(x.type())                                   # 输出的类型和形状同 x
    return out

register_custom_op_symbolic("ai_infra::rms_norm", rms_norm_symbolic, OPSET)

torch.manual_seed(0)
base = DecoderLayer(impl="eager").cuda()
with torch.no_grad():
    base.norm1.weight.normal_(); base.norm2.weight.normal_()
layer = DecoderLayer(impl=impl).cuda().eval()
layer.load_state_dict(base.state_dict())

for tag, (b, s) in SHAPES.items():
    x = torch.randn(b, s, 2048, device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    with torch.no_grad():
        ref = layer(x)
        path = os.path.join(out_dir, f"layer_{impl}_{tag}.onnx")
        torch.onnx.export(layer, (x,), path, opset_version=OPSET, input_names=["x"], output_names=["out"],
                          do_constant_folding=True, custom_opsets={"ai_infra": 1})
    torch.save({"x": x.cpu(), "out": ref.cpu()}, os.path.join(out_dir, f"io_{tag}.pt"))
    print(f"{tag}: {path} {os.path.getsize(path) / 2**20:.1f} MiB", flush=True)

# 数一下图里每种 ONNX 算子出现几次，并按顺序列出节点（权重不读进来）
import onnx
m = onnx.load(os.path.join(out_dir, f"layer_{impl}_decode_b1.onnx"), load_external_data=False)
onnx.checker.check_model(m)
ops = collections.Counter(n.op_type for n in m.graph.node)
print(f"opset {m.opset_import[0].version}, {len(m.graph.node)} nodes:", ", ".join(f"{k}x{v}" for k, v in ops.most_common()))
with open(os.path.join(out_dir, f"graph_{impl}.txt"), "w") as fh:
    for n in m.graph.node:
        name = f"{n.domain}::{n.op_type}" if n.domain else n.op_type
        attrs = " ".join(f"{a.name}={onnx.helper.get_attribute_value(a)}" for a in n.attribute if a.type == onnx.AttributeProto.FLOAT)
        fh.write(f"{name:18s} {list(n.input)} -> {list(n.output)} {attrs}\n")
