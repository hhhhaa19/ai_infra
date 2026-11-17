# SmolVLA 第一块（引擎 A）：视觉编码器 SigLIP + connector，导出成 ONNX 给 TensorRT 用。
# 用法（在仓库根目录，用 Quest 上的 lerobot 环境；要加载整个模型，走 srun，不在登录节点跑）：
#   HF_HOME=/projects/p53267/ai_infra/hf_cache \
#   /projects/p53267/ai_infra/conda-envs/lerobot/bin/python src/vla/export_vision.py [输出目录]
# 输出（默认 build-onnx/vla/）：
#   vision_fp32.onnx   模型
#   vision_io.pt       几组输入和 PyTorch 输出，之后 TensorRT 用同样的输入对答案
# ONNX 的边界：
#   输入  pixels [3, 3, 512, 512]：3 个相机，已经 resize + pad 到 512×512 的 RGB，范围 [0, 1]
#         （resize_with_pad 留在外面，部署时用 CUDA 预处理 kernel 做）
#   输出  img_emb [3, 64, 960]：每个相机 64 个 token，每个 960 维，已经乘过 sqrt(960)，直接拼进 prefix
#   中间做的事：[0,1] → [-1,1]（SigLIP 要求）→ SigLIP 12 层（1024 个 patch）→ connector（pixel shuffle + 线性层，1024 → 64 个 token）→ × sqrt(960)
import collections, math, os, sys
import torch
from huggingface_hub import snapshot_download
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

REPO = "lerobot/smolvla_base"
OPSET = 16          # Orin 的 TensorRT 8.5 不认识 opset 17 的 LayerNormalization；和 ACT 一样用 16
N_CAM = 3
N_SAMPLES = 2
out_dir = sys.argv[1] if len(sys.argv) > 1 else "build-onnx/vla"
os.makedirs(out_dir, exist_ok=True)


class VisionForExport(torch.nn.Module):
    def __init__(self, vlm_with_expert):
        super().__init__()
        self.vwe = vlm_with_expert

    def forward(self, pixels):
        # 和 vlm_with_expert.embed_image 做的事一样，只是拆开调用，不让 transformers 生成 attention mask：
        # 我们的图都是满的 512×512，没有 padding，mask 全是 1，等于不加 mask；
        # 而导出（tracing）时它生成 mask 的代码会报错（形状被当成张量，走进了旧版兼容分支）。
        # 下面会用原版 embed_image 对比，确认结果一样。
        x = pixels * 2.0 - 1.0
        vlm = self.vwe.get_vlm_model()
        vm = vlm.vision_model
        # embeddings 也拆开：原版为了支持任意尺寸的图，用 bucketize + 下标赋值算每个 patch 的位置编号，
        # 导出后变成 NonZero / ScatterND 一类的节点，onnxruntime 都加载不了（ScatterND 类型对不上）。
        # 图是满的 512×512 时，位置编号就是 0, 1, …, 1023，直接加整张位置编码表即可（下面会和原版对比）。
        emb_layer = vm.embeddings
        h = emb_layer.patch_embedding(x)                                    # [B, 768, 32, 32]：16×16 stride 16 的卷积
        h = h.flatten(2).transpose(1, 2)                                    # [B, 1024, 768]
        h = h + emb_layer.position_embedding.weight                         # 第 i 个 patch 加第 i 行位置编码
        out = vm.encoder(inputs_embeds=h, attention_mask=None)              # 12 层 transformer
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h = vm.post_layernorm(h)
        emb = vlm.connector(h)                                              # [B, 64, 960]
        return emb * math.sqrt(emb.shape[-1])


path = snapshot_download(REPO)
cfg = SmolVLAConfig.from_pretrained(path)
cfg.load_vlm_weights = False        # VLM 结构按 config 建，权重全部从 smolvla_base 的 safetensors 读
cfg.device = "cpu"
print("resize_imgs_with_padding:", cfg.resize_imgs_with_padding, flush=True)
policy = SmolVLAPolicy.from_pretrained(path, config=cfg).eval()
model = VisionForExport(policy.model.vlm_with_expert).eval()
print("vision dtype:", next(model.parameters()).dtype, flush=True)

torch.manual_seed(0)
pixels = torch.rand(N_SAMPLES, N_CAM, 3, 512, 512)
with torch.inference_mode():
    refs = torch.stack([model(pixels[i]) for i in range(N_SAMPLES)])
# 拆开调用的结果要和原版 embed_image 一致
with torch.inference_mode():
    x0 = pixels[0] * 2.0 - 1.0
    orig = policy.model.vlm_with_expert.embed_image(x0)
    orig = orig * math.sqrt(orig.shape[-1])
print(f"wrapper vs embed_image: max_abs_diff {(orig - refs[0]).abs().max():.2e}", flush=True)
torch.save({"pixels": pixels, "img_emb": refs}, os.path.join(out_dir, "vision_io.pt"))
print(f"PyTorch: img_emb {tuple(refs.shape[1:])}, range [{refs.min():.3f}, {refs.max():.3f}]", flush=True)

onnx_path = os.path.join(out_dir, "vision_fp32.onnx")
with torch.no_grad():
    torch.onnx.export(model, (pixels[0],), onnx_path, opset_version=OPSET,
                      input_names=["pixels"], output_names=["img_emb"],
                      do_constant_folding=True, dynamo=False)
print(f"ONNX: {onnx_path} {os.path.getsize(onnx_path) / 2**20:.1f} MiB", flush=True)

# 检查：ONNX 格式、算子统计，用 ONNX Runtime（CPU）跑同样的输入和 PyTorch 对比
import onnx
import onnxruntime as ort
m = onnx.load(onnx_path)
onnx.checker.check_model(m)
ops = collections.Counter(n.op_type for n in m.graph.node)
print(f"opset {m.opset_import[0].version}, {len(m.graph.node)} nodes:", ", ".join(f"{k}x{v}" for k, v in ops.most_common()))
so = ort.SessionOptions()
so.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))   # 不指定的话它按整台机器的核数绑线程，srun 里会刷一堆报错
sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
for i in range(N_SAMPLES):
    out = sess.run(None, {"pixels": pixels[i].numpy()})[0]
    d = torch.from_numpy(out) - refs[i]
    rel = (d.norm() / refs[i].norm()).item()
    print(f"sample {i}: max_abs_err {d.abs().max():.2e}  rel_l2 {rel:.2e}")
