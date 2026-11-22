# SmolVLA 第二块（引擎 B）：VLM 前缀（16 层 Llama）→ 每层的 K/V cache，导出成 ONNX 给 TensorRT 用。
# 用法（在仓库根目录，用 Quest 上的 lerobot 环境；要加载整个模型，走 srun，不在登录节点跑）：
#   HF_HOME=/projects/p53267/ai_infra/hf_cache HF_HUB_OFFLINE=1 \
#   /projects/p53267/ai_infra/conda-envs/lerobot/bin/python src/vla/export_prefix.py [输出目录]
# 输出（默认 build-onnx/vla/，要先有 export_vision.py 的 vision_io.pt，图像特征直接用它的 img_emb）：
#   prefix_fp32.onnx   模型
#   prefix_io.pt       几组输入和原版 PyTorch 算出的 K/V cache
# ONNX 的边界：
#   输入  img_emb     [3, 64, 960]  引擎 A 的输出，原样接过来
#         lang_tokens [1, 48] int32  任务描述的 token（分词在 CPU 上做，按 max_length=48 右侧补齐）
#         lang_mask   [1, 48] float  1 = 真 token，0 = 补齐
#         state       [1, 32]        归一化后的关节状态，补零到 32 维
#   输出  k_cache, v_cache [16, 5, 241, 64]：16 层 × 5 个 KV 头 × 241 个前缀 token × 64，K 已经过 RoPE
#   前缀 241 = 192 图像 + 48 语言 + 1 状态。补齐位置的 K/V 是没用的值，引擎 C 会用 lang_mask 把它们遮掉。
# 和原版不同的地方（都只为能导出 / 让 TensorRT 好处理，下面会和原版逐层对比）：
#   1. apply_rope 原版用下标赋值写结果（导出成 ScatterND），这里改成 cat，数学上完全一样。
#   2. attention mask 和位置编号放进图里算：mask 只取决于 lang_mask；cumsum 用乘上三角全 1 矩阵来做（MatMul，TensorRT 一定支持）。
#   3. 补齐 token 那一行 mask 全是 0，原版 softmax 时整行都是极小值；FP16 下会变成 -inf → NaN 并传到下一层的 V。
#      这里让补齐 token 至少看自己（对角线置 1），只影响补齐位置自己的输出，真 token 的 K/V 不变。
import collections, math, os, sys
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
import lerobot.policies.smolvla.smolvlm_with_expert as swe

REPO = "lerobot/smolvla_base"
OPSET = 16
N_CAM, N_IMG_TOK, N_LANG, N_STATE = 3, 64, 48, 32
PREFIX = N_CAM * N_IMG_TOK + N_LANG + 1          # 241
PROMPTS = ["Pick up the red cube and place it in the bowl.\n", "Open the drawer.\n"]
out_dir = sys.argv[1] if len(sys.argv) > 1 else "build-onnx/vla"


def apply_rope_cat(x, positions, max_wavelength=10_000):
    # 和 swe.apply_rope 一样，只是最后用 cat 拼两半，不用下标赋值
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=x.device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class PrefixForExport(torch.nn.Module):
    def __init__(self, flow):                      # flow = policy.model（VLAFlowMatching）
        super().__init__()
        self.flow = flow
        n_vis = N_CAM * N_IMG_TOK
        # 原版 att_masks = [0]*240 + [1]：前 240 个 token 互相都能看，但看不到最后的状态 token；状态 token 能看所有
        ar = torch.zeros(PREFIX); ar[-1] = 1
        c = torch.cumsum(ar, 0)
        self.register_buffer("att_2d", (c[None, :] <= c[:, None]).float(), persistent=False)
        self.register_buffer("eye", torch.eye(PREFIX), persistent=False)
        self.register_buffer("triu", torch.triu(torch.ones(PREFIX, PREFIX)), persistent=False)   # x @ triu = cumsum(x)
        self.register_buffer("ones_vis", torch.ones(1, n_vis), persistent=False)
        self.register_buffer("ones_state", torch.ones(1, 1), persistent=False)

    def forward(self, img_emb, lang_tokens, lang_mask, state):
        vwe = self.flow.vlm_with_expert
        img = img_emb.reshape(1, N_CAM * N_IMG_TOK, img_emb.shape[-1])
        lang = vwe.embed_language_tokens(lang_tokens) * math.sqrt(img_emb.shape[-1])
        st = self.flow.state_proj(state)[:, None, :]
        embs = torch.cat([img, lang, st], dim=1)                               # [1, 241, 960]
        pad = torch.cat([self.ones_vis, lang_mask, self.ones_state], dim=1)   # [1, 241]
        mask = self.att_2d[None] * pad[:, None, :] * pad[:, :, None]
        mask = torch.maximum(mask, self.eye[None]) > 0.5                       # 补齐 token 看自己（说明见文件头第 3 点）
        pos = pad @ self.triu - 1.0                                            # cumsum(pad) - 1
        _, cache = vwe.forward(attention_mask=mask, position_ids=pos, past_key_values=None,
                               inputs_embeds=[embs, None], use_cache=True)
        k = torch.stack([cache.layers[i].keys[0] for i in range(len(cache.layers))])     # [16, 5, 241, 64]
        v = torch.stack([cache.layers[i].values[0] for i in range(len(cache.layers))])
        return k, v


def reference_kv(flow, pixels, lang_tokens, lang_mask, state):
    # 原版流程：embed_prefix（含 SigLIP）→ make_att_2d_masks → cumsum 位置 → vlm_with_expert.forward 填 cache
    images = [pixels[c:c + 1] * 2.0 - 1.0 for c in range(N_CAM)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAM)]
    embs, pad, att = flow.embed_prefix(images, img_masks, lang_tokens.long(), lang_mask.bool(), state=state)
    att_2d = make_att_2d_masks(pad, att)
    pos = torch.cumsum(pad, dim=1) - 1
    _, cache = flow.vlm_with_expert.forward(attention_mask=att_2d, position_ids=pos, past_key_values=None,
                                            inputs_embeds=[embs, None], use_cache=True)
    k = torch.stack([cache.layers[i].keys[0] for i in range(len(cache.layers))])
    v = torch.stack([cache.layers[i].values[0] for i in range(len(cache.layers))])
    return k, v, pad[0].bool()


def compare(name, a, b, valid):
    # 只比真 token 的位置（补齐位置在两边本来就不同，也用不到）
    a, b = a[:, :, valid], b[:, :, valid]
    d = a - b
    per_layer = (d.flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1))
    print(f"  {name}: max_abs {d.abs().max():.2e}  rel_l2 {(d.norm() / b.norm()).item():.2e}  "
          f"worst-layer rel_l2 {per_layer.max():.2e} (layer {per_layer.argmax().item()})  range [{b.min():.1f}, {b.max():.1f}]",
          flush=True)


def load_policy(fp32=True):
    # checkpoint 里 VLM 文本部分和动作专家是 bf16（config 的 text_config.torch_dtype=bfloat16，transformers 按它建模型），
    # 视觉部分是 FP32。TensorRT 8.5 不支持 bf16、onnxruntime CPU 也没有 bf16 MatMul，所以整体转成 FP32 再导出。
    # bf16 → FP32 是精确的（权重不变），只是计算更精确；原生 bf16 的结果另外留一份当"噪声底线"（见 export_denoise.py）。
    path = snapshot_download(REPO)
    cfg = SmolVLAConfig.from_pretrained(path)
    cfg.load_vlm_weights = False
    cfg.device = "cpu"
    print("config: pad_language_to", cfg.pad_language_to, " tokenizer_max_length", cfg.tokenizer_max_length,
          " attention_mode", cfg.attention_mode, " num_steps", cfg.num_steps, " chunk_size", cfg.chunk_size, flush=True)
    policy = SmolVLAPolicy.from_pretrained(path, config=cfg).eval()
    dtypes = sorted({str(p.dtype) for p in policy.parameters()})
    if fp32:
        policy = policy.float()
    print(f"param dtypes as loaded {dtypes} -> {'float32' if fp32 else 'unchanged'}", flush=True)
    return cfg, policy


def make_inputs(cfg):
    # 两条指令（长度不同，测补齐）+ 随机状态；图像特征直接用 vision_io.pt 里的前两组
    tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    enc = tok(PROMPTS, padding="max_length", padding_side="right", max_length=N_LANG, truncation=True, return_tensors="pt")
    lang_tokens = enc["input_ids"].to(torch.int32)                 # [S, 48]
    lang_mask = enc["attention_mask"].to(torch.float32)            # [S, 48]
    print("lang lengths:", lang_mask.sum(1).tolist(), flush=True)
    torch.manual_seed(0)
    state = torch.zeros(len(PROMPTS), N_STATE)
    state[:, :6] = torch.randn(len(PROMPTS), 6)                    # 真机 6 维，其余补零（和 prepare_state 一样）
    return lang_tokens, lang_mask, state


def main():
    cfg, policy = load_policy()
    flow = policy.model
    vio = torch.load(os.path.join(out_dir, "vision_io.pt"))
    lang_tokens, lang_mask, state = make_inputs(cfg)

    # 1) 原版（未改 apply_rope）算参考 K/V
    refs = []
    with torch.inference_mode():
        for s in range(len(PROMPTS)):
            refs.append(reference_kv(flow, vio["pixels"][s], lang_tokens[s:s + 1], lang_mask[s:s + 1], state[s:s + 1]))
    print(f"reference: k {tuple(refs[0][0].shape)}", flush=True)

    # 2) 换成 cat 版 RoPE，用包装层（输入是引擎 A 的 img_emb）再算一遍，对比真 token 位置
    swe.apply_rope = apply_rope_cat
    model = PrefixForExport(flow).eval()
    inputs = [(vio["img_emb"][s], lang_tokens[s:s + 1], lang_mask[s:s + 1], state[s:s + 1]) for s in range(len(PROMPTS))]
    with torch.inference_mode():
        for s in range(len(PROMPTS)):
            k, v = model(*inputs[s])
            print(f"wrapper vs original, sample {s}:", flush=True)
            compare("k", k, refs[s][0], refs[s][2]); compare("v", v, refs[s][1], refs[s][2])
    torch.save({"img_emb": vio["img_emb"][:len(PROMPTS)], "lang_tokens": lang_tokens, "lang_mask": lang_mask,
                "state": state, "k_cache": torch.stack([r[0] for r in refs]), "v_cache": torch.stack([r[1] for r in refs]),
                "valid": torch.stack([r[2] for r in refs]), "prompts": PROMPTS},
               os.path.join(out_dir, "prefix_io.pt"))

    onnx_path = os.path.join(out_dir, "prefix_fp32.onnx")
    with torch.no_grad():
        torch.onnx.export(model, inputs[0], onnx_path, opset_version=OPSET,
                          input_names=["img_emb", "lang_tokens", "lang_mask", "state"],
                          output_names=["k_cache", "v_cache"], do_constant_folding=True, dynamo=False)
    print(f"ONNX: {onnx_path} {os.path.getsize(onnx_path) / 2**20:.1f} MiB", flush=True)

    import onnx
    import onnxruntime as ort
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    print(f"opset {m.opset_import[0].version}, {len(m.graph.node)} nodes:", ", ".join(f"{k}x{v}" for k, v in ops.most_common()))
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    for s in range(len(PROMPTS)):
        feed = {n: t.numpy() for n, t in zip(["img_emb", "lang_tokens", "lang_mask", "state"], inputs[s])}
        k, v = (torch.from_numpy(o) for o in sess.run(None, feed))
        print(f"onnxruntime vs original, sample {s}:", flush=True)
        compare("k", k, refs[s][0], refs[s][2]); compare("v", v, refs[s][1], refs[s][2])


if __name__ == "__main__":
    main()
