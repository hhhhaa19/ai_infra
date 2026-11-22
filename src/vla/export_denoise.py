# SmolVLA 第三块（引擎 C）：动作专家（action expert，16 层）走一步去噪，导出成 ONNX 给 TensorRT 用。
# 用法（在仓库根目录，用 Quest 上的 lerobot 环境；要加载整个模型，走 srun，不在登录节点跑）：
#   HF_HOME=/projects/p53267/ai_infra/hf_cache HF_HUB_OFFLINE=1 \
#   /projects/p53267/ai_infra/conda-envs/lerobot/bin/python src/vla/export_denoise.py [输出目录]
# 要先有 export_vision.py 的 vision_io.pt；输入构造、RoPE 替换沿用 export_prefix.py。
# 输出（默认 build-onnx/vla/）：
#   denoise_fp32.onnx  模型（一步去噪；10 步循环和 Euler 更新 x ← x + dt·v 在 C++ 里做）
#   denoise_io.pt      原版 PyTorch 完整采样的中间量：每一步的 x_t、v_t，10 步后的动作，时间编码表，K/V cache
# ONNX 的边界：
#   输入  x_t      [1, 50, 32]   当前带噪动作（50 步 × 32 维，真机只用前 6 维）
#         t_emb    [1, 720]      时间步的正余弦编码；10 个时间步固定（1.0, 0.9, …, 0.1），整张表离线算好，C++ 按步取一行
#         k_cache, v_cache [16, 5, 241, 64]  引擎 B 的输出
#         lang_mask [1, 48]      和引擎 B 同一个，用来遮掉前缀里的补齐 token
#   输出  v_t      [1, 50, 32]   速度场（流匹配预测的方向）
#   中间：动作和时间编码 → MLP 融合成 50 个 token（720 维）→ 16 层专家：
#         偶数层 self-attn：专家的 Q 看 [前缀 241 个 K/V ; 自己 50 个 K/V]，动作 token 之间是因果的
#         奇数层 cross-attn：专家的 Q 只看前缀；前缀 K/V 先过专家自己的 k_proj/v_proj（320→320）
#   → final norm → action_out_proj（720→32）
# 和原版的区别：只有 RoPE 换成 cat 版（同 export_prefix.py），K/V cache 由输入张量现建；下面逐步和原版对比。
# 还没做的优化：奇数层对前缀 K/V 的投影每步都一样，可以挪到引擎 B 只算一次（约省 C 每步 7% 的计算量），
#   先导出和原版一一对应的版本当基线，之后再改、再量。
import collections, os
import torch
import torch.nn.functional as F
import lerobot.policies.smolvla.smolvlm_with_expert as swe
from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from export_prefix import (OPSET, N_CAM, N_IMG_TOK, PREFIX, PROMPTS, out_dir,
                           apply_rope_cat, load_policy, make_inputs, PrefixForExport)

CHUNK = 50


class DenoiseForExport(torch.nn.Module):
    def __init__(self, flow):                      # flow = policy.model（VLAFlowMatching）
        super().__init__()
        self.flow = flow
        self.register_buffer("tril", torch.tril(torch.ones(CHUNK, CHUNK)), persistent=False)   # 动作 token 之间因果
        self.register_buffer("ones_vis", torch.ones(1, N_CAM * N_IMG_TOK), persistent=False)
        self.register_buffer("ones_state", torch.ones(1, 1), persistent=False)
        self.register_buffer("arange", torch.arange(CHUNK, dtype=torch.float32)[None], persistent=False)

    def forward(self, x_t, t_emb, k_cache, v_cache, lang_mask):
        f = self.flow
        vwe = f.vlm_with_expert
        # embed_suffix：动作投影 + 时间编码拼起来过 MLP（SiLU）
        a = f.action_in_proj(x_t)                                              # [1, 50, 720]
        h = torch.cat([a, t_emb[:, None, :].expand_as(a)], dim=2)
        h = f.action_time_mlp_out(F.silu(f.action_time_mlp_in(h)))
        # mask [1, 50, 291]：前缀部分按 pad 遮，动作部分下三角
        pad = torch.cat([self.ones_vis, lang_mask, self.ones_state], dim=1)   # [1, 241]
        mask = torch.cat([pad[:, None, :].expand(1, CHUNK, PREFIX), self.tril[None]], dim=2) > 0.5
        # 位置：原版 = 前缀真 token 数 + cumsum(全 1) - 1 = n + i
        pos = pad.sum(dim=1, keepdim=True) + self.arange
        cache = swe.DynamicCache()
        for i in range(k_cache.shape[0]):
            cache.update(k_cache[i:i + 1], v_cache[i:i + 1], i)
        out, _ = vwe.forward(attention_mask=mask, position_ids=pos, past_key_values=cache,
                             inputs_embeds=[None, h], use_cache=True)
        return f.action_out_proj(out[1][:, -CHUNK:].to(torch.float32))


def reference_sampling(flow, cfg, pixels, lang_tokens, lang_mask, state, noise):
    # 原版：embed_prefix → 填 cache → 10 步 denoise_step + Euler（照抄 euler_integrate 的步长和时间）
    images = [pixels[c:c + 1] * 2.0 - 1.0 for c in range(N_CAM)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAM)]
    lt, lm = lang_tokens.long(), lang_mask.bool()
    embs, pad, att = flow.embed_prefix(images, img_masks, lt, lm, state=state)
    att_2d = make_att_2d_masks(pad, att)
    pos = torch.cumsum(pad, dim=1) - 1
    _, cache = flow.vlm_with_expert.forward(attention_mask=att_2d, position_ids=pos, past_key_values=None,
                                            inputs_embeds=[embs, None], use_cache=True)
    k = torch.stack([cache.layers[i].keys[0] for i in range(len(cache.layers))])
    v = torch.stack([cache.layers[i].values[0] for i in range(len(cache.layers))])
    dt = -1.0 / cfg.num_steps
    x = noise
    xs, vs, tembs = [], [], []
    for step in range(cfg.num_steps):
        t = torch.tensor(1.0 + step * dt, dtype=torch.float32).expand(1)
        tembs.append(create_sinusoidal_pos_embedding(t, flow.vlm_with_expert.expert_hidden_size, cfg.min_period,
                                                     cfg.max_period, device=torch.device("cpu")).to(torch.float32))
        v_t = flow.denoise_step(prefix_pad_masks=pad, past_key_values=cache, x_t=x, timestep=t)
        xs.append(x); vs.append(v_t)
        x = x + dt * v_t
    # 自己展开的循环要和原版 sample_actions 一致
    full = flow.sample_actions(images, img_masks, lt, lm, state, noise=noise)
    print(f"  manual loop vs sample_actions: max_abs {(full - x).abs().max():.2e}", flush=True)
    return dict(k=k, v=v, xs=torch.stack(xs), vs=torch.stack(vs), t_emb=torch.stack(tembs), actions=x)


def report(name, a, b):
    d = a - b
    print(f"  {name}: max_abs {d.abs().max():.2e}  rel_l2 {(d.norm() / b.norm()).item():.2e}  "
          f"real-6-dims max_abs {d[..., :6].abs().max():.2e}  range [{b.min():.2f}, {b.max():.2f}]", flush=True)


def main():
    cfg, policy = load_policy(fp32=False)
    flow = policy.model
    vio = torch.load(os.path.join(out_dir, "vision_io.pt"))
    lang_tokens, lang_mask, state = make_inputs(cfg)
    S = len(PROMPTS)
    torch.manual_seed(1)
    noise = torch.randn(S, 1, CHUNK, cfg.max_action_dim)

    # 0) 原生 bf16（checkpoint 本来的精度）跑一遍，只留动作：它和 FP32 的差距就是"模型自己的数值噪声"，
    #    之后 TensorRT FP16 的端到端误差如果比它小，就说明 FP16 没有带来超出模型本身精度的损失。
    with torch.inference_mode():
        actions_bf16 = torch.stack([
            reference_sampling(flow, cfg, vio["pixels"][s], lang_tokens[s:s + 1], lang_mask[s:s + 1],
                               state[s:s + 1], noise[s])["actions"].float() for s in range(S)])
    policy = policy.float()
    flow = policy.model
    print("model -> float32", flush=True)

    # 1) 原版（未改 apply_rope）完整采样，留下每一步
    refs = []
    with torch.inference_mode():
        for s in range(S):
            print(f"reference sample {s}:", flush=True)
            refs.append(reference_sampling(flow, cfg, vio["pixels"][s], lang_tokens[s:s + 1], lang_mask[s:s + 1],
                                           state[s:s + 1], noise[s]))
    t_emb = refs[0]["t_emb"]                                                   # [10, 1, 720]，和输入无关
    for s in range(S):
        print(f"native bf16 vs fp32 actions, sample {s} (noise floor):", flush=True)
        report("actions", actions_bf16[s], refs[s]["actions"])

    # 2) 换 RoPE，用包装层逐步对比（输入用原版的 x_t 和 K/V，只测引擎 C 自己）
    swe.apply_rope = apply_rope_cat
    model = DenoiseForExport(flow).eval()
    prefix = PrefixForExport(flow).eval()
    with torch.inference_mode():
        for s in range(S):
            r = refs[s]
            v_w = torch.stack([model(r["xs"][i], t_emb[i], r["k"], r["v"], lang_mask[s:s + 1])
                               for i in range(cfg.num_steps)])
            print(f"wrapper vs original, sample {s} (v_t, all 10 steps):", flush=True)
            report("v_t", v_w, r["vs"])
            # 3) 整条链：B 包装层 → C 包装层 × 10 → 动作，和原版 10 步后的动作比
            k, v = prefix(vio["img_emb"][s], lang_tokens[s:s + 1], lang_mask[s:s + 1], state[s:s + 1])
            x = noise[s]
            for i in range(cfg.num_steps):
                x = x + (-1.0 / cfg.num_steps) * model(x, t_emb[i], k, v, lang_mask[s:s + 1])
            print(f"chain B→C×10 (wrappers) vs original actions, sample {s}:", flush=True)
            report("actions", x, r["actions"])
    torch.save({"noise": noise, "t_emb": t_emb, "lang_mask": lang_mask,
                "k_cache": torch.stack([r["k"] for r in refs]), "v_cache": torch.stack([r["v"] for r in refs]),
                "x_steps": torch.stack([r["xs"] for r in refs]), "v_steps": torch.stack([r["vs"] for r in refs]),
                "actions": torch.stack([r["actions"] for r in refs]), "actions_bf16": actions_bf16,
                "num_steps": cfg.num_steps},
               os.path.join(out_dir, "denoise_io.pt"))

    r = refs[0]
    example = (r["xs"][0], t_emb[0], r["k"], r["v"], lang_mask[0:1])
    names = ["x_t", "t_emb", "k_cache", "v_cache", "lang_mask"]
    onnx_path = os.path.join(out_dir, "denoise_fp32.onnx")
    with torch.no_grad():
        torch.onnx.export(model, example, onnx_path, opset_version=OPSET, input_names=names,
                          output_names=["v_t"], do_constant_folding=True, dynamo=False)
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
    for s in range(S):
        r = refs[s]
        outs = []
        for i in range(cfg.num_steps):
            feed = dict(zip(names, (t.numpy() for t in (r["xs"][i], t_emb[i], r["k"], r["v"], lang_mask[s:s + 1]))))
            outs.append(torch.from_numpy(sess.run(None, feed)[0]))
        print(f"onnxruntime vs original, sample {s} (v_t, all 10 steps):", flush=True)
        report("v_t", torch.stack(outs), r["vs"])


if __name__ == "__main__":
    main()
