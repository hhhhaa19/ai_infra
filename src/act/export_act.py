# 把 LeRobot 官方 ACT（ALOHA 仿真里双臂传递方块）导出成 ONNX，给 TensorRT 用。
# 用法（在仓库根目录，用 Quest 上的 lerobot 环境；登录节点 CPU 即可）：
#   HF_HOME=/projects/p53267/ai_infra/hf_cache TORCH_HOME=/projects/p53267/ai_infra/torch_cache \
#   /projects/p53267/ai_infra/conda-envs/lerobot/bin/python src/act/export_act.py [输出目录]
# 输出（默认 build-onnx/act/）：
#   act_aloha_fp32.onnx  模型
#   io_samples.pt        几组输入和 PyTorch 输出，之后 TensorRT 用同样的输入对答案
# ONNX 的边界：输入原始图像（[0,1]，[1,3,480,640]）和原始 14 维状态，输出反归一化后的动作 [1,100,14]。
# 新版 LeRobot 加载这个旧 checkpoint 时不会用里面的归一化参数，这里直接从 model.safetensors 取出来放进 ONNX，
# 部署程序只要喂原始数据，不会漏掉归一化。
import collections, os, sys
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from lerobot.policies.act.modeling_act import ACTPolicy

REPO = "lerobot/act_aloha_sim_transfer_cube_human"
OPSET = 16          # Orin 的 TensorRT 8.5 不认识 opset 17 的 LayerNormalization 节点；用 16，LayerNorm 拆成基础算子
N_SAMPLES = 4       # 存几组输入输出
out_dir = sys.argv[1] if len(sys.argv) > 1 else "build-onnx/act"
os.makedirs(out_dir, exist_ok=True)

# checkpoint 里归一化参数的名字 → 这里用的名字
STATS = {
    "normalize_inputs.buffer_observation_images_top.mean": "img_mean",    # [3,1,1]
    "normalize_inputs.buffer_observation_images_top.std": "img_std",
    "normalize_inputs.buffer_observation_state.mean": "state_mean",       # [14]
    "normalize_inputs.buffer_observation_state.std": "state_std",
    "unnormalize_outputs.buffer_action.mean": "act_mean",                 # [14]
    "unnormalize_outputs.buffer_action.std": "act_std",
}


class ActForExport(torch.nn.Module):
    """输入归一化 → ACT 网络（推理时潜变量 z = 0，不走 VAE encoder）→ 动作反归一化。"""

    def __init__(self, policy, stats):
        super().__init__()
        self.model = policy.model
        for name, t in stats.items():
            self.register_buffer(name, t.float())

    def forward(self, image, state):
        image = (image - self.img_mean) / self.img_std
        state = (state - self.state_mean) / self.state_std
        actions = self.model({"observation.state": state, "observation.images": [image]})[0]
        return actions * self.act_std + self.act_mean


path = snapshot_download(REPO)
with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
    stats = {name: f.get_tensor(key) for key, name in STATS.items()}
policy = ACTPolicy.from_pretrained(path).eval()
model = ActForExport(policy, stats).eval()

# 输入：图像是 [0,1] 的随机数，状态在训练数据的均值附近随机取（真实数据之后用数据集替换）
torch.manual_seed(0)
images = torch.rand(N_SAMPLES, 1, 3, 480, 640)
states = stats["state_mean"] + stats["state_std"] * torch.randn(N_SAMPLES, 1, 14)
with torch.inference_mode():
    refs = torch.stack([model(images[i], states[i]) for i in range(N_SAMPLES)])
torch.save({"image": images, "state": states, "action": refs}, os.path.join(out_dir, "io_samples.pt"))
print(f"PyTorch: action {tuple(refs.shape[1:])}, range [{refs.min():.3f}, {refs.max():.3f}]", flush=True)

onnx_path = os.path.join(out_dir, "act_aloha_fp32.onnx")
with torch.no_grad():
    torch.onnx.export(model, (images[0], states[0]), onnx_path, opset_version=OPSET,
                      input_names=["image", "state"], output_names=["action"],
                      do_constant_folding=True, dynamo=False)
print(f"ONNX: {onnx_path} {os.path.getsize(onnx_path) / 2**20:.1f} MiB", flush=True)

# 检查：ONNX 格式、算子统计，用 ONNX Runtime（CPU）跑同样的输入和 PyTorch 对比
import onnx
import onnxruntime as ort
m = onnx.load(onnx_path)
onnx.checker.check_model(m)
ops = collections.Counter(n.op_type for n in m.graph.node)
print(f"opset {m.opset_import[0].version}, {len(m.graph.node)} nodes:", ", ".join(f"{k}x{v}" for k, v in ops.most_common()))
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
for i in range(N_SAMPLES):
    out = sess.run(None, {"image": images[i].numpy(), "state": states[i].numpy()})[0]
    d = torch.from_numpy(out) - refs[i]
    rel = (d.norm() / refs[i].norm()).item()
    print(f"sample {i}: max_abs_err {d.abs().max():.2e}  rel_l2 {rel:.2e}")
