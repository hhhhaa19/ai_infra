# 最小的 LLaMA 风格 decoder layer（只做推理前向）。
# RMSNorm 可以切换实现：impl="eager" 用 PyTorch 运算，impl="ours" 用我们的 CUDA 扩展。
# 暂不包含 RoPE 和 KV cache。
import sys, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# 让 Python 找到 src/rmsnorm/torch/build_ext.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "rmsnorm" / "torch"))
from build_ext import load_rmsnorm

_op = None
def rms_norm_op():
    # 第一次用到时才编译/加载扩展，返回 torch.ops.ai_infra.rms_norm；只用 eager 时不需要它
    global _op
    if _op is None:
        _op = load_rmsnorm()
    return _op


class RMSNorm(nn.Module):
    def __init__(self, hidden, eps=1e-5, impl="eager"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden)) # TODO 1: 用 nn.Parameter 创建 weight，形状 [hidden]，初始值全 1（torch.ones）
        self.eps= eps
        self.impl = impl# TODO 2: 把 eps 和 impl 存到 self 上
        if impl == "ours":
            rms_norm_op()

    def forward(self, x):
        # TODO 3: impl == "eager" 时，用 PyTorch 运算写 RMSNorm
        if self.impl == "eager":
            return x * torch.rsqrt(x.pow(2).mean(-1,keepdim =True)+self.eps) * self.weight
        # impl == "ours"：调用注册好的算子 torch.ops.ai_infra.rms_norm
        if self.impl == "ours":
            return rms_norm_op()(x, self.weight, self.eps)
        raise ValueError(f"unknown impl: {self.impl}")
        


class MLP(nn.Module):
    # LLaMA 的 SwiGLU：down( silu(gate(x)) * up(x) )
    # 形状：[.., hidden] → gate/up → [.., intermediate] → down → [.., hidden]
    def __init__(self, hidden, intermediate):
        super().__init__()
        # TODO 1: 三个 nn.Linear，都不要 bias（bias=False）
        self.gate_proj= nn.Linear(hidden,intermediate,bias = False)
        #   self.gate_proj: hidden → intermediate
        self.up_proj = nn.Linear(hidden,intermediate,bias = False)
        #   self.up_proj:   hidden → intermediate
        self.down_proj = nn.Linear(intermediate,hidden,bias = False)
        #   self.down_proj: intermediate → hidden

    def forward(self, x):
        # TODO 2: 用 F.silu 写出 down(silu(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


class Attention(nn.Module):
    # 多头自注意力（不含 RoPE 和 KV cache）
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        # TODO 3: 四个 nn.Linear（bias=False），都是 hidden → hidden：q_proj、k_proj、v_proj、o_proj

    def forward(self, x):
        b, s, h = x.shape                      # batch、seq、hidden
        q = self.q_proj(x).reshape(b, s, self.heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).reshape(b, s, self.heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(b, s, self.heads, self.head_dim).transpose(1,2)
        # TODO 4: q、k、v 分别投影，再 reshape 成 [b, s, heads, head_dim]，然后 transpose(1, 2) 变成 [b, heads, s, head_dim]
        o = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        # TODO 5: o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        

        # TODO 6: o.transpose(1, 2) 变回 [b, s, heads, head_dim]，reshape 成 [b, s, hidden]，最后过 o_proj
        return self.o_proj(o.transpose(1,2).reshape(b,s,h))


class DecoderLayer(nn.Module):
    # x = x + attn(norm1(x))；x = x + mlp(norm2(x))
    # norm1、norm2 是两个位置不同的 RMSNorm（各有自己的 weight），impl 对两者相同
    def __init__(self, hidden=2048, heads=16, intermediate=5632, impl="eager"):
        super().__init__()
        self.norm1 = RMSNorm(hidden, impl=impl)     # attention 前
        self.attn = Attention(hidden, heads)
        self.norm2 = RMSNorm(hidden, impl=impl)     # MLP 前
        self.mlp = MLP(hidden, intermediate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
