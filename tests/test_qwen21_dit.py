"""Full-DiT prefill equivalence (32 blocks, streaming weights, REAL BF16->fp32).

Compares per-block isolated error AND chained compounding + tail.
Layout: text(8) + cond 4x4(16) + target 4x4(16) = S=40, segments=[(0,8,True),(8,24,False)].
"""

import gc

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from qwen_mlx.qwen21_block import DiTBlock
from qwen_mlx.qwen21_dit import DiTModel
from qwen_mlx.qwen21_rope import build_rope_freqs, build_tables
from qwen_mlx.qwen21_units import adaln_scale

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
SHAPES = [[(1, 4, 4), (1, 4, 4)]]
SEGS = [(0, 8, True), (8, 24, False)]
S = 40

GLOBAL_KEYS = [
    "img_in.weight",
    "modulation.1.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "txt_in.in_layer.weight",
    "txt_in.out_layer.weight",
    "txt_in.text_norm.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
]
BLOCK_SUFFIXES = [
    "attn.norm_k.weight",
    "attn.norm_q.weight",
    "attn.to_k.weight",
    "attn.to_out.0.weight",
    "attn.to_q.weight",
    "attn.to_v.weight",
    "img_mlp.gate_layer.weight",
    "img_mlp.out.weight",
    "img_mlp.proj.weight",
]


def test_full_dit_prefill_streaming():
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21Rope,
        QwenImage21TransformerBlock,
    )
    from huggingface_hub import hf_hub_download

    f1 = hf_hub_download(REPO, "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", revision=REVISION)
    f2 = hf_hub_download(REPO, "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", revision=REVISION)

    def get_torch(name: str) -> torch.Tensor:
        for path in (f1, f2):
            with safe_open(path, framework="pt") as f:
                if name in f.keys():
                    return f.get_slice(name)[:].float()
        raise KeyError(name)

    torch.manual_seed(21)
    # globals (torch, fp32): temb/modulation + tail
    g = {k: get_torch(k) for k in GLOBAL_KEYS}
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TimestepProjEmbeddings

    tproj = QwenImage21TimestepProjEmbeddings(embedding_dim=4096).float()
    tproj.timestep_embedder.linear_1.weight.data = g["time_text_embed.timestep_embedder.linear_1.weight"]
    tproj.timestep_embedder.linear_2.weight.data = g["time_text_embed.timestep_embedder.linear_2.weight"]
    t = torch.tensor([0.5])
    h_dtype = torch.float32
    with torch.no_grad():
        temb = tproj(t, torch.zeros(1, 1, 4096))
        modulation = torch.nn.functional.silu(temb) @ g["modulation.1.weight"].T

    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    mask1d = torch.tensor([False] * 8 + [True] * 32)
    with torch.no_grad():
        r_freqs = rope(SHAPES[0], mask1d, torch.device("cpu"))

    h0 = torch.randn(1, S, 4096) * 0.5
    tables = build_tables()
    m_mask = mx.array([[False] * 8 + [True] * 32])
    m_freqs = build_rope_freqs(SHAPES, m_mask, tables)
    mg = {k: mx.array(v.numpy()) for k, v in g.items()}
    m_t = mx.array(np.array([0.5], dtype=np.float32))

    # reference modulation check (MLX temb path vs torch)
    from qwen_mlx.qwen21_dit import DiTModel

    dit0 = DiTModel(mg, [])
    m_temb, m_mod = dit0.time_embed(m_t)
    mx.eval(m_temb, m_mod)
    dmod = (torch.from_numpy(np.array(m_mod, dtype=np.float32)) - modulation).abs()
    print("modulation max diff:", dmod.max().item())
    assert dmod.max().item() < 0.05

    # stream blocks
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock as TB

    t_h, m_h = h0, mx.array(h0.numpy())
    per_block_max, chained = [], []
    for i in range(32):
        wb = {s: get_torch(f"transformer_blocks.{i}.{s}") for s in BLOCK_SUFFIXES}
        tb = TB(dim=4096, num_attention_heads=32, attention_head_dim=128).float().eval()
        tb.load_state_dict(wb, strict=True)
        with torch.no_grad():
            t_out = tb(t_h, modulation, rotary_emb=r_freqs, attention_mask=None, segments=SEGS)
        mb = DiTBlock({s: mx.array(wb[s].numpy()) for s in BLOCK_SUFFIXES})
        # isolated: MLX block on torch trajectory input
        iso = mb(mx.array(t_h.numpy()), mx.array(modulation.numpy()), rotary_freqs=m_freqs, segments=SEGS)
        mx.eval(iso)
        iso_t = torch.from_numpy(np.array(iso, dtype=np.float32))
        per_block_max.append((iso_t - t_out).abs().max().item())
        # chained: MLX trajectory
        m_h = mb(m_h, mx.array(modulation.numpy()), rotary_freqs=m_freqs, segments=SEGS)
        mx.eval(m_h)
        chained.append((torch.from_numpy(np.array(m_h, dtype=np.float32)) - t_out).abs().max().item())
        t_h = t_out
        del tb, mb, wb
        gc.collect()

    print("per-block isolated max:", round(max(per_block_max), 3))
    print("chained final max:", round(chained[-1], 3))

    # tail (torch)
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AdaLayerNormContinuous

    norm_out = QwenImage21AdaLayerNormContinuous(4096, 4096).float().eval()
    norm_out.linear.weight.data = g["norm_out.linear.weight"]
    proj = torch.nn.Linear(4096, 64, bias=False).float().eval()
    proj.weight.data = g["proj_out.weight"]
    with torch.no_grad():
        ref_final = proj(norm_out(t_h, temb, None))
    m_tail = adaln_scale(m_h, mx.array(temb.numpy()), mg["norm_out.linear.weight"]) @ mg["proj_out.weight"].T
    mx.eval(m_tail)
    fin = (torch.from_numpy(np.array(m_tail, dtype=np.float32)) - ref_final).abs()
    print("final max:", fin.max().item(), "mean:", fin.mean().item())

    assert max(per_block_max) < 8.0, per_block_max  # ~2x measured 3.92; rel ~5e-3 = platform noise (see plan §Precision)
    assert chained[-1] < 10.0, chained[-1]  # ~2x measured 4.67: residuals+norms contain compounding
    assert fin.max().item() < 0.1 and fin.mean().item() < 0.05  # measured 0.011/0.0014: AdaLN contracts error
