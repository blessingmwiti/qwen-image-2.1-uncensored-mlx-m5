"""Load exported quantized MLX DiT (see scripts/export_qdit.py)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

SUFFIXES = [
    "attn.norm_k.weight", "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight",
    "attn.to_q.weight", "attn.to_v.weight", "img_mlp.gate_layer.weight",
    "img_mlp.out.weight", "img_mlp.proj.weight",
]
GLOBALS = [
    "img_in.weight", "modulation.1.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "txt_in.in_layer.weight", "txt_in.out_layer.weight", "txt_in.text_norm.weight",
    "norm_out.linear.weight", "proj_out.weight",
]
N_LAYERS = 32


def _restore_linear(weight: mx.array, scales: mx.array | None, biases: mx.array | None, bits: int, group_size: int):
    if scales is None:
        lin = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        lin.weight = weight
        return lin
    lin = nn.QuantizedLinear(weight.shape[1], weight.shape[0], bias=False, group_size=group_size, bits=bits)
    lin.weight = weight
    lin.scales = scales
    lin.biases = biases
    return lin


def load_qdit(path: str, bits: int = 4, group_size: int = 64):
    """Returns (globals_dict, [DiTBlock x32]) with weights restored from export."""
    from safetensors.numpy import load_file

    from qwen_mlx.qwen21_block import DiTBlock

    raw = load_file(path)
    G = lambda k: mx.array(raw[f"globals.{k}"])
    globals_w = {k: G(k) for k in GLOBALS}
    blocks = []
    for i in range(N_LAYERS):
        b = DiTBlock()
        for attr, sfx in (
            ("to_q", "attn.to_q.weight"), ("to_k", "attn.to_k.weight"), ("to_v", "attn.to_v.weight"),
            ("to_out", "attn.to_out.0.weight"), ("gate", "img_mlp.gate_layer.weight"),
            ("proj", "img_mlp.proj.weight"), ("out", "img_mlp.out.weight"),
        ):
            p = f"blocks.{i}.{attr}"
            w = mx.array(raw[f"{p}.weight"])
            sc = mx.array(raw[f"{p}.scales"]) if f"{p}.scales" in raw else None
            bi = mx.array(raw[f"{p}.biases"]) if f"{p}.biases" in raw else None
            setattr(b, attr, _restore_linear(w, sc, bi, bits, group_size))
        b._norm_q = mx.array(raw[f"blocks.{i}._norm_q"])
        b._norm_k = mx.array(raw[f"blocks.{i}._norm_k"])
        blocks.append(b)
    return globals_w, blocks
