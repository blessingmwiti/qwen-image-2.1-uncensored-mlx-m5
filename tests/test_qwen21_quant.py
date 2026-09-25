"""Quantization feasibility: Q8/Q4 MLX DiT block vs fp32 MLX and torch (REAL weights)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from qwen_mlx.qwen21_block import DiTBlock


def _block0_fp32():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    p = hf_hub_download(
        "Qwen/Qwen-Image-2.1",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        revision="790c92633540aa0cb11d9abf19eb46d861714758",
    )
    sd = load_file(p, device="cpu")
    return {k[len("transformer_blocks.0."):]: v.float() for k, v in sd.items() if k.startswith("transformer_blocks.0.")}


def _sizes(m: nn.Module):
    nfp32 = sum(mod.weight.size for _, mod in m.named_modules() if isinstance(mod, nn.Linear)) * 4
    return nfp32 / 1e9


def test_quant_block():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

    torch.manual_seed(7)
    w = _block0_fp32()
    tb = QwenImage21TransformerBlock(dim=4096, num_attention_heads=32, attention_head_dim=128).float().eval()
    tb.load_state_dict(w, strict=True)
    h = torch.randn(1, 16, 4096) * 0.5
    modulation = torch.randn(1, 4 * 4096) * 0.2
    with torch.no_grad():
        ref = tb(h, modulation, rotary_emb=None, attention_mask=None)

    mh, mm = mx.array(h.numpy()), mx.array(modulation.numpy())
    out = {}
    fp = DiTBlock.from_dict({k: mx.array(v.numpy()) for k, v in w.items()})
    fgb = _sizes(fp)
    got = fp(mh, mm)
    mx.eval(got)
    out["fp32"] = torch.from_numpy(np.array(got, dtype=np.float32))
    print(f"fp32 GB: {fgb:.3f}")

    for bits in (8, 4):
        q = DiTBlock.from_dict({k: mx.array(v.numpy()) for k, v in w.items()})
        nn.quantize(q, group_size=64, bits=bits)
        nq = fgb * bits / 32 * 1.05  # packed weights + fp32 scales/biases overhead ~5%
        gotq = q(mh, mm)
        mx.eval(gotq)
        out[bits] = torch.from_numpy(np.array(gotq, dtype=np.float32))
        print(f"Q{bits} GB: {nq:.3f}")

    import json

    rep = {"fp32_gb": round(fgb, 3)}
    for key in ("fp32", 8, 4):
        d = (out[key] - ref).abs()
        rep[str(key)] = {"max": round(d.max().item(), 4), "mean": round(d.mean().item(), 6)}
        print(key, rep[str(key)])
    p = __import__("pathlib").Path("experiments/exp-002/metrics/quant_block0.json")
    p.write_text(json.dumps(rep, indent=2))

    assert rep["fp32"]["mean"] < 0.1
    assert rep["8"]["mean"] < 0.5, rep["8"]  # Q8 must stay close
    # Q4 recorded only (no assert: fidelity gate for the matrix)

    # full-DiT size projection (32 blocks, linears only; DiT BF16 file total 14.23GB incl. norms/temb)
    print(f"32-block proj: fp32 {fgb*32:.2f}GB, Q8 ~{fgb*32/4*1.05:.2f}GB, Q4 ~{fgb*32/8*1.05:.2f}GB")
