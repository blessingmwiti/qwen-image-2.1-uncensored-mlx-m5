"""Single-block equivalence: MLX DiTBlock vs torch QwenImage21TransformerBlock (REAL weights, fp32)."""

import mlx.core as mx
import numpy as np
import torch

from qwen_mlx.qwen21_block import DiTBlock

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
BLOCK = "transformer_blocks.0."


def load_block0_fp32():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    p = hf_hub_download(REPO, "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", revision=REVISION)
    sd = load_file(p, device="cpu")
    out = {}
    for k, v in sd.items():
        if k.startswith(BLOCK):
            out[k[len(BLOCK):]] = v.float()
    assert len(out) == 9, len(out)
    return out


def test_block_dense_no_rope():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

    torch.manual_seed(7)
    w = load_block0_fp32()
    tblock = QwenImage21TransformerBlock(dim=4096, num_attention_heads=32, attention_head_dim=128).float().eval()
    tblock.load_state_dict({k: v for k, v in w.items()}, strict=True)

    B, S, D = 1, 16, 4096
    # Realistic scales: latent-derived h ~ O(1), modulation rows ~ O(0.1) (SiLU+Linear outputs, not N(0,1))
    h = torch.randn(B, S, D) * 0.5
    modulation = torch.randn(B, 4 * D) * 0.2  # target_token_mask=None path
    with torch.no_grad():
        ref = tblock(h, modulation, rotary_emb=None, attention_mask=None)

    mw = {k: mx.array(v.numpy()) for k, v in w.items()}
    mblock = DiTBlock.from_dict(mw)
    got = mblock(mx.array(h.numpy()), mx.array(modulation.numpy()))
    mx.eval(got)
    g = torch.from_numpy(np.array(got, dtype=np.float32))
    diff = (g - ref).abs()
    print("block max abs diff:", diff.max().item(), "mean:", diff.mean().item(), "ref max:", ref.abs().max().item())
    # Platform-calibrated tolerance (see docs/mlx_port_plan.md §Precision):
    # MLX Metal fp32 matmuls carry ~1e-3 relative error vs torch CPU (K-independent;
    # torch CPU-vs-MPS control agrees to 4e-3 max, so this is an MLX-Metal property, not a math bug).
    # Bulk must match closely; tails bounded relative to local magnitude.
    assert torch.allclose(g, ref, atol=0.5, rtol=5e-3)
    assert diff.mean().item() < 0.1
