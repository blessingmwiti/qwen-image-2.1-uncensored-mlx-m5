"""Rope-builder + segmented-prefill equivalence (toy layout, REAL weights, fp32).

Layout B=1: text(3) + cond 2x2(4) + target 2x2(4) = S=11.
segments = [(0,3,True),(3,7,False)], prefix_len=7.
"""

import mlx.core as mx
import numpy as np
import torch

from qwen_mlx.qwen21_block import DiTBlock
from qwen_mlx.qwen21_rope import build_rope_freqs, build_tables

SHAPES = [[(1, 2, 2), (1, 2, 2)]]
SEGS = [(0, 3, True), (3, 7, False)]
MASK = [[False] * 3 + [True] * 8]


def to_torch(a: mx.array) -> torch.Tensor:
    mx.eval(a)
    r = np.array(a.real, dtype=np.float32)
    i = np.array(a.imag, dtype=np.float32)
    return torch.view_as_complex(torch.stack([torch.from_numpy(r), torch.from_numpy(i)], dim=-1))


def test_rope_builder():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope

    mod = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    ref = mod(SHAPES[0], torch.tensor(MASK[0]), torch.device("cpu"))
    got = build_rope_freqs(SHAPES, mx.array(MASK), build_tables())
    g = to_torch(got)
    assert g.shape == ref.shape, (g.shape, ref.shape)
    assert torch.allclose(torch.view_as_real(g).float(), torch.view_as_real(ref).float(), atol=1e-5, rtol=1e-5)


def load_block0():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    p = hf_hub_download(
        "Qwen/Qwen-Image-2.1",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        revision="790c92633540aa0cb11d9abf19eb46d861714758",
    )
    sd = load_file(p, device="cpu")
    return {k[len("transformer_blocks.0."):]: v.float() for k, v in sd.items() if k.startswith("transformer_blocks.0.")}


def test_block_segmented_with_rope():
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21Rope,
        QwenImage21TransformerBlock,
    )

    torch.manual_seed(11)
    w = load_block0()
    tb = QwenImage21TransformerBlock(dim=4096, num_attention_heads=32, attention_head_dim=128).float().eval()
    tb.load_state_dict(w, strict=True)
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])

    h = torch.randn(1, 11, 4096) * 0.5
    modulation = torch.randn(1, 4 * 4096) * 0.2
    with torch.no_grad():
        r_freqs = rope(SHAPES[0], torch.tensor(MASK[0]), torch.device("cpu"))
        ref = tb(h, modulation, rotary_emb=r_freqs, attention_mask=None, segments=SEGS)

    mw = {k: mx.array(v.numpy()) for k, v in w.items()}
    mb = DiTBlock.from_dict(mw)
    m_freqs = build_rope_freqs(SHAPES, mx.array(MASK), build_tables())
    got = mb(mx.array(h.numpy()), mx.array(modulation.numpy()), rotary_freqs=m_freqs, segments=SEGS)
    mx.eval(got)
    g = torch.from_numpy(np.array(got, dtype=np.float32))
    diff = (g - ref).abs()
    print("segblock max:", diff.max().item(), "mean:", diff.mean().item(), "refmax:", ref.abs().max().item())
    assert torch.allclose(g, ref, atol=0.5, rtol=5e-3)
    assert diff.mean().item() < 0.1
