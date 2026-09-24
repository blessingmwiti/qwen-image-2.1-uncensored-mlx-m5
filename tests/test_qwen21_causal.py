"""Causal row-select + masked block/tail equivalence vs torch (REAL weights, fp32)."""

import mlx.core as mx
import numpy as np
import torch

from qwen_mlx.qwen21_block import DiTBlock
from qwen_mlx.qwen21_units import select_rows

MASK11 = [False] * 7 + [True] * 4  # prefix 7, target 4 (toy S=11 layout)


def test_select_rows():
    from diffusers.models.transformers.transformer_qwenimage21 import _select_modulation_rows

    torch.manual_seed(0)
    p = torch.randn(2, 8)  # B+1 = 2 rows
    m = torch.tensor(MASK11)
    ref = _select_modulation_rows(p, m)
    got = select_rows(mx.array(p.numpy()), mx.array(np.array(MASK11)))
    mx.eval(got)
    assert torch.allclose(torch.from_numpy(np.array(got, dtype=np.float32)), ref, atol=1e-6, rtol=1e-6)
    # mask None path
    ref0 = _select_modulation_rows(p[:1], None)
    got0 = select_rows(mx.array(p[:1].numpy()), None)
    mx.eval(got0)
    assert torch.allclose(torch.from_numpy(np.array(got0, dtype=np.float32)), ref0, atol=1e-6, rtol=1e-6)


def _block0():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    p = hf_hub_download(
        "Qwen/Qwen-Image-2.1",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        revision="790c92633540aa0cb11d9abf19eb46d861714758",
    )
    sd = load_file(p, device="cpu")
    return {k[len("transformer_blocks.0."):]: v.float() for k, v in sd.items() if k.startswith("transformer_blocks.0.")}


def test_block_causal_mask():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

    torch.manual_seed(3)
    w = _block0()
    tb = QwenImage21TransformerBlock(dim=4096, num_attention_heads=32, attention_head_dim=128).float().eval()
    tb.load_state_dict(w, strict=True)
    h = torch.randn(1, 11, 4096) * 0.5
    modulation = torch.randn(2, 4 * 4096) * 0.2  # B+1 rows (causal t=0 row)
    tmask = torch.tensor(MASK11)
    with torch.no_grad():
        ref = tb(h, modulation, rotary_emb=None, attention_mask=None, target_token_mask=tmask)

    mb = DiTBlock({k: mx.array(v.numpy()) for k, v in w.items()})
    got = mb(
        mx.array(h.numpy()), mx.array(modulation.numpy()),
        rotary_freqs=None, segments=None, target_token_mask=mx.array(np.array(MASK11)),
    )
    mx.eval(got)
    g = torch.from_numpy(np.array(got, dtype=np.float32))
    diff = (g - ref).abs()
    print("causalblock max:", diff.max().item(), "mean:", diff.mean().item())
    assert torch.allclose(g, ref, atol=0.5, rtol=5e-3)
    assert diff.mean().item() < 0.1


def test_tail_causal_mask():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AdaLayerNormContinuous
    from qwen_mlx.qwen21_units import adaln_scale

    torch.manual_seed(4)
    mod = QwenImage21AdaLayerNormContinuous(embedding_dim=64, conditioning_embedding_dim=32).float()
    x = torch.randn(1, 11, 64) * 0.5
    c = torch.randn(2, 32) * 0.3
    tmask = torch.tensor(MASK11)
    with torch.no_grad():
        ref = mod(x, c, tmask)
    got = adaln_scale(mx.array(x.numpy()), mx.array(c.numpy()), mx.array(mod.linear.weight.detach().numpy()), mask=mx.array(np.array(MASK11)))
    mx.eval(got)
    assert torch.allclose(torch.from_numpy(np.array(got, dtype=np.float32)), ref, atol=1e-3, rtol=1e-3)
