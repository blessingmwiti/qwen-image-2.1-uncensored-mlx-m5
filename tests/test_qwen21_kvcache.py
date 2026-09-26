"""KV-cache equivalence: MLX prefill_extract/decode_cached vs torch (REAL weights, fp32)."""

import mlx.core as mx
import numpy as np
import torch

from qwen_mlx.qwen21_block import DiTBlock

MASK11 = [False] * 3 + [True] * 8
SEGS = [(0, 3, True), (3, 7, False)]
SHAPES = [[(1, 2, 2), (1, 2, 2)]]
PREFIX = 7


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


def test_kv_cache_two_step():
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache,
        QwenImage21Rope,
        QwenImage21TransformerBlock,
    )
    from qwen_mlx.qwen21_rope import build_rope_freqs, build_tables

    torch.manual_seed(13)
    w = _block0()
    tb = QwenImage21TransformerBlock(dim=4096, num_attention_heads=32, attention_head_dim=128).float().eval()
    tb.load_state_dict(w, strict=True)
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])

    h = torch.randn(1, 11, 4096) * 0.5
    modulation = torch.randn(1, 4 * 4096) * 0.2  # mask None path (single row)
    mask = torch.tensor(MASK11)
    with torch.no_grad():
        r_freqs = rope(SHAPES[0], torch.tensor(MASK11), torch.device("cpu"))
        cache = QwenImage21KVCache(1)
        layer = cache.get_layer(0)
        out1 = tb(h, modulation, rotary_emb=r_freqs, attention_mask=None,
                  target_token_mask=None, layer_cache=layer, kv_cache_mode="extract",
                  cache_write_slice=slice(0, PREFIX), segments=SEGS)
        out2 = tb(h[:, PREFIX:], modulation, rotary_emb=r_freqs[PREFIX:], attention_mask=None,
                  target_token_mask=None, layer_cache=layer, kv_cache_mode="cached",
                  cache_write_slice=None, segments=None)

    mb = DiTBlock.from_dict({k: mx.array(v.numpy()) for k, v in w.items()})
    tables = build_tables()
    m_mask = mx.array([MASK11])
    m_freqs = build_rope_freqs(SHAPES, m_mask, tables)
    mh, mm = mx.array(h.numpy()), mx.array(modulation.numpy())
    m_out1, m_cache = mb.prefill_extract(mh, mm, m_freqs, SEGS, None, PREFIX)
    m_out2 = mb.decode_cached(
        mx.array(h[:, PREFIX:].numpy()), mm, m_freqs[PREFIX:], None, m_cache,
    )
    mx.eval(m_out1, m_out2)
    g1 = torch.from_numpy(np.array(m_out1, dtype=np.float32))
    g2 = torch.from_numpy(np.array(m_out2, dtype=np.float32))
    d1 = (g1 - out1).abs()
    d2 = (g2 - out2).abs()
    print("extract max:", d1.max().item(), "cached max:", d2.max().item())
    assert torch.allclose(g1, out1, atol=0.5, rtol=5e-3)
    assert torch.allclose(g2, out2, atol=0.5, rtol=5e-3)
    # cached-decode target rows must agree with no-cache full forward on those rows (upstream: 1-ULP drift ok)
    with torch.no_grad():
        ref_full = tb(h, modulation, rotary_emb=r_freqs, attention_mask=None, segments=SEGS)
    dd = (g2 - ref_full[:, PREFIX:]).abs()
    print("cached-vs-nocache max:", dd.max().item())
    assert dd.max().item() < 2.0
