"""Parity tests: MLX Qwen21 units vs torch diffusers (fp32, fixed seeds).

Covers rope freqs + rotary apply, timestep embeddings, zero-center norm,
modulation row selection. Run AFTER heavy MPS jobs finish (GPU courtesy).
"""

import numpy as np


def _t2m(t):
    import mlx.core as mx

    return mx.array(t.detach().cpu().float().numpy())


def test_rope_freqs_parity():
    import mlx.core as mx
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope as TorchRope

    from qwen_mlx.rope import QwenImage21Rope as MlxRope

    torch_rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56]).eval()
    mlx_rope = MlxRope(theta=10000, axes_dim=(16, 56, 56))
    # One text block of 5 + image 4x4 + trailing text 3 → total 24
    mask = np.array([False] * 5 + [True] * 16 + [False] * 3)
    import torch

    with torch.no_grad():
        t = torch_rope([(1, 4, 4)], torch.tensor(mask), torch.device("cpu"))
    m = mlx_rope([(1, 4, 4)], mx.array(mask))
    mx.eval(m)
    t_stack = np.stack([t.real.numpy(), t.imag.numpy()], axis=-1)
    np.testing.assert_allclose(np.array(m.tolist()), t_stack, rtol=1e-5, atol=1e-6)


def test_rotary_apply_parity():
    import mlx.core as mx
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

    from qwen_mlx.rope import apply_rotary_complex

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 8, 4, 16)).astype(np.float32)
    c = rng.normal(size=(8, 8)).astype(np.float32)
    s = rng.normal(size=(8, 8)).astype(np.float32)
    with torch.no_grad():
        out_t = apply_rotary_emb_qwen(torch.tensor(x), (torch.tensor(c), torch.tensor(s))).numpy()
    out_m = apply_rotary_complex(mx.array(x), mx.stack([mx.array(c), mx.array(s)], axis=-1))
    mx.eval(out_m)
    np.testing.assert_allclose(np.array(out_m.tolist()), out_t, rtol=1e-5, atol=1e-6)


def test_timestep_parity():
    import mlx.core as mx
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TimestepProjEmbeddings

    from qwen_mlx.timestep import TimestepProjEmbeddings

    torch.manual_seed(0)
    tmod = QwenImage21TimestepProjEmbeddings(embedding_dim=64).eval()
    mmod = TimestepProjEmbeddings(embedding_dim=64)
    mmod.lin1.weight = _t2m(tmod.timestep_embedder.linear_1.weight)
    mmod.lin2.weight = _t2m(tmod.timestep_embedder.linear_2.weight)
    ts = torch.tensor([0.5, 0.02])
    hs = torch.zeros(2, 64)
    with torch.no_grad():
        out_t = tmod(ts, hs).numpy()
    out_m = mmod(mx.array([0.5, 0.02]), dtype=mx.float32)
    mx.eval(out_m)
    np.testing.assert_allclose(np.array(out_m.tolist()), out_t, rtol=1e-4, atol=1e-5)


def test_norm_modulate_parity():
    import mlx.core as mx
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21ZeroCenterRMSNorm,
        _select_modulation_rows,
    )

    from qwen_mlx.norm import ZeroCenterRMSNorm, modulate

    torch.manual_seed(1)
    tn = QwenImage21ZeroCenterRMSNorm(32).eval()
    mn = ZeroCenterRMSNorm(32)
    mn.weight = _t2m(tn.weight)
    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        out_m = mn(mx.array(x.numpy()))
        mx.eval(out_m)
        np.testing.assert_allclose(
            np.array(out_m.tolist()),
            tn(x).numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
    params = torch.randn(3, 16)  # batch 2 + zero row
    tmask = torch.tensor([True, False, True])
    with torch.no_grad():
        out_t = _select_modulation_rows(params, tmask).numpy()
    from qwen_mlx.norm import select_modulation_rows

    out_m = select_modulation_rows(mx.array(params.numpy()), mx.array([True, False, True]))
    mx.eval(out_m)
    np.testing.assert_allclose(np.array(out_m.tolist()), out_t, rtol=1e-6, atol=1e-7)
    # modulate = scale+gate split
    h = mx.array(np.random.default_rng(2).normal(size=(2, 3, 8)).astype(np.float32))
    mp = mx.array(params.numpy()[:, :16])
    mh, mg = modulate(h, mx.concatenate([mp, mp], axis=-1), mx.array([True, False, True]))
    mx.eval(mh, mg)
    assert mh.shape == (2, 3, 8) and mg.shape == (2, 3, 8)
