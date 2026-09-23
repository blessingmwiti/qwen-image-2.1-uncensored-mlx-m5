"""Unit equivalence: MLX qwen21_units vs torch diffusers@8b3c707e (float32, fixed seeds)."""

import math

import mlx.core as mx
import numpy as np
import torch

from qwen_mlx.qwen21_units import (
    adaln_scale,
    apply_rotary_complex,
    layer_norm_no_affine,
    rope_freqs,
    swiglu,
    temporal_timesteps,
    zero_center_rmsnorm,
)


def to_mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().numpy())


def to_torch(a: mx.array) -> torch.Tensor:
    mx.eval(a)
    arr = np.array(a, dtype=np.float32) if a.dtype != mx.complex64 else None
    if arr is None:  # complex: stack real/imag
        r = np.array(a.real, dtype=np.float32)
        i = np.array(a.imag, dtype=np.float32)
        return torch.view_as_complex(torch.stack([torch.from_numpy(r), torch.from_numpy(i)], dim=-1))
    return torch.from_numpy(arr)


def close(a_mx: mx.array, b_t: torch.Tensor, tol=1e-5):
    mx.eval(a_mx)
    return torch.allclose(to_torch(a_mx).float(), b_t.float(), atol=tol, rtol=tol)


def test_temporal_timesteps():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TemporalTimesteps

    mod = QwenImage21TemporalTimesteps(timestep_dim=256).float()
    t = torch.tensor([0.04, 0.5, 1.0])
    ref = mod(t)
    got = temporal_timesteps(mx.array(np.array([0.04, 0.5, 1.0], dtype=np.float32)))
    # tol 1e-4: different libm sin/cos on MLX vs torch CPU (measured max diff 5.5e-05)
    assert close(got, ref, tol=1e-4), (to_torch(got).flatten()[:4], ref.flatten()[:4])


def test_zero_center_rmsnorm():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21ZeroCenterRMSNorm

    torch.manual_seed(0)
    mod = QwenImage21ZeroCenterRMSNorm(64).float()
    x = torch.randn(2, 16, 64)
    ref = mod(x)
    got = zero_center_rmsnorm(to_mx(x), to_mx(mod.weight.detach()))
    assert close(got, ref)


def test_swiglu():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21SwiGLUFeedForward

    torch.manual_seed(1)
    mod = QwenImage21SwiGLUFeedForward(hidden_size=32, mlp_hidden_size=96).float()
    x = torch.randn(2, 8, 32)
    ref = mod(x)
    got = swiglu(to_mx(x), to_mx(mod.gate_layer.weight.detach()), to_mx(mod.proj.weight.detach()), to_mx(mod.out.weight.detach()))
    # tol 1e-3: chained Metal matmuls (torch sigmoid itself matches to 1.2e-07, so this is matmul accumulation)
    assert close(got, ref, tol=1e-3)


def test_adaln_scale():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AdaLayerNormContinuous

    torch.manual_seed(2)
    mod = QwenImage21AdaLayerNormContinuous(embedding_dim=32, conditioning_embedding_dim=16).float()
    x = torch.randn(2, 8, 32)
    c = torch.randn(2, 16)
    ref = mod(x, c)
    got = adaln_scale(to_mx(x), to_mx(c), to_mx(mod.linear.weight.detach()))
    # tol 1e-3: Linear matmul on Metal (same as swiglu finding)
    assert close(got, ref, tol=1e-3)


def test_rope_params():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope

    mod = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    idx = torch.arange(10)
    ref = mod.rope_params(idx, 16)  # complex [10, 8]
    got = rope_freqs(mx.array(np.arange(10)), 16)
    g = to_torch(got)
    assert torch.allclose(torch.view_as_real(g).float(), torch.view_as_real(ref).float(), atol=1e-5, rtol=1e-5)


def test_apply_rotary_complex():
    from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

    torch.manual_seed(3)
    x = torch.randn(1, 6, 2, 128)
    ang = torch.randn(6, 64)
    freqs = torch.polar(torch.ones_like(ang), ang)
    ref = apply_rotary_emb_qwen(x, freqs, use_real=False)
    got = apply_rotary_complex(to_mx(x), to_mx(freqs))
    assert close(got, ref, tol=1e-4)


def test_layer_norm_no_affine():
    torch.manual_seed(4)
    x = torch.randn(2, 8, 32)
    ref = torch.nn.functional.layer_norm(x, (32,), eps=1e-6)
    got = layer_norm_no_affine(to_mx(x))
    assert close(got, ref)
