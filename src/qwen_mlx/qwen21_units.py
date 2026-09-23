"""MLX reference units for Qwen-Image-2.1 DiT (port plan step 1).

Mirrors `diffusers@8b3c707e`
`models/transformers/transformer_qwenimage21.py` math in native MLX.
No NumPy round-trips; torch appears ONLY in tests as the oracle.
"""

from __future__ import annotations

import math

import mlx.core as mx


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def temporal_timesteps(
    timestep: mx.array,
    dim: int = 256,
    max_period: int = 10000,
    time_factor: float = 1000.0,
) -> mx.array:
    """Sinusoidal embedding, cos first half then sin (matches QwenImage21TemporalTimesteps)."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = (time_factor * timestep.astype(mx.float32))[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb.astype(timestep.dtype)


def zero_center_rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """RMSNorm with zero-centered stored weight: scale = weight + 1, computed in fp32."""
    dtype = x.dtype
    xf = x.astype(mx.float32)
    rrms = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * rrms * (weight.astype(mx.float32) + 1)).astype(dtype)


def swiglu(x: mx.array, gate_w: mx.array, proj_w: mx.array, out_w: mx.array) -> mx.array:
    """SwiGLU FFN, no biases: out(silu(x @ gate) * (x @ proj)). Weights [out, in] (torch/MLX Linear layout)."""
    return (_silu(x @ gate_w.T) * (x @ proj_w.T)) @ out_w.T


def layer_norm_no_affine(x: mx.array, eps: float = 1e-6) -> mx.array:
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) / mx.sqrt(var + eps)


def adaln_scale(x: mx.array, cond: mx.array, linear_w: mx.array, eps: float = 1e-6) -> mx.array:
    """AdaLayerNormContinuous (scale-only): norm(x) * (1 + Linear(silu(cond))).

    Mirrors `_select_modulation_rows(..., mask=None)` = unsqueeze(1): scale broadcasts over tokens.
    """
    scale = _silu(cond).astype(x.dtype) @ linear_w.T
    return layer_norm_no_affine(x, eps) * (1 + mx.expand_dims(scale, 1))


def rope_freqs(index: mx.array, dim: int, theta: int = 10000) -> mx.array:
    """Complex RoPE freqs: polar(1, outer(index, 1/theta^(arange(0,dim,2)/dim))). Returns complex64 [N, dim//2]."""
    idx = index.astype(mx.float32)
    exponents = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    inv = 1.0 / (theta ** exponents)
    angles = idx[:, None] * inv[None, :]
    return (mx.cos(angles) + 1j * mx.sin(angles)).astype(mx.complex64)


def apply_rotary_complex(x: mx.array, freqs: mx.array) -> mx.array:
    """Complex rotary apply (matches DiT usage: use_real=False).
    x [B,S,H,D] with consecutive-element pairs; freqs complex [S,D//2] broadcast over heads.
    """
    dtype = x.dtype
    xf = x.astype(mx.float32)
    a = xf[..., 0::2]
    b = xf[..., 1::2]
    xc = (a + 1j * b).astype(mx.complex64)
    f = freqs[None, :, None, :].astype(mx.complex64)
    out = xc * f
    paired = mx.stack([out.real, out.imag], axis=-1)
    return mx.reshape(paired, xf.shape).astype(dtype)
