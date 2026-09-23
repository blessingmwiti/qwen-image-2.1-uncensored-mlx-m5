"""Qwen-Image-2.1 norms + modulation, MLX-native.

Port of `QwenImage21ZeroCenterRMSNorm`, `QwenImage21AdaLayerNormContinuous`,
`_select_modulation_rows`, and block `_modulate` semantics.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ZeroCenterRMSNorm(nn.Module):
    """Weight stored zero-centered; effective scale = weight + 1, fp32 compute."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        xf = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (xf * rrms * (self.weight.astype(mx.float32) + 1)).astype(dtype)


def select_modulation_rows(params: mx.array, target_token_mask: mx.array | None) -> mx.array:
    """params [B,D] (or [B+1,D] causal: rows[0,B)=timestep, row[-1]=t=0).
    mask [S] bool True at target-image positions. Returns [B,S,D] (mask None → [B,1,D])."""
    if target_token_mask is None:
        return params[:, None, :]
    real = params[:-1][:, None, :]  # [B,1,D]
    zero = params[-1:][None, :, :]  # [1,1,D]
    m = target_token_mask.reshape(1, -1, 1)
    return mx.where(m, mx.broadcast_to(real, (real.shape[0], m.shape[1], real.shape[2])),
                    mx.broadcast_to(zero, (real.shape[0], m.shape[1], zero.shape[2])))


def modulate(hidden: mx.array, mod_params: mx.array, target_token_mask: mx.array | None):
    """mod_params [B,2D] → (hidden*(1+scale), gate). Mirrors block _modulate."""
    d = mod_params.shape[-1] // 2
    scale, gate = mod_params[..., :d], mod_params[..., d:]
    scale = select_modulation_rows(scale, target_token_mask)
    gate = select_modulation_rows(gate, target_token_mask)
    return hidden * (1 + scale), gate
