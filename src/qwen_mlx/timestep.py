"""Qwen-Image-2.1 timestep embeddings, MLX-native.

Port of `QwenImage21TemporalTimesteps` + `QwenImage21TimestepProjEmbeddings`.
Sinusoidal (cos first half, sin second) dim 256, max_period 10000, time_factor
1000; then TimestepEmbedding 256 → embed_dim (SiLU MLP, no sample-proj bias).
Pipeline calls with timestep/1000, so time_factor restores seconds→ms scale.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class TemporalTimesteps(nn.Module):
    def __init__(self, timestep_dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
        self._freqs = freqs

    def __call__(self, timestep: mx.array) -> mx.array:
        t = self.time_factor * timestep.astype(mx.float32)
        args = t[:, None] * self._freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.timestep_dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb.astype(timestep.dtype)


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = TemporalTimesteps(timestep_dim=256)
        self.lin1 = nn.Linear(256, embedding_dim, bias=False)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def __call__(self, timestep: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
        p = self.time_proj(timestep).astype(dtype)
        return self.lin2(nn.silu(self.lin1(p)))
