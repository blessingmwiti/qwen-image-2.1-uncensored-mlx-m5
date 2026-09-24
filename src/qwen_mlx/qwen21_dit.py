"""MLX full DiT for Qwen-Image-2.1 (prefill, no cache yet).

Mirrors `QwenImage21Transformer2DModel.forward` prefill path (segments form,
target_token_mask=None variant) from `diffusers@8b3c707e`.
"""

from __future__ import annotations

import mlx.core as mx

from qwen_mlx.qwen21_block import DiTBlock
from qwen_mlx.qwen21_units import adaln_scale, modulation_proj, temporal_timesteps, timestep_mlp

DIM = 4096
HEADS = 32
HEAD_DIM = 128
N_LAYERS = 32


class DiTModel:
    """globals: dict of non-block weights; blocks: list of per-block dicts (torch names)."""

    def __init__(self, globals_w: dict[str, mx.array], blocks_w: list[dict[str, mx.array]]):
        self.g = globals_w
        self.blocks = [DiTBlock(w) for w in blocks_w]

    def time_embed(self, timestep: mx.array, dtype=None) -> tuple[mx.array, mx.array]:
        proj = temporal_timesteps(timestep)
        temb = timestep_mlp(proj, self.g["time_text_embed.timestep_embedder.linear_1.weight"], self.g["time_text_embed.timestep_embedder.linear_2.weight"])
        modulation = modulation_proj(temb, self.g["modulation.1.weight"])
        return temb, modulation

    def __call__(
        self,
        h: mx.array,
        timestep: mx.array,
        rotary_freqs: mx.array,
        segments: list[tuple[int, int, bool]],
        target_token_mask: mx.array | None = None,
    ) -> mx.array:
        temb, modulation = self.time_embed(timestep)
        for block in self.blocks:
            h = block(h, modulation, rotary_freqs=rotary_freqs, segments=segments, target_token_mask=target_token_mask)
        h = adaln_scale(h, temb, self.g["norm_out.linear.weight"], mask=target_token_mask)
        return h @ self.g["proj_out.weight"].T
