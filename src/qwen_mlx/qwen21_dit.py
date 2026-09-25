"""MLX full DiT for Qwen-Image-2.1 (prefill, no cache yet).

Mirrors `QwenImage21Transformer2DModel.forward` prefill path (segments form,
target_token_mask=None variant) from `diffusers@8b3c707e`.
"""

from __future__ import annotations

import mlx.core as mx

from qwen_mlx.qwen21_block import DiTBlock
from qwen_mlx.qwen21_units import adaln_scale, modulation_proj, temporal_timesteps, text_projection, timestep_mlp

DIM = 4096
HEADS = 32
HEAD_DIM = 128
N_LAYERS = 32
IMG_TOKENS_PER_SLOT = 4


def build_token_metadata(image_pad_mask: mx.array, img_shapes: list[tuple[int, int, int]]):
    """Mirror QwenImage21Transformer2DModel.build_token_metadata (row-0 layout).
    Returns image_ids [S] int (-1 text, block id per image block), target_token_mask [S] bool."""
    import math as _math

    mask = image_pad_mask.tolist()
    if mask and isinstance(mask[0], list):
        mask = mask[0]
    S = len(mask)
    image_positions = [i for i, m in enumerate(mask) if m]
    block_lengths = [_math.prod(s) for s in img_shapes]
    assert sum(block_lengths) == len(image_positions)
    image_ids = [-1] * S
    idx = 0
    for b, length in enumerate(block_lengths):
        for _ in range(length):
            image_ids[image_positions[idx]] = b
            idx += 1
    target = [False] * S
    for i in image_positions[-block_lengths[-1]:]:
        target[i] = True
    return mx.array(image_ids), mx.array(target)


def prefix_segments(image_ids: mx.array, prefix_len: int) -> list[tuple[int, int, bool]]:
    ids = image_ids.tolist()[:prefix_len]
    segs, start = [], 0
    for i in range(1, prefix_len + 1):
        if i == prefix_len or ids[i] != ids[start]:
            segs.append((start, i, ids[start] < 0))
            start = i
    return segs


class DiTModel:
    """globals: dict of non-block weights; blocks: list of per-block dicts (torch names)."""

    def __init__(self, globals_w: dict[str, mx.array], blocks_w: list[dict[str, mx.array]]):
        self.g = globals_w
        self.blocks = [DiTBlock.from_dict(w) for w in blocks_w]

    def time_embed(self, timestep: mx.array, dtype=None) -> tuple[mx.array, mx.array]:
        proj = temporal_timesteps(timestep)
        temb = timestep_mlp(proj, self.g["time_text_embed.timestep_embedder.linear_1.weight"], self.g["time_text_embed.timestep_embedder.linear_2.weight"])
        modulation = modulation_proj(temb, self.g["modulation.1.weight"])
        return temb, modulation

    def project_inputs(self, latents: mx.array, encoder_embeds: mx.array):
        """latents [B,L,64] packed VAE tokens; encoder_embeds [B,Lp,4096]."""
        h = latents @ self.g["img_in.weight"].T
        e = text_projection(
            encoder_embeds,
            self.g["txt_in.text_norm.weight"],
            self.g["txt_in.in_layer.weight"],
            self.g["txt_in.out_layer.weight"],
        )
        return h, e

    def build_joint(
        self,
        h_lat: mx.array,
        enc: mx.array,
        mask: mx.array,
        img_shapes: list[tuple[int, int, int]],
    ):
        """Mirror torch joint construction exactly (mask = pre-expansion [B, P+slots], True at img slots).
        joint0 = [enc | zeros(B, Tt//4, D)]; repeat_interleave x4 at img slots; scatter packed latents at mask.
        h_lat [B, Llat, D] (img_in-projected packed latents, Llat = #True in expanded mask).
        """
        import numpy as _np

        B = h_lat.shape[0]
        repeats = _np.where(_np.asarray(mask[0]), IMG_TOKENS_PER_SLOT, 1)
        mask_np = _np.asarray(mask[0])
        exp_mask = _np.repeat(mask_np, repeats)
        _, tgt = build_token_metadata(mx.array(exp_mask[None, :]), img_shapes)
        n_target_slots = int(_np.asarray(tgt).sum()) // IMG_TOKENS_PER_SLOT
        enc_np = _np.asarray(enc, dtype=_np.float32)
        joint = _np.concatenate(
            [enc_np, _np.zeros((B, n_target_slots, enc_np.shape[-1]), dtype=_np.float32)], axis=1
        )
        rep = _np.repeat(joint, repeats, axis=1)
        h_np = _np.asarray(h_lat, dtype=_np.float32).reshape(-1, h_lat.shape[-1])
        assert h_np.shape[0] == int(exp_mask.sum()), (h_np.shape, exp_mask.sum())
        rep[:, exp_mask] = h_np
        return mx.array(rep)

    def __call__(
        self,
        h: mx.array,
        timestep: mx.array,
        rotary_freqs: mx.array,
        segments: list[tuple[int, int, bool]],
        target_token_mask: mx.array | None = None,
    ) -> mx.array:
        """timestep: [t] (mask None) or [t, 0] (causal t=0 row + mask). Pipeline passes t/1000."""
        temb, modulation = self.time_embed(timestep)
        for block in self.blocks:
            h = block(h, modulation, rotary_freqs=rotary_freqs, segments=segments, target_token_mask=target_token_mask)
        h = adaln_scale(h, temb, self.g["norm_out.linear.weight"], mask=target_token_mask)
        return h @ self.g["proj_out.weight"].T
