"""Qwen-Image-2.1 rotary embeddings, MLX-native.

Faithful port of `QwenImage21Rope` + `apply_rotary_emb_qwen(use_real=False)` from
`diffusers@8b3c707e models/transformers/transformer_qwenimage21.py`.
3-axis (frame, height, width); text advances shared position on all axes; each
image block freezes frame at the preceding text position and centers h/w on zero.
Complex freqs path (matches checkpoint usage).
"""

from __future__ import annotations

import math

import mlx.core as mx


def rope_params(index: mx.array, dim: int, theta: int = 10000) -> mx.array:
    half = dim // 2
    ar = mx.arange(half, dtype=mx.float32) / dim
    inv = 1.0 / mx.power(theta * mx.ones_like(ar), ar)
    freqs = mx.outer(index.astype(mx.float32), inv)
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)


class QwenImage21Rope:
    def __init__(self, theta: int = 10000, axes_dim: tuple[int, int, int] = (16, 56, 56)):
        self.theta = theta
        self.axes_dim = list(axes_dim)
        pos = mx.arange(8192)
        neg = mx.flip(mx.arange(1024), axis=0) * -1 - 1
        self.freqs = [
            mx.concatenate([rope_params(pos, d, theta), rope_params(neg, d, theta)], axis=0)
            for d in self.axes_dim
        ]  # each [9216, d//2, 2] real-imag pairs

    def __call__(self, img_shapes: list[tuple[int, int, int]], image_pad_mask: mx.array) -> mx.array:
        total = image_pad_mask.shape[-1]
        is_img = image_pad_mask.tolist()
        frame, img_h, img_w = [], [], []
        cursor, position = 0, 0
        for _, h, w in img_shapes:
            start = is_img.index(True, cursor)
            tlen = start - cursor
            frame.extend(range(position, position + tlen))
            position += tlen
            cursor = start + h * w
            frame.extend([position] * (h * w))
            position += max(h, w)
            img_h.extend([hh for hh in range(-(h - h // 2), h // 2) for _ in range(w)])
            img_w.extend([ww for _ in range(h) for ww in range(-(w - w // 2), w // 2)])
        if cursor < total:
            frame.extend(range(position, position + total - cursor))
        frame_idx = mx.array(frame)
        h_idx = frame_idx.tolist()
        w_idx = frame_idx.tolist()
        flat = image_pad_mask.reshape(-1).tolist()
        j = 0
        for i, m in enumerate(flat):
            if m:
                h_idx[i] = img_h[j]
                w_idx[i] = img_w[j]
                j += 1
        h_idx = mx.array(h_idx)
        w_idx = mx.array(w_idx)
        parts = [
            self.freqs[0][frame_idx],
            self.freqs[1][h_idx],
            self.freqs[2][w_idx],
        ]
        return mx.concatenate(parts, axis=-1)  # [S, 128//2, 2]


def apply_rotary_complex(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Complex-path rotary: x [B,S,H,D] real, freqs [S,D//2,2] → rotated real [B,S,H,D]."""
    cos = freqs_cis[..., 0][None, :, None, :]  # [1,S,1,Dh]
    sin = freqs_cis[..., 1][None, :, None, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return mx.stack([o0, o1], axis=-1).reshape(x.shape)
