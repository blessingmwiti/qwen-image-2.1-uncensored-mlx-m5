"""MLX 3-axis RoPE index builder for Qwen-Image-2.1.

Mirrors `QwenImage21Rope.forward` (diffusers@8b3c707e): text tokens advance a
shared position on all axes; each image block freezes frame at the preceding
text position and uses a zero-centered h/w grid. Output complex [S,128] =
concat(frame[16], height[56], width[56]) frequency tables.
"""

from __future__ import annotations

import mlx.core as mx

from qwen_mlx.qwen21_units import rope_freqs

AXES_DIMS = [16, 56, 56]
THETA = 10000
POS_LEN = 8192
NEG_LEN = 1024


def build_tables() -> list[mx.array]:
    """Complex freq tables per axis: [pos(8192) + neg(1024)] rows, dim//2 cols. Built once, reused."""
    pos = mx.arange(POS_LEN)
    neg = -mx.arange(NEG_LEN)[::-1] - 1  # flip(arange(1024)) * -1 - 1
    tables = []
    for dim in AXES_DIMS:
        tables.append(mx.concatenate([rope_freqs(pos, dim), rope_freqs(neg, dim)], axis=0))
    return tables


def build_rope_freqs(
    img_shapes: list[list[tuple[int, int, int]]],
    image_pad_mask: mx.array,
    tables: list[mx.array] | None = None,
) -> mx.array:
    """img_shapes: per-sample list of (1,h,w) blocks (condition images + target).
    image_pad_mask: POST-EXPANSION bool mask over the joint sequence (length P + 4*slots);
    True at image-token positions. Batch shares layout: row 0. Returns complex [S,128]."""
    tables = tables if tables is not None else build_tables()
    mask = image_pad_mask.tolist()
    if mask and isinstance(mask[0], list):
        mask = mask[0]  # batch shares layout; row 0 (matches torch reader)
    total_len = len(mask)
    frame_index, h_idx, w_idx = [], [], []
    img_h, img_w = [], []
    cursor, position = 0, 0
    for _, height, width in img_shapes[0]:
        block_start = mask.index(True, cursor)
        text_len = block_start - cursor
        frame_index.extend(range(position, position + text_len))
        position += text_len
        cursor = block_start + height * width
        frame_index.extend([position] * (height * width))
        position += max(height, width)
        hh = height
        img_h.extend([h for h in range(-(hh - hh // 2), hh // 2) for _ in range(width)])
        img_w.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])
    if cursor < total_len:
        frame_index.extend(range(position, position + total_len - cursor))
    height_index = list(frame_index)
    width_index = list(frame_index)
    j = 0
    for i, m in enumerate(mask):
        if m:
            height_index[i] = img_h[j]
            width_index[i] = img_w[j]
            j += 1
    f = mx.array(frame_index)
    h = mx.array(height_index)
    w = mx.array(width_index)
    return mx.concatenate([tables[0][f], tables[1][h], tables[2][w]], axis=-1)
