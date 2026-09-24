"""MLX single DiT block for Qwen-Image-2.1 (dense attention, no cache).

Mirrors `QwenImage21TransformerBlock` + `QwenImage21Attention` +
`QwenImage21AttnProcessor` (segments=None path) from `diffusers@8b3c707e`.
Weights use torch/MLX Linear layout [out, in] (no transpose needed).
"""

from __future__ import annotations

import math

import mlx.core as mx

from qwen_mlx.qwen21_units import apply_rotary_complex, layer_norm_no_affine, swiglu


def rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    dtype = x.dtype
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps) * weight.astype(mx.float32)).astype(dtype)


def _modulate(h: mx.array, mod: mx.array) -> tuple[mx.array, mx.array]:
    """mod [..., 2*D] -> (h*(1+scale), gate); caller handles CausalCondition row select + seq broadcast."""
    scale, gate = mx.split(mod, 2, axis=-1)
    return h * (1 + scale), gate


class DiTBlock:
    """Single-stream block. Weights dict with torch names (see docs/architecture.md inventory)."""

    def __init__(self, w: dict[str, mx.array], dim: int = 4096, heads: int = 32, head_dim: int = 128, eps: float = 1e-6):
        self.w = w
        self.heads = heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.eps = eps

    def _project_norm_rope(self, h: mx.array, rotary_freqs: mx.array | None):
        B, S, _ = h.shape
        q = (h @ self.w["attn.to_q.weight"].T).reshape(B, S, self.heads, self.head_dim)
        k = (h @ self.w["attn.to_k.weight"].T).reshape(B, S, self.heads, self.head_dim)
        v = (h @ self.w["attn.to_v.weight"].T).reshape(B, S, self.heads, self.head_dim)
        q = rmsnorm(q, self.w["attn.norm_q.weight"], self.eps)
        k = rmsnorm(k, self.w["attn.norm_k.weight"], self.eps)
        if rotary_freqs is not None:
            q = apply_rotary_complex(q, rotary_freqs)
            k = apply_rotary_complex(k, rotary_freqs)
        return q, k, v

    def attn(
        self,
        h: mx.array,
        rotary_freqs: mx.array | None = None,
    ) -> mx.array:
        B, S, _ = h.shape
        q, k, v = self._project_norm_rope(h, rotary_freqs)
        # dense SDPA over [B,H,S,D]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        o = mx.transpose(o, (0, 2, 1, 3)).reshape(B, S, self.heads * self.head_dim)
        return o @ self.w["attn.to_out.0.weight"].T

    def attn_segmented(
        self,
        h: mx.array,
        rotary_freqs: mx.array | None,
        segments: list[tuple[int, int, bool]],
    ) -> mx.array:
        """Exact segmented prefill mirroring QwenImage21AttnProcessor (key_valid=None, no cache)."""
        B, S, _ = h.shape
        q, k, v = self._project_norm_rope(h, rotary_freqs)
        q4 = mx.transpose(q, (0, 2, 1, 3))
        k4 = mx.transpose(k, (0, 2, 1, 3))
        v4 = mx.transpose(v, (0, 2, 1, 3))
        prefix_len = segments[-1][1] if segments else 0
        outs = []
        for start, end, is_text in segments:
            seg_len = end - start
            if is_text:
                past = mx.zeros((seg_len, start))
                tri = mx.tril(mx.ones((seg_len, seg_len)))
                m = mx.concatenate([past, tri], axis=1)
            else:
                m = mx.ones((seg_len, end))
            add = mx.where(m > 0, 0.0, -1e9)[None, None, :, :]
            outs.append(
                mx.fast.scaled_dot_product_attention(
                    q4[:, :, start:end, :], k4[:, :, :end, :], v4[:, :, :end, :],
                    scale=self.scale, mask=add,
                )
            )
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q4[:, :, prefix_len:, :], k4, v4, scale=self.scale, mask=None
            )
        )
        o = mx.concatenate(outs, axis=2)
        o = mx.transpose(o, (0, 2, 1, 3)).reshape(B, S, self.heads * self.head_dim)
        return o @ self.w["attn.to_out.0.weight"].T

    def __call__(
        self,
        h: mx.array,
        modulation: mx.array,
        rotary_freqs: mx.array | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
    ) -> mx.array:
        """h [B,S,D], modulation [B,4D] (target_token_mask=None path: unsqueezed inside torch; pass per-sample row)."""
        mod1, mod2 = mx.split(modulation, 2, axis=1)  # each [B,2D]
        m1 = mx.expand_dims(mod1, 1)  # [B,1,2D] broadcast over tokens
        m2 = mx.expand_dims(mod2, 1)
        hm, g1 = _modulate(layer_norm_no_affine(h, self.eps), m1)
        attn_out = self.attn_segmented(hm, rotary_freqs, segments) if segments is not None else self.attn(hm, rotary_freqs)
        h = h + mx.tanh(g1) * attn_out
        hm2, g2 = _modulate(layer_norm_no_affine(h, self.eps), m2)
        return h + mx.tanh(g2) * swiglu(
            hm2, self.w["img_mlp.gate_layer.weight"], self.w["img_mlp.proj.weight"], self.w["img_mlp.out.weight"]
        )
