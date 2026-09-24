"""MLX DiT block for Qwen-Image-2.1 as nn.Module (quantizable).

Mirrors `QwenImage21TransformerBlock` + `QwenImage21Attention` +
`QwenImage21AttnProcessor` (segments + dense paths) from `diffusers@8b3c707e`.
Linear weights use torch layout [out, in] (MLX nn.Linear native).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from qwen_mlx.qwen21_units import apply_rotary_complex, layer_norm_no_affine, select_rows


def _silu_in(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    dtype = x.dtype
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps) * weight.astype(mx.float32)).astype(dtype)


def linear_from(weight: mx.array, bias: bool = False) -> nn.Linear:
    """nn.Linear with weight assigned from a torch-layout [out, in] array."""
    out_dim, in_dim = weight.shape
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    lin.weight = weight
    return lin


class DiTBlock(nn.Module):
    """Single-stream block. Construct via DiTBlock.from_dict(w) with torch names."""

    def __init__(self, dim: int = 4096, heads: int = 32, head_dim: int = 128, mlp_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.eps = eps
        self.to_q = nn.Linear(dim, heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, heads * head_dim, bias=False)
        self.to_out = nn.Linear(heads * head_dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim * mlp_ratio, bias=False)
        self.proj = nn.Linear(dim, dim * mlp_ratio, bias=False)
        self.out = nn.Linear(dim * mlp_ratio, dim, bias=False)
        self._norm_q = None
        self._norm_k = None

    @classmethod
    def from_dict(cls, w: dict[str, mx.array]) -> "DiTBlock":
        b = cls()
        b.to_q = linear_from(w["attn.to_q.weight"])
        b.to_k = linear_from(w["attn.to_k.weight"])
        b.to_v = linear_from(w["attn.to_v.weight"])
        b.to_out = linear_from(w["attn.to_out.0.weight"])
        b.gate = linear_from(w["img_mlp.gate_layer.weight"])
        b.proj = linear_from(w["img_mlp.proj.weight"])
        b.out = linear_from(w["img_mlp.out.weight"])
        b._norm_q = w["attn.norm_q.weight"]
        b._norm_k = w["attn.norm_k.weight"]
        return b

    def _project_norm_rope(self, h: mx.array, rotary_freqs: mx.array | None):
        B, S, _ = h.shape
        q = self.to_q(h).reshape(B, S, self.heads, self.head_dim)
        k = self.to_k(h).reshape(B, S, self.heads, self.head_dim)
        v = self.to_v(h).reshape(B, S, self.heads, self.head_dim)
        q = rmsnorm(q, self._norm_q, self.eps)
        k = rmsnorm(k, self._norm_k, self.eps)
        if rotary_freqs is not None:
            q = apply_rotary_complex(q, rotary_freqs)
            k = apply_rotary_complex(k, rotary_freqs)
        return q, k, v

    @staticmethod
    def _sdpa(q, k, v, scale, mask=None):
        q4 = mx.transpose(q, (0, 2, 1, 3))
        k4 = mx.transpose(k, (0, 2, 1, 3))
        v4 = mx.transpose(v, (0, 2, 1, 3))
        o = mx.fast.scaled_dot_product_attention(q4, k4, v4, scale=scale, mask=mask)
        return mx.transpose(o, (0, 2, 1, 3)).reshape(q.shape[0], q.shape[1], -1)

    def attn(self, h: mx.array, rotary_freqs: mx.array | None = None) -> mx.array:
        q, k, v = self._project_norm_rope(h, rotary_freqs)
        return self.to_out(self._sdpa(q, k, v, self.scale))

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
            mx.fast.scaled_dot_product_attention(q4[:, :, prefix_len:, :], k4, v4, scale=self.scale, mask=None)
        )
        o = mx.transpose(mx.concatenate(outs, axis=2), (0, 2, 1, 3)).reshape(B, S, -1)
        return self.to_out(o)

    def _modulate(self, h: mx.array, mod: mx.array, mask: mx.array | None):
        scale, gate = mx.split(mod, 2, axis=-1)
        return h * (1 + select_rows(scale, mask)), select_rows(gate, mask)

    def __call__(
        self,
        h: mx.array,
        modulation: mx.array,
        rotary_freqs: mx.array | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        target_token_mask: mx.array | None = None,
    ) -> mx.array:
        """h [B,S,D]; modulation [B,4D] (mask None) or [B+1,4D] with causal t=0 row + mask."""
        mod1, mod2 = mx.split(modulation, 2, axis=1)
        hm, g1 = self._modulate(layer_norm_no_affine(h, self.eps), mod1, target_token_mask)
        attn_out = self.attn_segmented(hm, rotary_freqs, segments) if segments is not None else self.attn(hm, rotary_freqs)
        h = h + mx.tanh(g1) * attn_out
        hm2, g2 = self._modulate(layer_norm_no_affine(h, self.eps), mod2, target_token_mask)
        h = h + mx.tanh(g2) * self.out(_silu_in(self.gate(hm2)) * self.proj(hm2))
        return h
