"""Export quantized MLX DiT weights to disk (deployment checkpoint, §7).

Converts BF16 HF weights -> MLX Q{bits} once, saves with mx.save_weights.
NOT committed to git (weights rule); reproducible via this script + pinned rev.

Usage: uv run --group dev python scripts/export_qdit.py --bits 4 --out models/qwen21-dit-q4.safetensors
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"

SUFFIXES = [
    "attn.norm_k.weight", "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight",
    "attn.to_q.weight", "attn.to_v.weight", "img_mlp.gate_layer.weight",
    "img_mlp.out.weight", "img_mlp.proj.weight",
]
GLOBALS = [
    "img_in.weight", "modulation.1.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "txt_in.in_layer.weight", "txt_in.out_layer.weight", "txt_in.text_norm.weight",
    "norm_out.linear.weight", "proj_out.weight",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--out", default="models/qwen21-dit-q4.safetensors")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from qwen_mlx.qwen21_block import DiTBlock

    t0 = time.time()
    paths = [
        hf_hub_download(REPO, f, revision=REVISION)
        for f in (
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        )
    ]

    def get(name: str) -> mx.array:
        for p in paths:
            with safe_open(p, framework="pt") as f:
                if name in f.keys():
                    return mx.array(f.get_slice(name)[:].float().numpy())
        raise KeyError(name)

    flat: dict[str, mx.array] = {}
    for k in GLOBALS:
        flat[f"globals.{k}"] = get(k)
    for i in range(32):
        w = {s: get(f"transformer_blocks.{i}.{s}") for s in SUFFIXES}
        b = DiTBlock.from_dict(w)
        if args.bits < 32:
            nn.quantize(b, group_size=64, bits=args.bits)
        mx.eval(b.parameters())
        for name, mod in b.named_modules():
            if isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
                flat[f"blocks.{i}.{name}.weight"] = mod.weight
                if isinstance(mod, nn.QuantizedLinear):
                    flat[f"blocks.{i}.{name}.scales"] = mod.scales
                    flat[f"blocks.{i}.{name}.biases"] = mod.biases
        for n in ("_norm_q", "_norm_k"):
            flat[f"blocks.{i}.{n}"] = getattr(b, n)
        del w, b
        gc.collect()
        print(f"block {i} done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as _np

    from safetensors.numpy import save_file

    np_flat = {}
    for k, v in flat.items():
        mx.eval(v)
        np_flat[k] = _np.array(v)
    save_file(np_flat, str(out))
    meta = {
        "repo": REPO, "revision": REVISION, "bits": args.bits, "group_size": 64,
        "seconds": round(time.time() - t0, 1), "bytes": out.stat().st_size,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
