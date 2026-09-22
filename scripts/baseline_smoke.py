"""Minimal upstream baseline smoke (Phase 1). MPS + cpu_offload, 1024x1024.

Usage: uv run --group dev python scripts/baseline_smoke.py [--steps 8] [--out experiments/baseline]
Writes machine-readable JSON per AGENTS.md §28. Does NOT modify weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default="A ceramic teapot on a wooden table, morning light")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/baseline")
    args = ap.parse_args()

    import torch

    out = Path(args.out)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    from diffusers import QwenImage21Pipeline

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=dtype)
    t_load = round(time.time() - t0, 1)
    pipe.enable_model_cpu_offload()

    gen = torch.Generator("cpu").manual_seed(args.seed)
    t1 = time.time()
    image = pipe(
        prompt=args.prompt, width=1024, height=1024,
        num_inference_steps=args.steps, generator=gen,
    ).images[0]
    t_gen = round(time.time() - t1, 1)

    img_path = out / "outputs" / f"smoke_s{args.steps}_seed{args.seed}.png"
    image.save(img_path)
    sha = hashlib.sha256(img_path.read_bytes()).hexdigest()[:16]

    try:
        import psutil

        rss = round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        rss = None

    res = {
        "model": REPO, "revision": REVISION, "resolution": "1024x1024",
        "steps": args.steps, "seed": args.seed, "device": device,
        "load_seconds": t_load, "generation_seconds": t_gen,
        "seconds_per_step": round(t_gen / max(args.steps, 1), 2),
        "rss_gb": rss, "output": str(img_path), "sha_prefix": sha,
        "status": "success",
    }
    print(json.dumps(res, indent=2))
    (out / f"smoke_s{args.steps}.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
