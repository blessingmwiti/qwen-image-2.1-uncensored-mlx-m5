"""Fresh-process VAE decode for hybrid latents (avoids OOM after long runs).

Usage: uv run --group dev python scripts/decode_latents.py --inp experiments/hybrid/latents_q8.pt
Writes: experiments/hybrid/outputs/hybrid_q{B}.png + metrics_hybrid_q{B}.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    args = ap.parse_args()
    inp = Path(args.inp)
    out = inp.parent
    d = torch.load(inp, map_location="cpu", weights_only=True)
    latents, bits = d["latents"], d["bits"]

    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor

    vae = AutoencoderKLQwenImage21.from_pretrained(REPO, subfolder="vae", revision=REVISION, dtype=torch.float32)
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    B, C = latents.shape[0], 64
    lat = latents.transpose(1, 2).reshape(B, C, 1, 64, 64)
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    t0 = time.time()
    with torch.no_grad():
        image = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
    image = proc.postprocess(image, output_type="pil")[0]
    (out / "outputs").mkdir(exist_ok=True)
    tag = inp.stem.replace("latents_", "")
    img_path = out / "outputs" / f"hybrid_{tag}.png"
    image.save(img_path)
    sha = hashlib.sha256(img_path.read_bytes()).hexdigest()[:16]
    res = {
        "dit": f"mlx-Q{bits}", "resolution": "1024x1024", "steps": d["steps"], "seed": d["seed"],
        "output": str(img_path), "sha_prefix": sha,
        "decode_seconds": round(time.time() - t0, 1),
        "status": "success",
    }
    mpath = out / f"metrics_hybrid_{tag}.json"
    if mpath.exists():
        try:
            prev = json.loads(mpath.read_text())
            prev.update(res)
            res = prev
        except Exception:
            pass
    mpath.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
