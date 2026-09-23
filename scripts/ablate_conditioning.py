"""exp-002 conditioning ablations (inference-time only, NO weight edits).

Usage:
  uv run --group dev python scripts/ablate_conditioning.py --arm uncond --out experiments/exp-002 --name abl_uncond
Arms: uncond (embeds*0), half (embeds*0.5), empty (prompt='').
Writes machine-readable JSON next to output PNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
PROMPT = "A capybara wearing a wizard hat, reading a book by candlelight, oil painting"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["uncond", "half", "empty"], required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/exp-002")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    import torch

    out = Path(args.out)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    gen = torch.Generator("cpu").manual_seed(args.seed)

    name = args.name or f"abl_{args.arm}"
    t1 = time.time()
    if args.arm == "empty":
        image = pipe(prompt="", width=1024, height=1024,
                     num_inference_steps=args.steps, generator=gen).images[0]
        detail = {"arm": "empty", "prompt": ""}
    else:
        embeds, mask, _ = pipe.encode_prompt(PROMPT, device=pipe._execution_device)
        scale = 0.0 if args.arm == "uncond" else 0.5
        embeds = embeds * scale
        # Unpadded prompts return mask None; pass through whatever encode_prompt gave.
        image = pipe(prompt_embeds=embeds, prompt_embeds_mask=mask,
                     width=1024, height=1024,
                     num_inference_steps=args.steps, generator=gen).images[0]
        detail = {"arm": args.arm, "scale": scale}
    t_gen = round(time.time() - t1, 1)

    img_path = out / "outputs" / f"{name}.png"
    image.save(img_path)
    sha = hashlib.sha256(img_path.read_bytes()).hexdigest()[:16]
    res = {"model": REPO, "revision": REVISION, "base_prompt": PROMPT,
           "steps": args.steps, "seed": args.seed, "device": device,
           "generation_seconds": t_gen, "output": str(img_path),
           "sha_prefix": sha, "status": "success", **detail}
    print(json.dumps(res, indent=2))
    (out / "metrics" / f"{name}.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
