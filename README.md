# Qwen Image 2.1 → MLX for Apple Silicon (M5, 16GB)

Reproducible MLX port of [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
(`790c9263`), hybrid pipeline: torch text encoder + **MLX diffusion transformer**
(Q8/Q4) + torch VAE. Verified end-to-end at 1024×1024 on a 16GB M5 Mac.

## Status
- Upstream baseline (torch, MPS): 8-step + 40-step smokes SUCCESS.
- MLX DiT: 12 unit/block tests + full 32-block prefill + Q8 e2e teapot SUCCESS.
- Q4 e2e: running/last — see `docs/STATUS.md`.
- Behavioral modification: DEFERRED (see Limitations).

## Reproduce
```bash
# 1. environment (Python 3.12 isolated via uv)
uv sync --group dev
uv run --group dev python scripts/diagnose_env.py

# 2. baseline weights (33GB, pinned revision, hashes recorded)
uv run --group dev python scripts/download_baseline.py  # bg for slow links

# 3. torch baseline smoke (1024, 8 steps, ~13 min with offload)
uv run --group dev python scripts/baseline_smoke.py --steps 8

# 4. hybrid MLX-DiT generation (Q8, ~7 min + CPU decode)
uv run --group dev python scripts/mlx_dit_hybrid.py --bits 8 --steps 8
uv run --group dev python scripts/decode_latents.py --inp experiments/hybrid/latents_q8.pt

# 5. tests (fast units ~4s; streaming DiT ~40s)
uv run --group dev pytest tests/ -q
```

## Layout
- `src/qwen_mlx/`: `qwen21_units.py` (math), `qwen21_rope.py` (3-axis RoPE),
  `qwen21_block.py` (DiT block, quantizable), `qwen21_dit.py` (model + builders)
- `scripts/`: diagnose/download/smoke/hybrid/decode/exp002
- `docs/`: STATUS, upstream, architecture, mlx_port_plan, benchmarks, compatibility
- `experiments/`: baseline (hashes + smokes), exp-001 (quality bar), exp-002 (conditioning)
- `tests/prompts/`: eval suite v1 (frozen)

## Key findings
- MLX Metal fp32 matmuls carry ~1e-3 relative error (measured) → quality-level (not pixel) equivalence gate.
- VAE weights are F32; real decode uses the temporal feat-cache path (torch VAE kept).
- 14GB BF16 DiT cannot stay resident on 16GB → Q8 (~7.3GB) / Q4 (~3.7GB) deployment.
- See `docs/mlx_port_plan.md` §Precision and §Sequencing decision.

## Limitations
- No KV-cache in MLX DiT yet (full prefill per step).
- No MLX VAE / text encoder (torch, verified).
- No refusal-behavior modification attempted (deferred explicitly).
- Peak unified-memory pressure not yet instrumented (post-run RSS only).
- See `docs/STATUS.md` and `docs/FINAL_REPORT.md` (when released).
