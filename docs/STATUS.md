# STATUS

## Current phase
Phase 1 — Upstream Baseline (weights downloaded + 8-step MPS smoke SUCCESS; 40-step running)
Phase 2 — Architecture Investigation (tensor inventory done from headers)

## Completed work
- Isolated venv via `uv` (Python 3.12.14, `.venv/`, `uv.lock` committed)
- `pyproject.toml` with pinned `mlx==0.32.0`, torch/diffusers/transformers stack
- `scripts/diagnose_env.py` + `docs/env_baseline.json` (status=success)
- Verified: M5 Metal OK, torch MPS OK, git-lfs OK
- Phase 1 research: `docs/upstream.md` (HF SHA 790c9263, 28 files, qwen-research license) + `docs/architecture.md` skeleton (NOT VERIFIED)
- Verified local `diffusers==0.40.0` lacks `QwenImage21*` — baseline needs git-main (PR #14804)

## Current experiment
- 40-step full baseline (1024, seed 42, MPS+offload) running in background. 8-step smoke is the known-good reference.

## Known failures
- Initial `uv sync` failed: hatchling could not infer wheel package (no src layout). Fixed by adding `src/qwen_mlx/__init__.py` + `[tool.hatch.build.targets.wheel]`.
- Global Python is 3.14.7; project pins 3.12.14 for torch wheel compatibility. Intentional divergence, recorded.

## Next action
1. Await 40-step baseline → record metrics + compare vs 8-step.
2. Phase 2: rotary/modulation code mapping for MLX port plan.
3. Behavioral-modification experiment design (exp-001) per §3/§4 — NO weight changes until hypothesis recorded.

## Blocked tasks
None.

## Benchmark summary
- Env smoke: `mlx_metal=true`, `torch_mps=true`, 17.18GB total.
- Baseline 8-step MPS 1024: load 1.0s, gen 752.6s, 94.08s/step, output 1024 RGBA sha `d128b8160676da8c`. RSS field is POST-RUN ONLY (peak NOT MEASURED — limitation).
- Weights: 33.12GB (TE 17.53 + DiT 14.23 + VAE 1.35), hashes in `experiments/baseline/results.json`.
