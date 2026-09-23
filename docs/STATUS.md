# STATUS

## Current phase
Phase 1 — Upstream Baseline (COMPLETE) + exp-001 quality characterization (COMPLETE)
Phase 2 — Architecture Investigation (inventory + port plan done)
Next: exp-002 conditioning-ablation design (no weight edits) → MLX rope/modulation units

## Completed work
- Isolated venv via `uv` (Python 3.12.14, `.venv/`, `uv.lock` committed)
- `pyproject.toml` with pinned `mlx==0.32.0`, torch/diffusers/transformers stack
- `scripts/diagnose_env.py` + `docs/env_baseline.json` (status=success)
- Verified: M5 Metal OK, torch MPS OK, git-lfs OK
- Phase 1 research: `docs/upstream.md` (HF SHA 790c9263, 28 files, qwen-research license) + `docs/architecture.md` skeleton (NOT VERIFIED)
- Verified local `diffusers==0.40.0` lacks `QwenImage21*` — baseline needs git-main (PR #14804)

## Current experiment
- exp-001 COMPLETE: 4/4 probes prompt-adherent (teapot40, capybara, astronaut, neon text). No weight modifications made anywhere.

## Known failures
- Initial `uv sync` failed: hatchling could not infer wheel package (no src layout). Fixed by adding `src/qwen_mlx/__init__.py` + `[tool.hatch.build.targets.wheel]`.
- Global Python is 3.14.7; project pins 3.12.14 for torch wheel compatibility. Intentional divergence, recorded.

## Next action
1. Design exp-002 (conditioning ablations, no weight edits) — next autonomous step.
2. Implement MLX rope/modulation unit tests vs torch (port plan step 1).
3. NOT started: weight modification, MLX full port, quantization. No push pending user wake (local commits only).

## Blocked tasks
None.

## Benchmark summary
- Env smoke: `mlx_metal=true`, `torch_mps=true`, 17.18GB total.
- Baseline 8-step MPS 1024: load 1.0s, gen 752.6s, 94.08s/step, output 1024 RGBA sha `d128b8160676da8c`. RSS field is POST-RUN ONLY (peak NOT MEASURED — limitation).
- Baseline 40-step MPS 1024: load 1.6s, gen 2434.3s, 60.86s/step, sha `692bc32c247519a8`. Verdict: prompt-adherent, sharper than 8-step. Full quality bar.
- Weights: 33.12GB (TE 17.53 + DiT 14.23 + VAE 1.35), hashes in `experiments/baseline/results.json`.
