# STATUS

## Current phase
Phase 1–2 COMPLETE. MLX DiT port + Q8/Q4 hybrid e2e VERIFIED. Behavioral modification EXPLICITLY DEFERRED (see FINAL_REPORT). Paused — awaiting user.

## Completed work (appended)
- Hybrid e2e: MLX-Q8 (`dbccf107`) and MLX-Q4 (`4aadc2c`) teapots, same prompt/seed/steps as smoke_s8 — both meet bar.
- Two-phase decode (latents.pt + fresh process) works around post-run VAE OOM.
- Regression tests (6, fast) + README + LICENSE info + benchmarks + FINAL_REPORT.

## Completed work
- Isolated venv via `uv` (Python 3.12.14, `.venv/`, `uv.lock` committed)
- `pyproject.toml` with pinned `mlx==0.32.0`, torch/diffusers/transformers stack
- `scripts/diagnose_env.py` + `docs/env_baseline.json` (status=success)
- Verified: M5 Metal OK, torch MPS OK, git-lfs OK
- Phase 1 research: `docs/upstream.md` (HF SHA 790c9263, 28 files, qwen-research license) + `docs/architecture.md` (configs + tensor shapes + inventory verified; 8-step + 40-step MPS baselines SUCCESS)
- `diffusers` pinned to git-main `8b3c707e` (PR #14804); `QwenImage21Pipeline` import OK
- MLX math units `src/qwen_mlx/qwen21_units.py` 7/7 vs torch (see `docs/mlx_port_plan.md`)

## Current experiment
- exp-001 COMPLETE: 4/4 probes prompt-adherent. exp-002 COMPLETE: V2 L2 6328/cos 0.51 (hook dominant), V3 exact, V4 second-order.
- No weight modifications made anywhere. No other session's files remain (see session note below).

## Known failures
- Initial `uv sync` failed: hatchling could not infer wheel package (no src layout). Fixed by adding `src/qwen_mlx/__init__.py` + `[tool.hatch.build.targets.wheel]`.
- Global Python is 3.14.7; project pins 3.12.14 for torch wheel compatibility. Intentional divergence, recorded.

## Next action
1. MLX single-DiT-block equivalence vs torch (port plan step 2). PAUSED per user ("continue later").
2. NOT started: weight modification, full MLX port, quantization. No push (local commits only).

## Session note (2026-09-24)
A second session's commit `030ec83` (split norm/rope/timestep modules + ablate script) was overridden per user instruction: consolidated `src/qwen_mlx/qwen21_units.py` (7/7 tests) is canonical. Their work is preserved in git at `030ec83`.

## Blocked tasks
None.

## Benchmark summary
- Env smoke: `mlx_metal=true`, `torch_mps=true`, 17.18GB total.
- Baseline 8-step MPS 1024: load 1.0s, gen 752.6s, 94.08s/step, output 1024 RGBA sha `d128b8160676da8c`. RSS field is POST-RUN ONLY (peak NOT MEASURED — limitation).
- Baseline 40-step MPS 1024: load 1.6s, gen 2434.3s, 60.86s/step, sha `692bc32c247519a8`. Verdict: prompt-adherent, sharper than 8-step. Full quality bar.
- Weights: 33.12GB (TE 17.53 + DiT 14.23 + VAE 1.35), hashes in `experiments/baseline/results.json`.
