# STATUS

## Current phase
Phase 0 — Environment (in progress, baseline established 2026-09-22)

## Completed work
- Isolated venv via `uv` (Python 3.12.14, `.venv/`, `uv.lock` committed)
- `pyproject.toml` with pinned `mlx==0.32.0`, torch/diffusers/transformers stack
- `scripts/diagnose_env.py` + `docs/env_baseline.json` (status=success)
- Verified: M5 Metal OK, torch MPS OK, git-lfs OK

## Current experiment
None. Phase 1 upstream baseline not started.

## Known failures
- Initial `uv sync` failed: hatchling could not infer wheel package (no src layout). Fixed by adding `src/qwen_mlx/__init__.py` + `[tool.hatch.build.targets.wheel]`.
- Global Python is 3.14.7; project pins 3.12.14 for torch wheel compatibility. Intentional divergence, recorded.

## Next action
1. Phase 1: identify official Qwen Image 2.1 repo + revision + model card (do NOT download yet without recording hash/revision).
2. Create `docs/architecture.md` skeleton + `tests/prompts/` eval suite skeleton.
3. Commit Phase 0.

## Blocked tasks
None.

## Benchmark summary
- Env smoke: `mlx_metal=true`, `mlx_smoke=14.0`, `torch_mps=true`, mem 17.18GB total / 4.93GB avail at baseline run.
- No generation benchmarks yet. See `docs/env_baseline.json`.
