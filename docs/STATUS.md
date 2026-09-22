# STATUS

## Current phase
Phase 1 — Upstream Baseline (research done, no weights downloaded yet)

## Completed work
- Isolated venv via `uv` (Python 3.12.14, `.venv/`, `uv.lock` committed)
- `pyproject.toml` with pinned `mlx==0.32.0`, torch/diffusers/transformers stack
- `scripts/diagnose_env.py` + `docs/env_baseline.json` (status=success)
- Verified: M5 Metal OK, torch MPS OK, git-lfs OK
- Phase 1 research: `docs/upstream.md` (HF SHA 790c9263, 28 files, qwen-research license) + `docs/architecture.md` skeleton (NOT VERIFIED)
- Verified local `diffusers==0.40.0` lacks `QwenImage21*` — baseline needs git-main (PR #14804)

## Current experiment
None. Upstream identified; baseline download + smoke not started.

## Known failures
- Initial `uv sync` failed: hatchling could not infer wheel package (no src layout). Fixed by adding `src/qwen_mlx/__init__.py` + `[tool.hatch.build.targets.wheel]`.
- Global Python is 3.14.7; project pins 3.12.14 for torch wheel compatibility. Intentional divergence, recorded.

## Next action
1. Pin diffusers git revision containing PR #14804 merge; record in docs (no upgrade without experiment per §16).
2. Download baseline weights with hashes (expect ~30GB+ total: 7B DiT + 8B text encoder + VAE) — confirm disk space first.
3. Minimal MPS smoke: 1024x1024, 40 steps (or fewer for smoke), seed 42, `enable_model_cpu_offload()`, record time/mem/checksum.
4. Create `tests/prompts/` eval suite skeleton.

## Blocked tasks
None.

## Benchmark summary
- Env smoke: `mlx_metal=true`, `mlx_smoke=14.0`, `torch_mps=true`, mem 17.18GB total / 4.93GB avail at baseline run.
- Upstream H100 reference: 56.5 GiB peak BF16 at 2048/20steps/bs1 → 16GB M5 requires 1024 + offload + quant. No local generation yet.
