# Compatibility — dependency upgrade for Qwen21 baseline (2026-09-22)

Per AGENTS.md §16: record current, upgrade in explicit commit, test, document.

## Before
- `diffusers==0.40.0` (PyPI) — verified `dir(diffusers)` has NO `QwenImage21*`
- `transformers==5.17.0`, `torch==2.14.0` — meet Qwen21 requirements already
- `accelerate` — not installed

## After (this commit)
- `diffusers @ git+https://github.com/huggingface/diffusers@8b3c707ebd3ec4881f4190cf42931da07eaf3b65`
  - HEAD 2026-09-22, contains PR #14804 merge (Sep 18). Verified `pipeline_qwenimage21.py` exists at this rev with `QwenImage21Pipeline`.
  - Pinned to full SHA, not `main`, for reproducibility.
- `transformers>=5.17` (was `>=4.55`) — Qwen21 model card requires `>=5.17`; installed 5.17.0 stays.
- Added `accelerate>=1.0` — required by model card (`enable_model_cpu_offload` needs it).

## Verification
- `uv lock` + `uv sync` must succeed
- `python -c "from diffusers import QwenImage21Pipeline"` must import
- `pytest -q` (no tests yet → 0 collected, not a failure)

## Rollback
- Previous lock: `uv.lock` at commit `29b98e5` had `diffusers 0.40.0`. Revert `pyproject.toml` + `uv lock` to restore.
