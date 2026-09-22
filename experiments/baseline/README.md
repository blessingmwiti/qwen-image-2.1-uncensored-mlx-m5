# Baseline — Qwen Image 2.1 upstream (unmodified)

## Hypothesis
N/A (baseline). The official checkpoint runs on M5/MPS with cpu_offload at 1024 and produces prompt-adherent images.

## Motivation
Ground truth for all later modification/conversion/quant work (§1.3).

## Baseline
- Model: `Qwen/Qwen-Image-2.1@790c9263` (see `results.json` for hashes, `../upstream` docs)
- Pipeline: `QwenImage21Pipeline` from `diffusers@8b3c707e`, `enable_model_cpu_offload()`
- Device: MPS, dtype bf16 (cpu fallback fp32 path unused)

## Modification
None. Weights untouched in HF cache.

## Expected result
Coherent 1024x1024 image, ~90s/step with offload.

## Actual result
- 8-step smoke: SUCCESS. 1024x1024 RGBA teapot, prompt-adherent (see `outputs/smoke_s8_seed42.png`, sha `d128b8160676da8c`).
- Metrics: load 1.0s (cached index), gen 752.6s, 94.08s/step. RSS 0.11GB POST-RUN ONLY — peak NOT MEASURED (limitation, see below).
- 40-step full baseline: see `smoke_s40.json` (running/pending at time of writing).

## Interpretation
Pipeline correct on MPS (no garbling → pre-norm hook + markers OK). Offload makes 33GB runnable on 16GB but slow (~63min for 40 steps at this rate). Quantization + MLX port justified.

## Next step
Phase 2 tensor inventory (shapes/names) → MLX conversion plan → Q4 target.

## Limitations (honest)
- RSS is post-run process memory, not peak unified pressure. No swap measurement yet (§9 NOT satisfied).
- 8 steps ≠ 40-step quality. Full 40-step pending.
