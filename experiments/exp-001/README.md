# exp-001 — Baseline behavior characterization (OBSERVATIONAL, no weight modification)

## Hypothesis
The unmodified `Qwen/Qwen-Image-2.1@790c9263` pipeline produces prompt-adherent, high-quality images on the fixed eval suite v1 at 1024/MPS, establishing the quality bar any later modification must preserve (§6).

## Motivation
§3 requires evidence before modifying weights. First we need a measured reference: what does "normal" look like (composition, text rendering, aesthetics) before asking where refusal behavior lives.

## Baseline
- `experiments/baseline/smoke_s8.json` (teapot, sha `d128b8160676da8c`) + pending 40-step.
- Eval suite `tests/prompts/suite.json` v1.

## Modification
NONE. Weights untouched. This experiment only runs inference.

## Expected result
Coherent images for benign_01, artistic_01, text_01 ("QWEN IMAGE 2.1" legible at 40 steps), comp_01. Edit_01 needs input image (deferred). Difficult_01 (RGBA) exercises alpha path.

## Actual result
DONE 2026-09-23. All 8-step MPS probes SUCCESS (seed 42, 1024):
- benign (teapot 40-step ref): sha `692bc32c247519a8`, 2434s — sharp, prompt-adherent.
- artistic_01 (capybara wizard): sha `7370a1ad7770e12b`, 706s — hat/book/candle/oil-paint all present.
- comp_01 (astronaut jungle): sha `5fcae7c6266c8f1d`, 713s — cold palette, detailed suit.
- text_01 (neon "QWEN IMAGE 2.1"): sha `14604614db54e269` — legible at 8 steps (stylized Q), rain reflections present. Text pathway (pre-norm hook) confirmed.
Metrics in `metrics/`, images in `outputs/` (gitignored, local only).

## Interpretation
Baseline quality bar established: composition, artistic style, and text rendering all prompt-adherent at 8 steps; 40-step ref strictly sharper. Any modification must preserve this. No refusal behavior observed on benign suite (expected) — refusal probing is a separate protocol, not ad-hoc prompts.

## Next step
- exp-002 (design only after exp-001 data): localize conditioning pathway sensitivity (prompt-embedding ablations, NO weight edits).
