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
PENDING. 40-step teapot running (pid 88084 at design time). Suite probes queued SEQUENTIALLY after it (no concurrent MPS load — 16GB constraint).

## Interpretation
TBD after outputs recorded.

## Next step
- Record per-prompt outputs + metrics in `metrics/`.
- exp-002 (design only after exp-001 data): localize conditioning pathway sensitivity (prompt-embedding ablations, NO weight edits).
