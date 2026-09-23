# exp-002 — Conditioning-embedding sensitivity (TEXT-ENCODER ONLY, no weight modification)

## Hypothesis
Small conditioning variations (pre-norm vs post-norm hidden state, empty-prompt handling, left- vs right-padding) produce measurable prompt-embedding shifts; quantifying them tells us which pathway details the DiT is sensitive to before any weight work (§3: evidence first).

## Motivation
PR #14804 reports: post-norm embeddings shift rendered images 5.35/255; wrong image marker shifts 4.02/255. Reproducing the *embedding-level* deltas locally (a) validates our pipeline understanding, (b) needs only the text encoder (minutes, not hours), (c) never touches weights.

## Baseline
exp-001 quality bar + `experiments/baseline/smoke_s40.json`.

## Modification
NONE to weights. Inference-time encoding variations only:
- V1: pipeline `encode_prompt` (pre-norm hook, left pad) — reference
- V2: same but WITHOUT the pre-norm hook (post-norm `hidden_states[-1]`) — expect large L2 delta
- V3: empty prompt `""` vs `" "` — expect `" "` valid, `""` degenerate
- V4: right-padding vs left-padding (batch of 2 unequal prompts) — expect position shift

## Expected result
V2 delta >> V4 delta > V3 delta, all computed as mean L2 / cosine on `prompt_embeds`. No images generated (DiT skipped) — fast.

## Actual result
PENDING (background run, text-encoder-only with cpu_offload).

## Interpretation
TBD. If V2 reproduces a large delta, our hook understanding is confirmed and the MLX TE port (if any) must use pre-norm features.

## Next step
MLX rope/modulation units (independent) → then decide TE port vs torch-TE-hybrid by measurement.
