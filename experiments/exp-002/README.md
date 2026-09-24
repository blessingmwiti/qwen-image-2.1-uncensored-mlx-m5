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
DONE 2026-09-23/24. Text-encoder-only run SUCCESS (`metrics/embeddings.json`):
- V1 reference embeds: [1, 19, 4096] for the teapot prompt.
- V2 (post-norm vs pre-norm hook): L2 6328.2, cosine 0.5137 — HUGE. Hook understanding confirmed; MLX TE port (if any) MUST use pre-norm features.
- V3 ("" vs " "): L2 0.0, cosine 1.0 — identical. Empty-guard confirmed.
- V4 (left vs right pad): short prompt L2 72.5/cos 0.9995 (positions matter modestly); longest prompt L2 0.0 (no padding → identical, as expected).
(V3 cosine prints 1.000003 — fp rounding in the metric, harmless.)

## Interpretation
Deltas rank V2 >> V4 > V3(=0) as predicted. The text encoder's final norm is the dominant conditioning detail — a wrong-norm MLX port would shift every generation. Padding side is second-order; empty handling is exact.

## Next step
MLX DiT-block equivalence (port plan step 2) → then TE port-vs-hybrid decision (V2 says: hybrid torch-TE is safe only WITH the hook).
