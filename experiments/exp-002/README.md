# exp-002 — Text-conditioning sensitivity (INFERENCE ABLATION, no weight modification)

## Hypothesis
DiT output adherence scales with text-conditioning magnitude: zeroing `prompt_embeds` collapses prompt adherence (unconditioned prior), halving it weakens adherence, and empty prompt falls back to the `" "` path without crashing.

## Motivation
§3 forbids weight edits without evidence. Before touching any representation, we must know whether the text pathway is even the lever: if halved conditioning barely changes output, DiT priors dominate and representation edits are the wrong tool.

## Baseline
exp-001 `artistic_01` (capybara wizard, 8 steps, seed 42, sha `7370a1ad`).

## Modification
NONE to weights. Inference-time only, via supported `prompt_embeds`/`prompt_embeds_mask` args:
- B `uncond`: `prompt_embeds × 0` (same mask) — pure prior.
- C `half`: `prompt_embeds × 0.5` — dose response.
- D `empty`: prompt `" "` path (pipeline converts `""` → `" "`).
Same prompt/seed/steps/device as baseline. 8 steps (relative change is what matters, not absolute quality).

## Expected result
B ≈ scene prior without capybara/wizard/book specificity; C intermediate; D valid image, no crash.

## Actual result
PENDING (queued sequentially, MPS).

## Interpretation
TBD. If B≈baseline, STOP: text pathway is not the lever (§12 correctness-first) and exp-003 must probe elsewhere (timestep modulation, image slots).

## Next step
Record sha + visual verdicts in `results.json`; design exp-003 from outcome.
