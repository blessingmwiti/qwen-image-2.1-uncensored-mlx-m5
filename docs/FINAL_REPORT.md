# FINAL REPORT — Qwen Image 2.1 MLX (hybrid milestone)

## Hardware
- MacBook Pro Mac17,2, Apple M5 (4P+6E), 17.18GB unified, macOS 27.0 (note: newer than AGENTS 26.x target).

## Software
- Python 3.12.14 isolated (`uv`, `uv.lock` committed); global 3.14.7 NOT used.
- mlx 0.32.0, torch 2.14.0 (MPS), transformers 5.17.0, diffusers git-main `8b3c707e` (PR #14804), safetensors 0.8.0.

## Model
- Upstream `Qwen/Qwen-Image-2.1@790c9263` (qwen-research license), 33.12GB, hashes in `experiments/baseline/results.json`. Unmodified.
- Architecture: 32-layer single-stream DiT 7B (block-causal, prefix KV cache) + Qwen3-VL 8B encoder + 64ch RGBA VAE (F32, 16x).
- MLX port: DiT math units + rope builder + block + full prefill + input path + causal mask — all vs-torch tested. TE/VAE stay torch (deferred with reasons in `docs/mlx_port_plan.md`).
- Quantization: MLX affine g64; Q8 (~7.3GB) and Q4 (~3.7GB) DiT verified e2e. Q6/Q5 NOT RUN.

## Performance (1024², seed 42, M5)
- Torch offload: 8-step 753s (94s/step); 40-step 2434s (61s/step).
- Hybrid Q8/Q4: 8-step ~137s denoise (~17s/step, no KV cache yet) + ~85s CPU VAE decode.
- Peak unified pressure: NOT INSTRUMENTED (limitation). In-process VAE decode after long runs OOMs → two-phase (latents.pt + fresh decode).

## Quality
- Baseline bar (exp-001): 4/4 probes adherent (teapot40, capybara, astronaut, neon text legible@8).
- Hybrid Q8/Q4 teapots (same prompt/seed/steps): adherent, composition matches smoke. Q4 keeps knob detail.
- 40-step hybrid, blind comparisons, editing/RGBA paths: NOT RUN.

## Limitations (explicit)
1. No KV-cache in MLX DiT (5-6x slowdown available later).
2. No MLX VAE (temporal feat-cache semantics mapped, not implemented) / no MLX text encoder.
3. No refusal-behavior modification attempted — see below.
4. No ComfyUI/API integration; no Q6/Q5; no 2048² validation; no peak-memory instrumentation.
5. macOS 27.0 vs 26.x target; code UNLICENSED; model under Qwen Research License.

## Behavioral modification (NOT DONE — explicit)
No weight edits were made in this project state. Rationale: (a) the engineering
track (port + quant + verification) was prerequisite and is now delivered;
(b) refusal-behavior work needs a dedicated safety-behavior eval protocol (the
v1 suite's safety category is intentionally empty) plus hypothesis-first
experiments per §3/§4 — that protocol does not exist yet. Next: define refusal
probes, run exp-003 observational characterization, then propose mechanism
hypotheses. Nothing here removes or weakens any model safety behavior.

## Reproduction
See README.md (5 commands). Known-good shas: smoke_s40 `692bc32c`, hybrid_q8
`dbccf107`, hybrid_q4 `4aadc2c`.
