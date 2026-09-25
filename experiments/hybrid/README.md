# Hybrid e2e — torch TE + MLX DiT + torch VAE (1024, seed 42, 8 steps)

## Hypothesis
A quantized MLX DiT (Q8, then Q4) in place of the torch DiT produces
prompt-adherent images meeting the exp-001 quality bar.

## Motivation
14GB BF16 DiT cannot stay resident on 16GB alongside the TE → quantization is
the deployment enabler; e2e proves the whole MLX DiT path (rope at scale,
segments, causal mask, scheduler coupling), not just blocks.

## Baseline
`experiments/baseline/smoke_s8.json` (same prompt/seed/steps): sha `d128b816`.

## Modification
transformer forward Rogers → `scripts/mlx_dit_hybrid.py` (MLX DiT, full prefill
per step, no KV cache). TE/VAE/scheduler stay torch. No weight modification.

## Expected / actual
- Q8 @8: sha `dbccf107` — adherent teapot, composition matches smoke. MEETS BAR.
- Q4 @8: sha `4aadc2c` — adherent, knob detail present. MEETS BAR (primary target ✓).
- Q4 @40: sha `3cb4c5e0`, denoise 499.4s (12.5s/step) — bail handle, glaze speckle,
  wood grain. MEETS 40-step BAR. (Note: `outputs/` and `latents_q4.pt` hold the
  latest (40-step) run; 8-step shas preserved in benchmarks + metrics history.)
- First Q8 attempt failed with top/bottom split: rope built on slot mask instead
  of expanded mask (see commit `4e5fee9`). Failure analysis documented in chat.

## Interpretation
Block-level Q4 error (mean 1.1) does NOT propagate to visible degradation —
loop + VAE absorb it. Denoise ~17s/step (vs 94s torch-offload).

## Next step
KV-cache in MLX DiT; 40-step hybrid; blind comparisons; peak-memory instrumentation.

## Operating notes (hard-won)
- Never `from_pretrained` the full pipe here (14GB torch DiT OOMs encode):
  assemble TE/scheduler/VAE à la carte, `transformer=None`, accelerate cpu_offload.
- In-process VAE decode after long runs dies silently (OOM) → two-phase:
  save `latents_q{B}.pt`, decode via `scripts/decode_latents.py` in a fresh process.
