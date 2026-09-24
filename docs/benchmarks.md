# Benchmarks (machine-readable JSONs alongside; this file is the index)

## Environment
MacBook Pro Mac17,2 / M5 10-core / 17.18GB / macOS 27.0. See `docs/env_baseline.json`.

## Upstream reference (H100, from PR #14804 discussion)
- 2048²/20 steps/bs1/bf16 full pipeline: 56.5 GiB peak (after KV-clone fix), ~32s.

## Local torch baseline (M5, MPS + cpu_offload, 1024², seed 42)
| run | steps | load | gen | s/step | sha | verdict |
|---|---|---|---|---|---|---|
| smoke_s8 | 8 | 1.0s | 752.6s | 94.1 | d128b816 | adherent |
| smoke_s40 | 40 | 1.6s | 2434.3s | 60.9 | 692bc32c | adherent, sharper |
| artistic/comp/text probes | 8 | 1.6s | ~710s | ~89 | … | adherent; text legible |

## Hybrid MLX-DiT (M5; torch TE + MLX DiT + torch VAE/CPU-decode, 1024², seed 42)
| run | steps | denoise | s/step | decode | sha | verdict |
|---|---|---|---|---|---|---|
| hybrid_q8 | 8 | 136.9s | ~17 | 85.1s CPU | dbccf107 | adherent teapot, matches smoke composition |
| hybrid_q4 | 8 | ~135s | ~17 | 83.7s CPU | 4aadc2c | adherent teapot, knob detail present — MEETS BAR |

## Quantization matrix (DiT linears; block0-measured, full-DiT projected)
| config | size | block0 meanΔ vs torch | status |
|---|---|---|---|
| fp32 MLX | 27.9GB | 0.0079 | reference (does not fit 16GB) |
| BF16 file | 14.23GB | n/a | does not fit resident w/ TE |
| Q8 (g64 affine) | ~7.3GB | 0.067 | e2e VERIFIED (teapot) |
| Q4 (g64 affine) | ~3.7GB | 1.10 | e2e VERIFIED (teapot meets bar) |
| Q6/Q5 | — | — | NOT RUN (MLX affine supports any bits; run if Q4 fails bar) |

## Memory verdict (preliminary)
- Q8 hybrid runs on 16GB (denoise resident ~7.3GB + TE streaming). In-process VAE
  decode OOMs after long runs → two-phase (save latents, fresh-process decode).
- Peak unified pressure: NOT YET INSTRUMENTED (post-run RSS only) — required before release.

## Quality verdict
- Q8 @8 steps meets the exp-001 bar (composition + subject).
- Q4 @8 steps meets the bar (knob detail, glaze, wood). Block-level error does NOT
  propagate to visible degradation (loop + VAE absorb it).
- 40-step + blind comparisons NOT RUN.
