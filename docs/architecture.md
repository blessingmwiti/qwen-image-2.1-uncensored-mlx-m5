# Architecture — Qwen Image 2.1 (initial, from upstream; NOT VERIFIED locally)

Source: `QwenLM/Qwen-Image-2.1` README + `diffusers#14804`. Local weight inspection pending (Phase 2).

```
Qwen3-VL 8B text encoder (unified text + condition images)
↓ prompt_embeds + mask (+ image_pad slots)
Qwen Image 2.1 transformer / DiT (32 layers, 7B, single-stream, block-causal)
↓ latent (64ch, 16x compressed)
VAE (64ch RGBA autoencoder, 16x spatial)
↓
image (RGB/RGBA)
```

## Components
- **Text encoder**: Qwen3-VL 8B VLM. Reads `<image1>…` markers, left-padded, alpha-composited-over-white condition images. Output used PRE-final-RMSNorm (`hidden_states[-1]` pre-norm via hook on transformers>=5.0). Empty prompt → space.
- **Transformer**: 32-layer single-stream DiT, 7B. Block-causal attention: joint text/image sequence causal, each image block internally bidirectional. `causal_condition:true` default. Prefix (text + condition images) modulated from t=0 → KV cached after step 1 (`extract`/`cached`/`extend` modes). Two processors: default exact SDPA multi-pass; flex opt-in (compiled only, else OOM at 2K).
- **VAE**: `AutoencoderKLQwenImage21`, `z_dim=64`, `decoder_base_dim=144`, 4ch in/out, `scale_factor_spatial=16`. 1024² → (1,64,1,64,64) latent. Tiling supported but per-tile causal-conv cache restarts (values differ, shapes must match).
- **Scheduler**: Flow Matching, Euler discrete + dynamic shifting. Defaults: 40 steps, `true_cfg_scale=1.0` (no CFG unless negative prompt + scale raised).
- **Resolutions**: native 2K (2048² default; 16:9 2752×1536 etc.). Project primary target 1024² (downscaled, TBD validation).
- **Editing**: up to 10 ref images, circles/masks/annotations, RGBA transparent gen via `This is an RGBA image with transparency…` prompt format.

## Data flow (to verify in Phase 2)
1. Tokenize prompt with image markers → processor (left pad) → Qwen3-VL → pre-norm hidden → prompt_embeds.
2. VAE-encode condition images (RGBA 4ch) for latents; vision branch gets RGB-composited copy.
3. DiT denoises target latents with block-causal attn + prefix KV cache (step1 prefill, steps 2..N cached decode).
4. VAE-decode latents → PIL RGB/RGBA.

## Open questions (Phase 2)
- Exact `transformer/config.json`, `text_encoder/config.json`, `vae/config.json` values.
- `model_index.json` component wiring.
- Parameter names/shapes, rotary embedding scheme, timestep modulation details.
- 1024² vs 2048² quality/memory tradeoff on M5.

Status: NOT VERIFIED — no local checkpoint inspection yet.
