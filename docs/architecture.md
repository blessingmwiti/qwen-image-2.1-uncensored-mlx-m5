# Architecture — Qwen Image 2.1 (configs verified from HF rev 790c9263; weights NOT downloaded)

Source: HF `Qwen/Qwen-Image-2.1@790c9263` config JSONs (fetched 2026-09-22, weights excluded) + `QwenLM/Qwen-Image-2.1` README + `diffusers#14804`.

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
- **Text encoder**: `Qwen3VLForConditionalGeneration`, model_type `qwen3_vl`. text_config hidden 4096, 36 layers; vision_config hidden 1152, depth 27, patch 16, spatial_merge 2. Total 8,767,123,696 params / 17.53GB BF16 (from index). Reads `<image1>…` markers, left-padded, alpha-composited-over-white condition images. Output used PRE-final-RMSNorm (`hidden_states[-1]` pre-norm via hook on transformers>=5.0). Empty prompt → space.
- **Transformer**: `QwenImage21Transformer2DModel`, 32 layers, 32 heads, head_dim 128, in/out channels 64, context_in_dim 4096, `axes_dims_rope=[16,56,56]`, patch_size 1, mlp_ratio 3, eps 1e-6, `causal_condition:true`. Total 14,230,249,472 bytes (~14.2GB BF16). Block-causal attention: joint text/image sequence causal, each image block internally bidirectional. Prefix (text + condition images) modulated from t=0 → KV cached after step 1 (`extract`/`cached`/`extend` modes). Two processors: default exact SDPA multi-pass; flex opt-in (compiled only, else OOM at 2K).
- **VAE**: `AutoencoderKLQwenImage21`, base_dim 96, `decoder_base_dim=144`, dim_mult `[1,2,4,8,8]`, num_res_blocks 2, 4ch in/out, `z_dim=64`, `scale_factor_spatial=16`, `scale_factor_temporal=8`. 1024² → (1,64,1,64,64) latent. Tiling supported but per-tile causal-conv cache restarts (values differ, shapes must match).
- **Scheduler**: `FlowMatchEulerDiscreteScheduler`, base_image_seq_len 256, max_image_seq_len 8192 (checkpoint; code default 4096 — USE CHECKPOINT), base_shift 0.5, max_shift 0.9 (checkpoint; code default 1.15 — USE CHECKPOINT), dynamic shifting true, exponential time shift, 1000 train steps. Defaults: 40 steps, `true_cfg_scale=1.0` (no CFG unless negative prompt + scale raised).
- **Resolutions**: native 2K (2048² default; 16:9 2752×1536 etc.). Project primary target 1024² (downscaled, TBD validation).
- **Editing**: up to 10 ref images, circles/masks/annotations, RGBA transparent gen via `This is an RGBA image with transparency…` prompt format.

## Data flow (to verify in Phase 2)
1. Tokenize prompt with image markers → processor (left pad) → Qwen3-VL → pre-norm hidden → prompt_embeds.
2. VAE-encode condition images (RGBA 4ch) for latents; vision branch gets RGB-composited copy.
3. DiT denoises target latents with block-causal attn + prefix KV cache (step1 prefill, steps 2..N cached decode).
4. VAE-decode latents → PIL RGB/RGBA.

## Open questions (Phase 2)
- Parameter names/shapes, rotary embedding application, timestep modulation details (need weight inspection).
- `model_index.json` wiring verified: QwenImage21Pipeline, _diffusers_version 0.37.0.dev0, processor Qwen3VLProcessor, scheduler FlowMatchEulerDiscrete, text_encoder Qwen3VLForConditionalGeneration, transformer QwenImage21Transformer2DModel, vae AutoencoderKLQwenImage21.
- 1024² vs 2048² quality/memory tradeoff on M5.

Status: CONFIGS VERIFIED from JSON; WEIGHTS NOT DOWNLOADED, no local generation yet.
