# Architecture — Qwen Image 2.1 (configs + tensor shapes verified; weights downloaded unmodified)

Source: HF `Qwen/Qwen-Image-2.1@790c9263` config JSONs + safetensors headers + `QwenLM/Qwen-Image-2.1` README + `diffusers#14804`.

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

## Tensor inventory (verified from safetensors headers, 2026-09-23; weights NOT modified)
- **Transformer** (297 tensors, all BF16): 32 blocks × 9 params. Per block: `attn.norm_q/k [128]`, `attn.to_q/k/v/out [4096,4096]`, `img_mlp.gate/proj [12288,4096]`, `img_mlp.out [4096,12288]`. Single-stream (no per-block text MLP). Non-block: `img_in [4096,64]`, `txt_in.in/out [4096,4096]` + `text_norm [4096]`, `time_text_embed.timestep_embedder.linear_1 [4096,256]` / `linear_2 [4096,4096]`, `modulation.1 [16384,4096]`, `norm_out.linear [4096,4096]`, `proj_out [64,4096]`.
- **Text encoder** (Qwen3-VL): LM 36 layers × 11 params (`input_layernorm/post_attention_layernorm [4096]`, `mlp.gate/up [12288,4096]`, `mlp.down [4096,12288]`, `self_attn.q/o [4096,4096]`, `k/v [1024,4096]` = GQA 32Q:8KV × d128, `q/k_norm [128]`). Vision 27 layers (`attn.qkv [3456,1152]`, `proj [1152,1152]`, `mlp.fc1 [4304,1152]` / `fc2 [1152,4304]`). All BF16.
- **VAE** (238 tensors, all F32 — note: fp32 even in BF16 checkpoint): `decoder.conv_in [1152,64,3,3]`, `conv_out [4,144,3,3]`, mid attentions + resnets at 1152ch. Keep fp32 in MLX port unless verified otherwise.
## Data flow (baseline verified end-to-end 2026-09-23: 8-step MPS smoke produced prompt-adherent 1024 RGBA)
1. Tokenize prompt with image markers → processor (left pad) → Qwen3-VL → pre-norm hidden → prompt_embeds.
2. VAE-encode condition images (RGBA 4ch) for latents; vision branch gets RGB-composited copy.
3. DiT denoises target latents with block-causal attn + prefix KV cache (step1 prefill, steps 2..N cached decode).
4. VAE-decode latents → PIL RGB/RGBA.

## Open questions (remaining)
- Rotary embedding application details + timestep modulation wiring (need code-level port planning).
- 1024² vs 2048² quality/memory tradeoff on M5 (40-step baseline pending).

Status: CONFIGS + TENSOR SHAPES VERIFIED from headers; WEIGHTS DOWNLOADED (33.12GB, hashes in experiments/baseline/results.json, unmodified); 8-STEP MPS BASELINE SUCCESS.
