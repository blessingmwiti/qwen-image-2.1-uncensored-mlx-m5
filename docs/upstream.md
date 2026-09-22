# Upstream — Qwen Image 2.1 (Phase 1 research, 2026-09-22)

All facts below from authoritative sources per AGENTS.md §15. No weights downloaded yet.

## 1. Official Hugging Face checkpoint (ground truth candidate)
- Repo: `Qwen/Qwen-Image-2.1` — https://huggingface.co/Qwen/Qwen-Image-2.1
- SHA (main, 2026-09-22): `790c92633540aa0cb11d9abf19eb46d861714758`
- pipeline_tag: `text-to-image`, library: diffusers, license: `qwen-research` (NOT Apache-2.0)
- Files (28 total via HfApi):
  - `model_index.json`, `README.md`, `LICENSE`, `.gitattributes`
  - `processor/` (tokenizer, chat template, preprocessor)
  - `text_encoder/` — 4 shards + index (`model-0000X-of-00004.safetensors`)
  - `transformer/` — 2 shards + index (`diffusion_pytorch_model-0000X-of-00002.safetensors`)
  - `vae/` — `diffusion_pytorch_model.safetensors` + config
  - `scheduler/scheduler_config.json`
- Model card notes: 7B visual-gen params, BF16, T2I + editing + RGBA in one `QwenImage21Pipeline`, default 40 steps, native 2K.
- Reason: canonical weights + revision for baseline (hashes TBD on download).

## 2. Official GitHub repo
- Repo: `QwenLM/Qwen-Image-2.1` — https://github.com/QwenLM/Qwen-Image-2.1
- State 2026-09-22: 23 commits, 1.2k stars, 56 forks, released 2026-09-20
- Relevant files: `README.md` (arch + quickstart), `prompt_rewrite/` (T2I/Edit PE models), `LICENSE` (Qwen Research)
- Reason: architecture description + canonical inference params + prompt-rewrite tooling.

## 3. Official Diffusers implementation
- PR: `huggingface/diffusers#14804` — "Add Qwen-Image 2.1", merged Sep 18 2026 (20 commits, base→main)
- New classes: `QwenImage21Transformer2DModel`, `AutoencoderKLQwenImage21`, `QwenImage21Pipeline`, `QwenImage21AttnProcessor` (default exact SDPA multi-pass) + `QwenImage21FlexAttnProcessor` (opt-in, requires `torch.compile`)
- Key fixes in PR history (must respect in MLX port):
  - Text encoder pre-norm hook for transformers>=5.0 (else 5.35/255 image shift, garbled text)
  - Image marker `<image1>…` (not `Picture 1:`), left padding, alpha-composite-over-white for vision encoder, empty prompt → space
  - VAE `scale_factor_spatial=16` (not 8), 4ch in/out, `z_dim=64, decoder_base_dim=144`
  - KV cache `clone()` fix: 64.5→56.5 GiB peak on H100 at 2048/20steps/bs1/bf16
  - Defaults: `num_inference_steps=40`, `true_cfg_scale=1.0` (no CFG)
- Local `diffusers==0.40.0` does NOT contain `QwenImage21*` (verified via `dir(diffusers)`). Need git-main for baseline.
- Requirements per model card: `torch>=2.4.0` (we have 2.14.0 OK), `transformers>=5.17` (we have 5.17.0 OK), `diffusers` from git + `accelerate pillow`.
- Reason: exact reference for transformer/VAE/pipeline behavior.

## 4. Official ComfyUI implementation
- Native Day-0: weights `Comfy-Org/Qwen-Image-2.1`, workflows `image_qwen_image_2_1_t2i.json` + `image_qwen_image_2_1_image_edit.json`
- Reason: validation target for inference behavior, alternate loader.

## 5. Related (not authoritative)
- Prompt rewriters: `Qwen/Qwen-Image-2.1-PE-T2I`, `Qwen/Qwen-Image-2.1-PE-I2I` (Qwen3.5-VL 9B)
- Servers: vLLM-Omni, SGLang (Day-0), LightX2V — perf references only
- Community MLX conversions for 2.1: none found yet (expected — released 2 days ago). Do NOT copy old Qwen-Image (20B) conversions blindly.

## Memory warning (for STATUS)
- Upstream H100 peak: 56.5 GiB BF16 at 2048x2048/20steps/bs1. Our target is 16GB unified → 1024x1024 + offload + Q4 mandatory. Baseline on M5 must use `enable_model_cpu_offload()` + small resolution first.

## Next
- Pin diffusers git revision for baseline, record file hashes on download, implement minimal `QwenImage21Pipeline` smoke (1024, MPS) before any MLX work.
- NOT VERIFIED: exact tensor shapes, checkpoint hashes, local generation.
