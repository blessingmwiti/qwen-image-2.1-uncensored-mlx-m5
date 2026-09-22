# MLX Port Plan — Qwen Image 2.1 (Phase 2, from `diffusers@8b3c707e` source + verified headers)

No code written yet. This is the conversion blueprint; implement smallest-first per §11/§12.

## Source map (installed `diffusers@8b3c707e`, `.venv/.../diffusers/`)
- Transformer: `models/transformers/transformer_qwenimage21.py`
  - `QwenImage21Transformer2DModel` (~L713): `pos_embed=QwenImage21Rope(theta=10000, axes [16,56,56])`, `txt_in` (TextProjection 4096→4096 + norm), `img_in=Linear(64,4096,bias=False)`, shared `modulation=Sequential(SiLU, Linear(4096→16384,bias=False))`, 32× `QwenImage21TransformerBlock`, `norm_out=AdaLayerNormContinuous`, `proj_out=Linear(4096→64,bias=False)`
  - `QwenImage21TransformerBlock` (~L584): single-stream, no per-block modulation params; `_modulate` = norm → scale/shift → gate.tanh() residual; `modulation.chunk(2)` → mod1 (attn) / mod2 (mlp); `causal_condition` selects t=0 rows for prefix via `_select_modulation_rows`
  - `QwenImage21Attention` (~L551) + `QwenImage21AttnProcessor` (~L462, DEFAULT exact multi-pass SDPA) / `QwenImage21FlexAttnProcessor` (~L369, opt-in compiled flex). Shared `_prepare_qkv` (~L327) + `build_qwenimage21_block_causal_mask` (~L257) + `_prefix_segments` (~L309). KV cache stores POST-RoPE K/V (`QwenImage21KVCache` ~L77, modes extract/cached/extend)
  - RoPE: `apply_rotary_emb_qwen` (~L88), `QwenImage21Rope` (~L652, `rope_params(index,dim,theta=10000)`, pos+neg index concat)
  - Timestep: `TemporalTimesteps(dim=256, max_period=10000, time_factor=1000)` (~L136) → `TimestepEmbedding(256→4096→4096)`; pipeline calls with `timestep/1000`
- VAE: `models/autoencoders/autoencoder_kl_qwenimage21.py` — `CausalConv3d`, `RMS_norm`, `AvgDown3D/DupUp3D/Resample`, `ResidualBlock`, `AttentionBlock`, `MidBlock`, `Encoder3d/Decoder3d`, `AutoencoderKLQwenImage21` (~L977). Weights F32. Single-frame fold (no temporal cache).
- Pipeline: `pipelines/qwenimage21/pipeline_qwenimage21.py` — `vae_scale_factor=16`, `latent_channels=64`, t2i/ti2i templates, left-pad, pre-norm hook, `output_resolution=1024` default in `__call__` (note: model card says 2048 native; code default 1024 — USE EXPLICIT w/h).

## Weight mapping (torch → MLX, per verified headers)
- DiT: `img_in.weight [4096,64]` → transpose to MLX Linear `[64,4096]` convention (verify per-op); `to_q/k/v/out [4096,4096]`; `img_mlp.gate/proj [12288,4096]`, `out [4096,12288]` (SwiGLU: gate+proj → silu(gate)*proj → out); `txt_in.in/out [4096,4096]`; `timestep_embedder.linear_1 [4096,256]`, `linear_2 [4096,4096]`; `modulation.1 [16384,4096]` (no bias); `norm_out [4096,4096]` (AdaLN scale/shift tables); `proj_out [64,4096]`; norms `norm_q/k [128]`, `text_norm [4096]`.
- TE: Qwen3-VL — port ONLY if needed for MLX text encoding; alternative: keep TE on torch/MPS + port DiT+VAE first (smaller step, §11 minimal transfers violated though — decide by measurement).
- VAE: keep F32; port conv/resnet/attention blocks 1:1; verify `scale_factor_spatial=16` tiling math.

## Order (smallest-first)
1. RoPE + timestep + modulation unit tests vs torch (numpy-free, MLX-native).
2. Single DiT block forward equivalence (BF16, fixed seed, tol 1e-3).
3. Full DiT forward (no cache) → with KV cache.
4. VAE encode/decode round-trip.
5. End-to-end BF16 MLX → quant matrix (Q8/Q6/Q5/Q4) → 16GB benchmark.

## Risks
- Block-causal mask in MLX: no flex_attention; implement exact segmented SDPA (default processor logic, not flex).
- GQA in TE (32Q:8KV) if porting TE; DiT itself is MHA 32×d128 (no GQA).
- VAE CausalConv3d cache semantics on tiling.
- 40-step MPS baseline (~1h) is the correctness oracle; MLX must match sha-tolerance, not bit-exact.
