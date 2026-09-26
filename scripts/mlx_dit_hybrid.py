"""Hybrid e2e: torch TE + MLX DiT (quantized) + torch VAE/scheduler.

Same prompt/seed/resolution as experiments/baseline smoke_s8 for direct
quality comparison. No weight modification (reads HF cache). No KV cache
(full prefill per step — slower but exact).

Usage: uv run --group dev python scripts/mlx_dit_hybrid.py [--bits 8] [--steps 8]
Writes: experiments/hybrid/metrics_hybrid_q{B}.json + outputs/hybrid_q{B}.png
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
PROMPT = "A ceramic teapot on a wooden table, morning light"


def t2m(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def m2t(a: mx.array) -> torch.Tensor:
    mx.eval(a)
    return torch.from_numpy(np.array(a, dtype=np.float32))


def load_mlx_dit(bits: int):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from qwen_mlx.qwen21_block import DiTBlock
    from qwen_mlx.qwen21_dit import DiTModel

    paths = [
        hf_hub_download(REPO, f, revision=REVISION)
        for f in (
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        )
    ]

    def get(name: str) -> mx.array:
        for p in paths:
            with safe_open(p, framework="pt") as f:
                if name in f.keys():
                    return mx.array(f.get_slice(name)[:].float().numpy())
        raise KeyError(name)

    suffixes = [
        "attn.norm_k.weight", "attn.norm_q.weight", "attn.to_k.weight", "attn.to_out.0.weight",
        "attn.to_q.weight", "attn.to_v.weight", "img_mlp.gate_layer.weight",
        "img_mlp.out.weight", "img_mlp.proj.weight",
    ]
    globals_w = {
        k: get(k)
        for k in (
            "img_in.weight", "modulation.1.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
            "txt_in.in_layer.weight", "txt_in.out_layer.weight", "txt_in.text_norm.weight",
            "norm_out.linear.weight", "proj_out.weight",
        )
    }
    print("globals loaded", flush=True)
    blocks = []
    for i in range(32):
        w = {s: get(f"transformer_blocks.{i}.{s}") for s in suffixes}
        b = DiTBlock.from_dict(w)
        if bits < 32:
            nn.quantize(b, group_size=64, bits=bits)
        mx.eval(b.parameters())
        blocks.append(b)
        del w, b
        gc.collect()
        print(f"block {i} {'Q'+str(bits) if bits < 32 else 'fp32'}", flush=True)
    dit = DiTModel(globals_w, [])
    dit.blocks = blocks
    return dit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dit-weights", default=None)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    from diffusers import QwenImage21Pipeline

    out = Path("experiments/hybrid")
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    tag = args.tag or f"q{args.bits}"

    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    t0 = time.time()
    # Load ONLY what the hybrid needs (TE + scheduler + VAE). The 14GB torch DiT is
    # NEVER materialized (OOM: full from_pretrained exceeds unified memory before del).
    from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    te = Qwen3VLForConditionalGeneration.from_pretrained(
        REPO, subfolder="text_encoder", revision=REVISION, dtype=dtype,
    )
    proc = Qwen3VLProcessor.from_pretrained(REPO, subfolder="processor", revision=REVISION)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler", revision=REVISION)
    vae = AutoencoderKLQwenImage21.from_pretrained(REPO, subfolder="vae", revision=REVISION, dtype=torch.float32)
    # Same streaming strategy as the proven smoke runs: accelerate CPU-offload hooks
    # per component (never materialize the 14GB torch DiT at all).
    from accelerate import cpu_offload

    te = cpu_offload(te)
    vae = cpu_offload(vae)
    pipe = QwenImage21Pipeline(scheduler=sched, vae=vae, text_encoder=te, processor=proc, transformer=None)
    print(f"components loaded in {time.time()-t0:.1f}s (torch DiT skipped)", flush=True)

    t0 = time.time()
    if args.dit_weights:
        from qwen_mlx.qwen21_dit import DiTModel
        from qwen_mlx.qwen21_load import load_qdit

        globals_w, blocks = load_qdit(args.dit_weights, bits=args.bits)
        dit = DiTModel(globals_w, [])
        dit.blocks = blocks
        print(f"MLX DiT loaded from {args.dit_weights} in {time.time()-t0:.1f}s", flush=True)
    else:
        dit = load_mlx_dit(args.bits)
        print(f"MLX DiT Q{args.bits} ready in {time.time()-t0:.1f}s", flush=True)

    from qwen_mlx.qwen21_dit import build_token_metadata, prefix_segments
    from qwen_mlx.qwen21_rope import build_rope_freqs, build_tables

    gen = torch.Generator("cpu").manual_seed(args.seed)
    t1 = time.time()
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    cond_pil, vae_img, input_size = None, None, None
    if args.image:
        from PIL import Image as PILImage

        img0 = PILImage.open(args.image).convert("RGBA")
        iw, ih = img0.size
        input_size = calculate_dimensions(1024 * 1024, iw / ih)[:2]
        cond_pil = pipe.image_processor.resize(img0, width=input_size[0], height=input_size[1])
        vae_img = pipe.image_processor.preprocess(img0, width=input_size[0], height=input_size[1]).unsqueeze(2)
    prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
        args.prompt, image=[cond_pil] if cond_pil is not None else None, device=pipe._execution_device,
    )
    print(f"encoded in {time.time()-t1:.1f}s", flush=True)

    # latents via pipeline helper (T2I: no condition images)
    latents, cond_latents = pipe.prepare_latents(
        [vae_img] if vae_img is not None else None, 1, 64, 1024, 1024,
        prompt_embeds.dtype, pipe._execution_device, gen, None,
    )
    if cond_pil is not None:
        vw, vh = input_size
        shapes = [(1, vh // 16, vw // 16), (1, 64, 64)]
    else:
        shapes = [(1, 64, 64)]
    # append target slots to the prefix mask (mirror pipeline append_target_slots)
    n_slots = latents.shape[1] // 4
    pre_mask_t = torch.cat([image_pad_mask, image_pad_mask.new_ones(image_pad_mask.shape[0], n_slots)], dim=1)
    # img_shapes for 1024 T2I target
    # (metadata built with MLX builders below; torch reference mirrored in tests)

    # scheduler timesteps (torch, checkpoint config)
    import numpy as _np

    sigmas = _np.linspace(1.0, 1 / args.steps, args.steps)
    cfg = pipe.scheduler.config
    m_slope = (cfg.get("max_shift", 1.15) - cfg.get("base_shift", 0.5)) / (cfg.get("max_image_seq_len", 4096) - cfg.get("base_image_seq_len", 256))
    mu = latents.shape[1] * m_slope + cfg.get("base_shift", 0.5) - m_slope * cfg.get("base_image_seq_len", 256)
    pipe.scheduler.set_timesteps(args.steps, device=pipe._execution_device, sigmas=sigmas.tolist(), mu=mu)
    timesteps = pipe.scheduler.timesteps

    # prompt/latent projections in MLX (per step inputs are torch; convert)
    m_enc = t2m(prompt_embeds.float())
    m_cond = t2m(cond_latents.float()) if cond_latents is not None else None
    tables = build_tables()
    pre_np = np.array(pre_mask_t)
    rep_pat = [4 if b else 1 for b in pre_np[0]]
    exp_np = np.repeat(pre_np, rep_pat, axis=1)
    m_ids, m_tgt_full = build_token_metadata(mx.array(exp_np), shapes)
    mx.eval(m_ids, m_tgt_full)
    P = int(exp_np.shape[1] - np.asarray(m_tgt_full).sum())  # non-target prefix length
    pre_mask = mx.array(pre_np)
    m_ids, _ = build_token_metadata(mx.array(exp_np), shapes)
    m_segs = prefix_segments(m_ids, P)
    # rope MUST use the post-expansion mask (length P + target tokens), not the slot mask
    m_freqs = build_rope_freqs([shapes], mx.array(exp_np), tables)
    # target_token_mask over JOINT sequence: prefix False, target True
    tgt_np = np.asarray(m_tgt_full)
    joint_tgt = mx.array(tgt_np[None, :])

    t_denoise = time.time()
    peak = {"rss": 0.0, "swap": 0.0}
    stop_flag = {"stop": False}

    def _sample():
        try:
            import psutil

            proc = psutil.Process()
            while not stop_flag["stop"]:
                peak["rss"] = max(peak["rss"], proc.memory_info().rss / 1e9)
                try:
                    peak["swap"] = max(peak["swap"], psutil.swap_memory().used / 1e9)
                except Exception:
                    pass
                time.sleep(0.5)
        except Exception:
            pass

    import threading as _th

    _sampler = _th.Thread(target=_sample, daemon=True)
    _sampler.start()
    caches = None
    for i, t in enumerate(timesteps):
        s0 = time.time()
        m_lat = t2m(latents.float())
        if m_cond is not None:
            m_lat = mx.concatenate([m_cond, m_lat], axis=1)
        h_lat, enc = dit.project_inputs(m_lat, m_enc)
        joint = dit.build_joint(h_lat, enc, pre_mask, shapes)
        ts = mx.array(np.array([float(t) / 1000.0, 0.0], dtype=np.float32))
        if i == 0:
            noise_full, caches = dit.prefill(joint, ts, m_freqs, m_segs, joint_tgt, P)
            mx.eval(noise_full)
            L = latents.shape[1]
            npred = m2t(noise_full)[:, -L:].to(device=latents.device, dtype=latents.dtype)
        else:
            tgt = joint[:, P:]
            mx.eval(tgt)
            noise_t = dit.decode(tgt, ts, m_freqs[P:], joint_tgt[:, P:], caches)
            mx.eval(noise_t)
            npred = m2t(noise_t).to(device=latents.device, dtype=latents.dtype)
        latents = pipe.scheduler.step(npred, t, latents, return_dict=False)[0]
        print(f"step {i+1}/{len(timesteps)} {time.time()-s0:.1f}s", flush=True)
    stop_flag["stop"] = True

    gen_s = time.time() - t_denoise
    stop_flag["stop"] = True
    print(f"denoise done in {gen_s:.1f}s; saving latents before decode", flush=True)
    torch.save({"latents": latents.cpu(), "steps": args.steps, "seed": args.seed, "bits": args.bits, "prompt": args.prompt, "input_image": args.image}, out / f"latents_{tag}.pt")
    (out / f"metrics_hybrid_{tag}.json").write_text(json.dumps({
        "model": REPO, "revision": REVISION, "dit": f"mlx-Q{args.bits}", "resolution": "1024x1024",
        "steps": args.steps, "seed": args.seed, "generation_seconds": round(gen_s, 1),
        "seconds_per_step": round(gen_s / args.steps, 1),
        "peak_rss_gb": round(peak["rss"], 2), "swap_used_gb": round(peak["swap"], 2),
        "kv_cache": True, "status": "denoised",
    }, indent=2))
    print("latents+metrics saved; freeing MLX DiT before VAE decode", flush=True)
    try:
        import mlx.metal as _metal

        _has_metal_cache = hasattr(_metal, "clear_cache")
    except Exception:
        _has_metal_cache = False
    del dit
    gc.collect()
    if _has_metal_cache:
        import mlx.metal as _metal2

        _metal2.clear_cache()
    # decode (mirror pipeline tail)
    print("vae decode start", flush=True)
    B, C = latents.shape[0], 64
    H = W = 1024
    lat = latents.transpose(1, 2).reshape(B, C, 1, H // 16, W // 16).to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 64, 1, 1, 1).to(lat.device, lat.dtype)
    std = torch.tensor(pipe.vae.config.latents_std).view(1, 64, 1, 1, 1).to(lat.device, lat.dtype)
    image = pipe.vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    img_path = out / "outputs" / f"hybrid_{tag}.png"
    image.save(img_path)
    sha = hashlib.sha256(img_path.read_bytes()).hexdigest()[:16]
    res = {
        "model": REPO, "revision": REVISION, "dit": f"mlx-Q{args.bits}", "resolution": "1024x1024",
        "steps": args.steps, "seed": args.seed, "prompt": args.prompt, "input_image": args.image, "output": str(img_path), "sha_prefix": sha,
        "generation_seconds": round(gen_s, 1), "seconds_per_step": round(gen_s / args.steps, 1),
        "peak_rss_gb": round(peak["rss"], 2), "swap_used_gb": round(peak["swap"], 2),
        "kv_cache": True,
        "status": "success",
    }
    (out / f"metrics_hybrid_{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
