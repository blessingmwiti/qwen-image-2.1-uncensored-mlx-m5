"""exp-002: text-encoder-only conditioning ablations (no weight mods, no images).

Usage: uv run --group dev python scripts/exp002_embeddings.py
Writes: experiments/exp-002/metrics/embeddings.json (machine-readable)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
OUT = Path("experiments/exp-002/metrics")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = "A ceramic teapot on a wooden table, morning light"


def cos_l2(a, b):
    import torch

    a, b = a.float(), b.float()
    l2 = (a - b).norm().item()
    cos = (a.flatten() @ b.flatten() / (a.norm() * b.norm())).item()
    return round(l2, 4), round(cos, 6)


def main() -> int:
    import torch

    from diffusers import QwenImage21Pipeline

    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    pipe = QwenImage21Pipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    dev = pipe._execution_device
    res: dict = {"model": REPO, "revision": REVISION, "status": "success", "variants": {}}

    # V1 reference
    e1, m1, ip1 = pipe.encode_prompt(PROMPT, device=dev)
    res["variants"]["V1"] = {"shape": list(e1.shape)}

    # V2: post-norm (no hook) vs pre-norm (hook) — manual encoder call
    tmpl = pipe.prompt_template_t2i.format(PROMPT)
    inp = pipe.processor(text=[tmpl], padding=True, padding_side="left", return_tensors="pt").to(dev)
    kw = {"input_ids": inp.input_ids, "attention_mask": inp.attention_mask, "output_hidden_states": True}
    with torch.no_grad():
        out_post = pipe.text_encoder(**kw)
    post = out_post.hidden_states[-1]
    # mask-extract + drop_idx mirror (single prompt, keep simple: compare full-sequence L2 pre/post)
    tm = getattr(pipe.text_encoder.model, "language_model", pipe.text_encoder.model)
    handle = tm.norm.register_forward_hook(lambda mod, a, o: a[0])
    try:
        with torch.no_grad():
            out_pre = pipe.text_encoder(**kw)
    finally:
        handle.remove()
    pre = out_pre.hidden_states[-1]
    l2, cos = cos_l2(pre.cpu(), post.cpu())
    res["variants"]["V2_no_hook_vs_hook"] = {"l2": l2, "cosine": cos}

    # V3: empty vs space (pipeline guards "" -> " ")
    e_empty, _, _ = pipe.encode_prompt("", device=dev)
    e_space, _, _ = pipe.encode_prompt(" ", device=dev)
    l2, cos = cos_l2(e_empty.cpu(), e_space.cpu())
    res["variants"]["V3_empty_vs_space"] = {"l2": l2, "cosine": cos}

    # V4: left (pipeline) vs right pad, batch of unequal prompts
    prompts = ["short", "a much longer prompt describing a ceramic teapot in detail"]
    e_left, _, _ = pipe.encode_prompt(prompts, device=dev)
    # manual right-pad replication of template tokenization
    tmpls = [pipe.prompt_template_t2i.format(t) for t in prompts]
    inp_r = pipe.processor(text=tmpls, padding=True, padding_side="right", return_tensors="pt").to(dev)
    kw_r = {"input_ids": inp_r.input_ids, "attention_mask": inp_r.attention_mask, "output_hidden_states": True}
    handle = tm.norm.register_forward_hook(lambda mod, a, o: a[0])
    try:
        with torch.no_grad():
            out_r = pipe.text_encoder(**kw_r)
    finally:
        handle.remove()
    # extract valid tokens per sample (mirror _extract_masked_hidden + drop_idx)
    import torch as _t

    def extract(hs, mask):
        sel = hs[mask.bool()]
        lens = mask.bool().sum(dim=1).tolist()
        parts = _t.split(sel, lens, dim=0)
        return [p[pipe._drop_idx :] for p in parts]

    pre_r = extract(out_r.hidden_states[-1], inp_r.attention_mask)
    # left-pad reference per-sample
    inp_l = pipe.processor(text=tmpls, padding=True, padding_side="left", return_tensors="pt").to(dev)
    kw_l = {"input_ids": inp_l.input_ids, "attention_mask": inp_l.attention_mask, "output_hidden_states": True}
    handle = tm.norm.register_forward_hook(lambda mod, a, o: a[0])
    try:
        with torch.no_grad():
            out_l = pipe.text_encoder(**kw_l)
    finally:
        handle.remove()
    pre_l = extract(out_l.hidden_states[-1], inp_l.attention_mask)
    d = []
    for a, b in zip(pre_l, pre_r):
        n = min(a.shape[0], b.shape[0])
        l2c, cosc = cos_l2(a[:n].cpu(), b[:n].cpu())
        d.append({"l2": l2c, "cosine": cosc, "len_l": a.shape[0], "len_r": b.shape[0]})
    res["variants"]["V4_left_vs_right_pad"] = d

    (OUT / "embeddings.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
