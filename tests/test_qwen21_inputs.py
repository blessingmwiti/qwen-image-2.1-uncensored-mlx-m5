"""Input-path equivalence: txt_in/img_in/joint/metadata vs torch (REAL weights, fp32)."""

import mlx.core as mx
import numpy as np
import torch

from qwen_mlx.qwen21_dit import DiTModel, build_token_metadata


def _globals():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    rev = "790c92633540aa0cb11d9abf19eb46d861714758"
    paths = [
        hf_hub_download("Qwen/Qwen-Image-2.1", f, revision=rev)
        for f in (
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        )
    ]
    keys = [
        "img_in.weight", "txt_in.in_layer.weight", "txt_in.out_layer.weight", "txt_in.text_norm.weight",
        "norm_out.linear.weight", "proj_out.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_2.weight",
        "modulation.1.weight",
    ]
    out = {}
    for k in keys:
        for p in paths:
            with safe_open(p, framework="pt") as f:
                if k in f.keys():
                    out[k] = f.get_slice(k)[:].float()
                    break
        assert k in out, k
    return out


def test_input_path():
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21TextProjection,
        QwenImage21Transformer2DModel,
    )

    torch.manual_seed(9)
    g = _globals()
    enc = torch.randn(1, 6, 4096) * 0.5
    lat = torch.randn(1, 4, 64) * 0.5  # 1 target slot x4 = 4 packed tokens
    mask = torch.tensor([[False] * 6 + [True]])
    shapes = [(1, 2, 2)]

    txt = QwenImage21TextProjection(4096, 4096).float().eval()
    txt.text_norm.weight.data = g["txt_in.text_norm.weight"]
    txt.in_layer.weight.data = g["txt_in.in_layer.weight"]
    txt.out_layer.weight.data = g["txt_in.out_layer.weight"]
    img_in = torch.nn.Linear(64, 4096, bias=False).float().eval()
    img_in.weight.data = g["img_in.weight"]
    with torch.no_grad():
        e_ref = txt(enc)
        h_ref = img_in(lat)
        repeats = torch.where(mask, 4, 1)[0]
        morate = torch.cat([e_ref, e_ref.new_zeros(1, 1, 4096)], dim=1)
        morate = morate.repeat_interleave(repeats, dim=1)
        exp_mask = torch.repeat_interleave(mask[0], repeats)
        morate[:, exp_mask] = h_ref
        ids_ref, tgt_ref = QwenImage21Transformer2DModel.build_token_metadata(exp_mask, shapes)

    mg = {k: mx.array(v.numpy()) for k, v in g.items()}
    dit = DiTModel(mg, [])
    h_m, e_m = dit.project_inputs(mx.array(lat.numpy()), mx.array(enc.numpy()))
    pre = mx.array(np.array([[False] * 6 + [True]]))
    joint = dit.build_joint(h_m, e_m, pre, shapes)
    exp = mx.array(np.repeat(np.array([[False] * 6 + [True]]), [1] * 6 + [4], axis=1))
    ids_m, tgt_m = build_token_metadata(exp, shapes)
    mx.eval(joint, ids_m, tgt_m)
    # scale-aware tol: txt projection reaches |176| through 2 Metal matmuls (~1e-3 rel each);
    # 0.158 max diff measured = rel 9e-4 = platform noise (see plan §Precision). No bug.
    assert torch.allclose(torch.from_numpy(np.array(joint, dtype=np.float32)), morate, atol=0.5, rtol=5e-3)
    assert ids_m.tolist() == ids_ref.tolist()
    assert tgt_m.tolist() == tgt_ref.tolist()
    print("joint ok", tuple(np.array(joint).shape), ids_m.tolist())
