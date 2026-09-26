"""Export roundtrip: quantize -> safetensors.numpy -> restore -> identical outputs."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from qwen_mlx.qwen21_block import DiTBlock
from qwen_mlx.qwen21_load import _restore_linear


def test_export_roundtrip_block0():
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file as np_load
    from safetensors.numpy import save_file as np_save
    from safetensors.torch import load_file as t_load

    p = hf_hub_download(
        "Qwen/Qwen-Image-2.1",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        revision="790c92633540aa0cb11d9abf19eb46d861714758",
    )
    sd = t_load(p, device="cpu")
    w = {k[len("transformer_blocks.0."):]: v.float() for k, v in sd.items() if k.startswith("transformer_blocks.0.")}

    torch.manual_seed(7)
    b = DiTBlock.from_dict({k: mx.array(v.numpy()) for k, v in w.items()})
    nn.quantize(b, group_size=64, bits=4)
    h = mx.array(torch.randn(1, 16, 4096).numpy() * 0.5)
    mm = mx.array(torch.randn(1, 4 * 4096).numpy() * 0.2)
    ref = b(h, mm)
    mx.eval(ref)

    # save
    flat = {}
    for name, mod in b.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            mx.eval(mod.weight, mod.scales, mod.biases)
            flat[f"{name}.weight"] = np.array(mod.weight)
            flat[f"{name}.scales"] = np.array(mod.scales)
            flat[f"{name}.biases"] = np.array(mod.biases)
    tmp = "/tmp/qblock0_test.safetensors"
    np_save(flat, tmp)

    # restore into fresh block
    raw = np_load(tmp)
    b2 = DiTBlock()
    for attr in ("to_q", "to_k", "to_v", "to_out", "gate", "proj", "out"):
        setattr(b2, attr, _restore_linear(
            mx.array(raw[f"{attr}.weight"]), mx.array(raw[f"{attr}.scales"]),
            mx.array(raw[f"{attr}.biases"]), 4, 64))
    b2._norm_q = mx.array(w["attn.norm_q.weight"].numpy())
    b2._norm_k = mx.array(w["attn.norm_k.weight"].numpy())
    got = b2(h, mm)
    mx.eval(got)
    d = (torch.from_numpy(np.array(got, dtype=np.float32)) - torch.from_numpy(np.array(ref, dtype=np.float32))).abs()
    print("roundtrip max:", d.max().item())
    assert d.max().item() == 0.0
