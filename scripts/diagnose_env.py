"""Environment diagnostic for Qwen Image 2.1 MLX project (Phase 0).

Verifies Apple Silicon environment per AGENTS.md and emits machine-readable JSON.
Usage: uv run python scripts/diagnose_env.py [--json-out path]
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
import time


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cmd(cmd: list[str]) -> str | None:
    path = shutil.which(cmd[0])
    if path is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (out.stdout or out.stderr or "").strip()[:2000] or None
    except Exception as e:  # noqa: BLE001 - diagnostic must not crash
        return f"ERROR: {e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Hardware / OS
    mac_ver = platform.mac_ver()[0] or None
    hw_model = _cmd(["sysctl", "-n", "hw.model"])
    cpu_brand = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem_bytes = _cmd(["sysctl", "-n", "hw.memsize"])
    try:
        mem_gb = round(int(mem_bytes) / 1e9, 2) if mem_bytes and mem_bytes.isdigit() else None
    except ValueError:
        mem_gb = None

    # Toolchain
    result: dict = {
        "timestamp": ts,
        "hardware": {
            "model": hw_model,
            "chip": cpu_brand,
            "memory_gb": mem_gb,
            "arch": platform.machine(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "mac_version": mac_ver,
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "tools": {
            "uv": _cmd(["uv", "--version"]),
            "git": _cmd(["git", "--version"]),
            "git_lfs": _cmd(["git-lfs", "--version"]),
        },
        "packages": {},
        "accelerators": {},
        "status": "success",
    }

    for pkg in [
        "mlx",
        "mlx-metal",
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "safetensors",
        "huggingface_hub",
        "Pillow",
        "numpy",
        "psutil",
        "tqdm",
        "pytest",
    ]:
        result["packages"][pkg] = _pkg_version(pkg)

    # MLX check (native, no numpy round-trip)
    try:
        import mlx.core as mx

        t0 = time.time()
        x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        y = mx.sum(x * x)
        mx.eval(y)
        result["accelerators"]["mlx_metal"] = bool(mx.metal.is_available())
        result["accelerators"]["mlx_smoke"] = float(y.item())
        result["accelerators"]["mlx_seconds"] = round(time.time() - t0, 4)
    except Exception as e:  # noqa: BLE001
        result["accelerators"]["mlx_metal"] = False
        result["accelerators"]["mlx_error"] = str(e)[:500]
        result["status"] = "degraded"

    # Torch check
    try:
        import torch

        result["accelerators"]["torch_version"] = torch.__version__
        result["accelerators"]["torch_mps_available"] = bool(torch.backends.mps.is_available())
        result["accelerators"]["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001
        result["accelerators"]["torch_error"] = str(e)[:500]
        result["status"] = "degraded"

    # Memory snapshot
    try:
        import psutil

        vm = psutil.virtual_memory()
        result["memory"] = {
            "total_gb": round(vm.total / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
            "percent": vm.percent,
        }
    except Exception as e:  # noqa: BLE001
        result["memory"] = {"error": str(e)[:200]}

    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
