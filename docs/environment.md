# Environment

Target per AGENTS.md: Apple Silicon, M5, 16GB unified, macOS 26.x, MLX GPU, 1024x1024, Q4.

## Actual (2026-09-22, `docs/env_baseline.json`)
- Mac: MacBook Pro Mac17,2, Apple M5 10-core, 17.18GB (16GB class), arm64
- OS: Darwin 27.0.0, macOS 27.0 (note: newer than AGENTS.md 26.x target)
- Python: 3.12.14 isolated `.venv` via `uv` (global system Python is 3.14.7 Homebrew — NOT used)
- Tools: uv 0.12.17, git 2.54.0, git-lfs 3.8.0
- Packages (locked in `uv.lock`):
  - mlx 0.32.0, mlx-metal 0.32.0
  - torch 2.14.0, torchvision 0.29.0 (MPS available, CUDA unavailable as expected)
  - transformers 5.17.0, diffusers 0.40.0, safetensors 0.8.0, huggingface_hub 1.32.0
  - Pillow 12.3.0, numpy 2.5.3, psutil 7.2.2, tqdm 4.70.1, pytest 9.1.1

## Reproduce
```bash
uv sync --group dev
uv run --group dev python scripts/diagnose_env.py --json-out docs/env_baseline.json
```

## Notes
- Project Python pinned `==3.12.*` + `.python-version=3.12` for torch wheel maturity. Revisit only via explicit upgrade experiment per AGENTS.md §16.
- Do not use global `/opt/homebrew` Python for this project.
