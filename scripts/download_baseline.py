"""Download Qwen-Image-2.1 baseline snapshot (pinned revision) + record hashes.

Usage: uv run --group dev python scripts/download_baseline.py
Writes: experiments/baseline/{config.json,results.json,logs/download.log}
Do NOT modify weights in place. Snapshot goes to HF cache (shared), hashes recorded.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
OUT = Path("experiments/baseline")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
(OUT / "outputs").mkdir(exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    cfg = {"repo": REPO, "revision": REVISION, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))

    log_path = OUT / "logs" / "download.log"
    with log_path.open("w") as log:
        log.write(f"downloading {REPO}@{REVISION}\n")
        snap = snapshot_download(repo_id=REPO, revision=REVISION)
        log.write(f"snapshot: {snap}\n")

    # Record sizes + hashes for weight files only (configs already versioned)
    files = []
    snap_p = Path(snap)
    for p in sorted(snap_p.rglob("*.safetensors")):
        rel = str(p.relative_to(snap_p))
        sz = p.stat().st_size
        print(f"hashing {rel} ({sz/1e9:.2f}GB)...", flush=True)
        files.append({"path": rel, "bytes": sz, "sha256": sha256_file(p)})

    results = {
        "repo": REPO,
        "revision": REVISION,
        "snapshot": str(snap),
        "seconds": round(time.time() - t0, 1),
        "files": files,
        "total_gb": round(sum(f["bytes"] for f in files) / 1e9, 2),
        "status": "success",
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
