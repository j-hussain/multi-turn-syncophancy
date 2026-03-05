from __future__ import annotations

from pathlib import Path

from .config import env_snapshot, get_git_hash
from .io_utils import write_json


def write_manifest(cfg: dict, run_dir: Path, run_id: str) -> Path:
    payload = {
        "run_id": run_id,
        "config": cfg,
        "git_hash": get_git_hash(),
        "environment": env_snapshot(),
    }
    path = run_dir / "run_manifest.json"
    write_json(path, payload)
    return path
