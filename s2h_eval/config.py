from __future__ import annotations

import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_id(cfg: dict[str, Any]) -> str:
    explicit = cfg.get("run", {}).get("run_id")
    if explicit:
        return explicit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"s2h-{stamp}-{uuid.uuid4().hex[:8]}"


def get_git_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return None


def env_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
