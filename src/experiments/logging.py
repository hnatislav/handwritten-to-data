from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_experiment_metadata(
    experiment_dir: str | Path,
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> Path:
    experiment_path = Path(experiment_dir)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": current_git_commit(),
        "config": config,
        "dataset_info": dataset_info,
        "metrics": metrics or {},
    }
    return write_json(experiment_path / "metadata.json", payload)
