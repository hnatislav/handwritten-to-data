from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
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


def package_versions(package_names: list[str] | tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            (
                "torch",
                "transformers",
                "datasets",
                "Pillow",
                "numpy",
                "pandas",
                "opencv-python",
                "scikit-learn",
                "sentencepiece",
                "protobuf",
                "tiktoken",
            )
        ),
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as error:  # pragma: no cover - diagnostic metadata only.
        payload["torch"] = {"error": repr(error)}
    return payload


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
        "runtime": runtime_environment(),
        "git_commit": current_git_commit(),
        "config": config,
        "dataset_info": dataset_info,
        "metrics": metrics or {},
    }
    return write_json(experiment_path / "metadata.json", payload)
