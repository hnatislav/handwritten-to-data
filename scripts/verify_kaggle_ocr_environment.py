from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import sys
import argparse
from pathlib import Path
from typing import Any


MIN_VERSIONS = {
    "transformers": (4, 45),
    "tokenizers": (0, 20),
    "huggingface-hub": (0, 24),
    "safetensors": (0, 4, 3),
    "sentencepiece": (0, 1, 99),
    "datasets": (2, 18),
}

MAX_EXCLUSIVE_VERSIONS = {
    "transformers": (4, 56),
    "tokenizers": (0, 22),
    "huggingface-hub": (1, 0),
    "datasets": (5, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Kaggle OCR runtime without replacing native torch/CUDA.")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def parse_version(version: str | None) -> tuple[int, ...] | None:
    if version is None:
        return None
    parts: list[int] = []
    for raw_part in version.replace("-", ".").split("."):
        digits = "".join(char for char in raw_part if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def version_at_least(version: str | None, minimum: tuple[int, ...]) -> bool:
    parsed = parse_version(version)
    if parsed is None:
        return False
    width = max(len(parsed), len(minimum))
    padded = parsed + (0,) * (width - len(parsed))
    minimum_padded = minimum + (0,) * (width - len(minimum))
    return padded >= minimum_padded


def version_below(version: str | None, maximum: tuple[int, ...]) -> bool:
    parsed = parse_version(version)
    if parsed is None:
        return False
    width = max(len(parsed), len(maximum))
    padded = parsed + (0,) * (width - len(parsed))
    maximum_padded = maximum + (0,) * (width - len(maximum))
    return padded < maximum_padded


def collect_environment() -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in (
            "torch",
            "torchvision",
            "torchaudio",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "safetensors",
            "sentencepiece",
            "datasets",
            "numpy",
            "pandas",
            "protobuf",
            "Pillow",
        )
    }
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "checks": {},
    }
    try:
        import torch

        payload["torch_runtime"] = {
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
    except Exception as error:  # pragma: no cover - diagnostic script.
        payload["torch_runtime"] = {"error": repr(error)}

    checks: dict[str, Any] = {}
    for name, minimum in MIN_VERSIONS.items():
        maximum = MAX_EXCLUSIVE_VERSIONS.get(name)
        below_maximum = True if maximum is None else version_below(packages.get(name), maximum)
        checks[name] = {
            "installed": packages.get(name),
            "minimum": ".".join(str(part) for part in minimum),
            "maximum_exclusive": ".".join(str(part) for part in maximum) if maximum else None,
            "ok": version_at_least(packages.get(name), minimum) and below_maximum,
        }
    payload["checks"] = checks
    payload["ok"] = all(item["ok"] for item in checks.values())
    return payload


def main() -> None:
    args = parse_args()
    payload = collect_environment()
    cuda_ok = bool(payload.get("torch_runtime", {}).get("cuda_available"))
    payload["checks"]["cuda"] = {
        "required": args.require_cuda,
        "ok": cuda_ok if args.require_cuda else True,
    }
    payload["ok"] = all(item["ok"] for item in payload["checks"].values())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/kaggle_ocr_environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not payload["ok"]:
        raise SystemExit("Kaggle OCR environment check failed.")


if __name__ == "__main__":
    main()
