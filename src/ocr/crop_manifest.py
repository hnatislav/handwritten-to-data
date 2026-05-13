from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def crop_root_from_manifest(path: str | Path) -> Path:
    manifest_path = Path(path)
    # Expected layout: <export_root>/manifests/<split>.jsonl
    return manifest_path.parent.parent


def resolve_crop_path(manifest_path: str | Path, row: dict[str, Any]) -> Path:
    return crop_root_from_manifest(manifest_path) / row["crop_path"]


def text_length_bucket(text: str) -> str:
    length = len(text)
    if length <= 5:
        return "000_005"
    if length <= 15:
        return "006_015"
    if length <= 30:
        return "016_030"
    if length <= 60:
        return "031_060"
    if length <= 120:
        return "061_120"
    return "121_plus"
