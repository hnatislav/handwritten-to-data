from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitArtifactRef:
    name: str
    path: str
    sha256: str | None = None
    dataset_revision: str | None = None
    split_policy: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class PreprocessingManifest:
    version: str
    transforms: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    notes: str | None = None


@dataclass(frozen=True)
class ExperimentTrace:
    experiment_id: str
    config: dict[str, Any]
    split_artifacts: tuple[SplitArtifactRef, ...] = field(default_factory=tuple)
    preprocessing: PreprocessingManifest | None = None
    code_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_hash"] = config_hash(self.config)
        return payload


def write_experiment_trace(trace: ExperimentTrace, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(trace.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output
