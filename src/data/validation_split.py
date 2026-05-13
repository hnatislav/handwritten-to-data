from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any


def stable_hash(value: str, seed: int = 42) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def record_group_key(record: dict[str, Any]) -> str:
    """Build a deterministic image-level grouping key from available metadata."""
    parts = [
        record.get("source"),
        record.get("year"),
        record.get("grade"),
        record.get("subject"),
        record.get("group"),
        record.get("fund"),
        record.get("file_name"),
    ]
    return "|".join("" if part is None else str(part) for part in parts)


def deterministic_validation_ids(
    records: Iterable[dict[str, Any]],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> set[str]:
    """Create a deterministic image-level validation set.

    This is a draft split helper, not a final competition validation strategy.
    It avoids region-level leakage and is stable across runs for the same inputs.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    validation_ids: set[str] = set()
    threshold = int(validation_fraction * 10_000)
    for record in records:
        image_id = str(record.get("file_name") or record.get("id"))
        bucket = stable_hash(record_group_key(record), seed=seed) % 10_000
        if bucket < threshold:
            validation_ids.add(image_id)
    return validation_ids
