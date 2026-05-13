from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.data.schema import DetectionResult, coerce_bbox


def oracle_detections_from_regions(regions: Iterable[dict[str, Any]]) -> list[DetectionResult]:
    """Create detection results from ground-truth RUKOPYS region annotations."""
    detections: list[DetectionResult] = []
    for region in regions:
        detections.append(
            DetectionResult(
                bbox=coerce_bbox(region["bbox"]),
                type=str(region["type"]),
                score=1.0,
                metadata={
                    "language": region.get("language"),
                    "legibility": region.get("legibility"),
                    "text": region.get("text", ""),
                },
            )
        )
    return detections
