from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class RegionPrediction:
    image_id: str
    bbox: BBox
    type: str
    text: str = ""
    score: float | None = None
    source: str | None = None
    annotation_source: str | None = None
    language: str | None = None
    legibility: str | None = None

    def to_submission_dict(self) -> dict[str, Any]:
        text = "" if self.type in {"image", "graph"} else self.text
        return {
            "bbox": list(self.bbox),
            "type": self.type,
            "text": text,
        }


@dataclass(frozen=True)
class PagePrediction:
    image_id: str
    width: int
    height: int
    regions: tuple[RegionPrediction, ...] = field(default_factory=tuple)

    def to_submission_row(self) -> dict[str, Any]:
        return {
            "image": self.image_id,
            "regions": [region.to_submission_dict() for region in self.regions],
        }


@dataclass(frozen=True)
class OCRResult:
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    bbox: BBox
    type: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def coerce_bbox(raw_bbox: list[int] | tuple[int, int, int, int]) -> BBox:
    if len(raw_bbox) != 4:
        raise ValueError(f"Expected bbox with 4 coordinates, got {raw_bbox!r}")
    x1, y1, x2, y2 = (int(round(value)) for value in raw_bbox)
    return x1, y1, x2, y2


def validate_bbox(bbox: BBox, width: int | None = None, height: int | None = None) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox with non-positive area: {bbox!r}")
    if width is not None and (x1 < 0 or x2 > width):
        raise ValueError(f"BBox {bbox!r} exceeds image width {width}")
    if height is not None and (y1 < 0 or y2 > height):
        raise ValueError(f"BBox {bbox!r} exceeds image height {height}")
