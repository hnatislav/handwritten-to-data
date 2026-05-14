from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from src.data.schema import BBox, PagePrediction, RegionPrediction


TransformType = Literal["identity", "resize", "crop", "pad", "rotate", "deskew", "normalize"]


def stable_id(*parts: Any, prefix: str = "id") -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class TransformRecord:
    transform_id: str
    transform_type: TransformType
    input_frame: str
    output_frame: str
    params: dict[str, Any] = field(default_factory=dict)
    code_version: str | None = None


@dataclass(frozen=True)
class CoordinateRef:
    frame: str = "original_image"
    bbox: BBox | None = None


@dataclass(frozen=True)
class ConfidenceSummary:
    score: float | None = None
    source: str | None = None
    uncertainty_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OCRHypothesis:
    hypothesis_id: str
    text: str
    confidence: ConfidenceSummary = field(default_factory=ConfidenceSummary)
    rank: int = 0
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderMetadata:
    reading_index: int | None = None
    line_index: int | None = None
    within_line_index: int | None = None
    group_id: str | None = None
    order_score: float | None = None
    repair_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DocumentLine:
    line_id: str
    page_id: str
    region_ids: tuple[str, ...]
    bbox: BBox
    text: str = ""
    confidence: ConfidenceSummary = field(default_factory=ConfidenceSummary)
    order: OrderMetadata = field(default_factory=OrderMetadata)
    source_coordinate: CoordinateRef = field(default_factory=CoordinateRef)
    transform_chain: tuple[TransformRecord, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentRegion:
    region_id: str
    page_id: str
    bbox: BBox
    region_type: str
    image_id: str | None = None
    text: str = ""
    source_coordinate: CoordinateRef = field(default_factory=CoordinateRef)
    transform_chain: tuple[TransformRecord, ...] = field(default_factory=tuple)
    confidence: ConfidenceSummary = field(default_factory=ConfidenceSummary)
    ocr_hypotheses: tuple[OCRHypothesis, ...] = field(default_factory=tuple)
    order: OrderMetadata = field(default_factory=OrderMetadata)
    parent_region_id: str | None = None
    line_id: str | None = None
    lineage: tuple[str, ...] = field(default_factory=tuple)
    language: str | None = None
    legibility: str | None = None
    source: str | None = None
    annotation_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def best_text(self) -> str:
        if self.ocr_hypotheses:
            return sorted(self.ocr_hypotheses, key=lambda item: item.rank)[0].text
        return self.text

    def to_region_prediction(self) -> RegionPrediction:
        return RegionPrediction(
            image_id=self.image_id or self.page_id,
            bbox=self.bbox,
            type=self.region_type,
            text=self.best_text(),
            score=self.confidence.score,
            source=self.source,
            annotation_source=self.annotation_source,
            language=self.language,
            legibility=self.legibility,
        )


@dataclass(frozen=True)
class DocumentPage:
    page_id: str
    image_id: str
    width: int
    height: int
    regions: tuple[DocumentRegion, ...] = field(default_factory=tuple)
    lines: tuple[DocumentLine, ...] = field(default_factory=tuple)
    transform_chain: tuple[TransformRecord, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_page_prediction(self) -> PagePrediction:
        return PagePrediction(
            image_id=self.image_id,
            width=self.width,
            height=self.height,
            regions=tuple(region.to_region_prediction() for region in self.regions),
        )


def region_prediction_to_document_region(region: RegionPrediction, index: int = 0) -> DocumentRegion:
    region_id = stable_id(region.image_id, region.bbox, region.type, index, prefix="region")
    hypothesis = OCRHypothesis(
        hypothesis_id=stable_id(region_id, region.text, prefix="ocr"),
        text=region.text,
        confidence=ConfidenceSummary(score=region.score, source="legacy_region_prediction"),
        rank=0,
    )
    return DocumentRegion(
        region_id=region_id,
        page_id=region.image_id,
        image_id=region.image_id,
        bbox=region.bbox,
        region_type=region.type,
        text=region.text,
        source_coordinate=CoordinateRef(frame="original_image", bbox=region.bbox),
        confidence=ConfidenceSummary(score=region.score, source="legacy_region_prediction"),
        ocr_hypotheses=(hypothesis,),
        language=region.language,
        legibility=region.legibility,
        source=region.source,
        annotation_source=region.annotation_source,
        lineage=(region_id,),
    )


def page_prediction_to_document_page(page: PagePrediction) -> DocumentPage:
    regions = tuple(
        region_prediction_to_document_region(region, index=index)
        for index, region in enumerate(page.regions)
    )
    return DocumentPage(
        page_id=page.image_id,
        image_id=page.image_id,
        width=page.width,
        height=page.height,
        regions=regions,
        metadata={"source": "legacy_page_prediction"},
    )
