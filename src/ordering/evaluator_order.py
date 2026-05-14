from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.data.schema import BBox
from src.document.schema import DocumentLine, DocumentPage, DocumentRegion, OrderMetadata, stable_id


@dataclass(frozen=True)
class OrderingConfig:
    vertical_tolerance: float = 0.5
    overlap_threshold: float = 0.85
    duplicate_iou_threshold: float = 0.95
    duplicate_text_match_required: bool = True


@dataclass(frozen=True)
class OrderingDiagnostics:
    page_id: str
    input_regions: int
    suppressed_duplicates: int
    line_count: int
    repair_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OrderedPageResult:
    page: DocumentPage
    diagnostics: OrderingDiagnostics


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_union(bboxes: list[BBox] | tuple[BBox, ...]) -> BBox:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = bbox_area((ix1, iy1, ix2, iy2))
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


def y_center(bbox: BBox) -> float:
    return (bbox[1] + bbox[3]) / 2


def bbox_height(bbox: BBox) -> int:
    return max(1, bbox[3] - bbox[1])


def suppress_duplicate_regions(
    regions: tuple[DocumentRegion, ...],
    config: OrderingConfig,
) -> tuple[tuple[DocumentRegion, ...], list[dict[str, Any]]]:
    kept: list[DocumentRegion] = []
    events: list[dict[str, Any]] = []
    for region in sorted(regions, key=lambda item: (item.bbox[1], item.bbox[0], item.region_id)):
        duplicate_of: DocumentRegion | None = None
        for existing in kept:
            if bbox_iou(region.bbox, existing.bbox) < config.duplicate_iou_threshold:
                continue
            if config.duplicate_text_match_required and region.best_text() != existing.best_text():
                continue
            duplicate_of = existing
            break
        if duplicate_of is None:
            kept.append(region)
        else:
            events.append(
                {
                    "event": "duplicate_suppressed",
                    "region_id": region.region_id,
                    "duplicate_of": duplicate_of.region_id,
                    "iou": bbox_iou(region.bbox, duplicate_of.bbox),
                }
            )
    return tuple(kept), events


def cluster_regions_into_lines(
    regions: tuple[DocumentRegion, ...],
    page_id: str,
    config: OrderingConfig,
) -> tuple[tuple[DocumentLine, ...], list[dict[str, Any]]]:
    remaining = sorted(regions, key=lambda item: (item.bbox[1], item.bbox[0], item.bbox[2], item.region_id))
    lines: list[DocumentLine] = []
    events: list[dict[str, Any]] = []

    while remaining:
        anchor = remaining.pop(0)
        anchor_center = y_center(anchor.bbox)
        anchor_height = bbox_height(anchor.bbox)
        line_regions = [anchor]
        next_remaining: list[DocumentRegion] = []

        for region in remaining:
            tolerance = config.vertical_tolerance * max(anchor_height, bbox_height(region.bbox))
            if abs(y_center(region.bbox) - anchor_center) <= tolerance:
                line_regions.append(region)
            else:
                next_remaining.append(region)

        line_regions = sorted(line_regions, key=lambda item: (item.bbox[0], item.bbox[1], item.bbox[2], item.region_id))
        line_bbox = bbox_union(tuple(region.bbox for region in line_regions))
        line_index = len(lines)
        line_id = stable_id(page_id, line_index, line_bbox, prefix="line")
        if len(line_regions) > 1:
            events.append(
                {
                    "event": "line_cluster",
                    "line_id": line_id,
                    "region_ids": [region.region_id for region in line_regions],
                    "bbox": list(line_bbox),
                }
            )
        lines.append(
            DocumentLine(
                line_id=line_id,
                page_id=page_id,
                region_ids=tuple(region.region_id for region in line_regions),
                bbox=line_bbox,
                text="\n".join(region.best_text() for region in line_regions if region.best_text()),
                order=OrderMetadata(line_index=line_index, group_id=line_id),
            )
        )
        remaining = next_remaining

    return tuple(lines), events


def order_page_regions(page: DocumentPage, config: OrderingConfig | None = None) -> OrderedPageResult:
    cfg = config or OrderingConfig()
    deduped_regions, duplicate_events = suppress_duplicate_regions(page.regions, cfg)
    lines, line_events = cluster_regions_into_lines(deduped_regions, page.page_id, cfg)
    region_by_id = {region.region_id: region for region in deduped_regions}
    ordered_regions: list[DocumentRegion] = []
    repair_events = duplicate_events + line_events

    for line_index, line in enumerate(lines):
        line_regions = [region_by_id[region_id] for region_id in line.region_ids]
        for within_line_index, region in enumerate(line_regions):
            reading_index = len(ordered_regions)
            ordered_regions.append(
                replace(
                    region,
                    line_id=line.line_id,
                    order=OrderMetadata(
                        reading_index=reading_index,
                        line_index=line_index,
                        within_line_index=within_line_index,
                        group_id=line.line_id,
                        repair_flags=("duplicate_checked",),
                    ),
                )
            )

    ordered_page = DocumentPage(
        page_id=page.page_id,
        image_id=page.image_id,
        width=page.width,
        height=page.height,
        regions=tuple(ordered_regions),
        lines=lines,
        transform_chain=page.transform_chain,
        metadata={**page.metadata, "ordering_config": cfg.__dict__},
    )
    diagnostics = OrderingDiagnostics(
        page_id=page.page_id,
        input_regions=len(page.regions),
        suppressed_duplicates=len(duplicate_events),
        line_count=len(lines),
        repair_events=tuple(repair_events),
    )
    return OrderedPageResult(page=ordered_page, diagnostics=diagnostics)
