from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw

from src.document.schema import DocumentPage, DocumentRegion
from src.ordering.evaluator_order import bbox_area, bbox_iou


FailureType = Literal[
    "detection_failure",
    "split_merge_failure",
    "overlap_duplicate_failure",
    "ordering_failure",
    "ocr_failure",
    "postprocess_failure",
]


@dataclass(frozen=True)
class FailureAttribution:
    failure_type: FailureType
    region_id: str | None
    matched_region_id: str | None
    iou: float | None
    details: dict[str, str | int | float | bool | None]


def bbox_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return bbox_area((ix1, iy1, ix2, iy2))


def bbox_coverage(source: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> float:
    area = bbox_area(source)
    if area <= 0:
        return 0.0
    return bbox_intersection_area(source, target) / area


def match_regions(
    reference: tuple[DocumentRegion, ...],
    prediction: tuple[DocumentRegion, ...],
    iou_threshold: float = 0.5,
) -> tuple[dict[str, tuple[DocumentRegion, float]], set[str]]:
    matches: dict[str, tuple[DocumentRegion, float]] = {}
    used_predictions: set[str] = set()
    for ref in reference:
        candidates = [
            (pred, bbox_iou(ref.bbox, pred.bbox))
            for pred in prediction
            if pred.region_id not in used_predictions
        ]
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
        if candidates and candidates[0][1] >= iou_threshold:
            pred, iou = candidates[0]
            matches[ref.region_id] = (pred, iou)
            used_predictions.add(pred.region_id)
    return matches, used_predictions


def attribute_page_failures(
    reference: DocumentPage,
    prediction: DocumentPage,
    iou_threshold: float = 0.5,
    split_merge_coverage_threshold: float = 0.35,
) -> list[FailureAttribution]:
    matches, used_predictions = match_regions(reference.regions, prediction.regions, iou_threshold=iou_threshold)
    failures: list[FailureAttribution] = []

    for ref in reference.regions:
        overlapping_predictions = [
            pred
            for pred in prediction.regions
            if bbox_coverage(ref.bbox, pred.bbox) >= split_merge_coverage_threshold
        ]
        if len(overlapping_predictions) > 1:
            failures.append(
                FailureAttribution(
                    failure_type="split_merge_failure",
                    region_id=ref.region_id,
                    matched_region_id=None,
                    iou=None,
                    details={
                        "reason": "reference_region_split_across_predictions",
                        "overlap_count": len(overlapping_predictions),
                    },
                )
            )

    for pred in prediction.regions:
        overlapping_references = [
            ref
            for ref in reference.regions
            if bbox_coverage(ref.bbox, pred.bbox) >= split_merge_coverage_threshold
        ]
        if len(overlapping_references) > 1:
            failures.append(
                FailureAttribution(
                    failure_type="split_merge_failure",
                    region_id=None,
                    matched_region_id=pred.region_id,
                    iou=None,
                    details={
                        "reason": "predicted_region_merges_references",
                        "overlap_count": len(overlapping_references),
                    },
                )
            )

    for ref in reference.regions:
        match = matches.get(ref.region_id)
        if match is None:
            failures.append(
                FailureAttribution(
                    failure_type="detection_failure",
                    region_id=ref.region_id,
                    matched_region_id=None,
                    iou=None,
                    details={"reason": "reference_region_unmatched"},
                )
            )
            continue
        pred, iou = match
        if ref.order.reading_index != pred.order.reading_index:
            failures.append(
                FailureAttribution(
                    failure_type="ordering_failure",
                    region_id=ref.region_id,
                    matched_region_id=pred.region_id,
                    iou=iou,
                    details={
                        "reference_order": ref.order.reading_index,
                        "predicted_order": pred.order.reading_index,
                    },
                )
            )
        if ref.best_text() != pred.best_text():
            failures.append(
                FailureAttribution(
                    failure_type="ocr_failure",
                    region_id=ref.region_id,
                    matched_region_id=pred.region_id,
                    iou=iou,
                    details={
                        "reason": "matched_region_text_mismatch",
                        "reference_text": ref.best_text(),
                        "predicted_text": pred.best_text(),
                    },
                )
            )
        if pred.ocr_hypotheses and pred.text and pred.text != pred.best_text():
            failures.append(
                FailureAttribution(
                    failure_type="postprocess_failure",
                    region_id=ref.region_id,
                    matched_region_id=pred.region_id,
                    iou=iou,
                    details={
                        "reason": "stored_text_differs_from_selected_hypothesis",
                        "stored_text": pred.text,
                        "selected_text": pred.best_text(),
                    },
                )
            )

    for pred in prediction.regions:
        if pred.region_id not in used_predictions:
            failures.append(
                FailureAttribution(
                    failure_type="overlap_duplicate_failure",
                    region_id=None,
                    matched_region_id=pred.region_id,
                    iou=None,
                    details={"reason": "predicted_region_unmatched"},
                )
            )
    return failures


def write_failure_report(failures: list[FailureAttribution], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(failure) for failure in failures], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def draw_ordering_overlay(
    image: Image.Image,
    page: DocumentPage,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for region in page.regions:
        draw.rectangle(region.bbox, outline="red", width=3)
        label = "" if region.order.reading_index is None else str(region.order.reading_index)
        draw.text((region.bbox[0], max(0, region.bbox[1] - 14)), label, fill="red")
    canvas.save(output)
    return output
