from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.schema import PagePrediction, RegionPrediction
from src.detection.oracle import oracle_detections_from_regions
from src.ocr.recognizer import OCRRecognizer
from src.ordering.reading_order import sort_regions_reading_order
from src.preprocessing.image_ops import crop_region, ensure_rgb


def image_id_from_record(record: dict[str, Any]) -> str:
    for key in ("file_name", "image_id", "id"):
        value = record.get(key)
        if value:
            return str(value)

    image = record.get("image")
    image_filename = getattr(image, "filename", None)
    if image_filename:
        return Path(image_filename).name

    row_index = record.get("_row_index")
    if row_index is not None:
        return f"stream_{int(row_index):06d}.jpg"

    raise ValueError("Could not determine image id from record metadata or PIL image filename")


def page_from_rukopys_record(record: dict[str, Any]) -> PagePrediction:
    image_id = image_id_from_record(record)
    width = int(record.get("image_width") or record["image"].width)
    height = int(record.get("image_height") or record["image"].height)

    regions = []
    for detection in oracle_detections_from_regions(record.get("regions", [])):
        regions.append(
            RegionPrediction(
                image_id=image_id,
                bbox=detection.bbox,
                type=detection.type,
                text=str(detection.metadata.get("text") or ""),
                score=detection.score,
                source=record.get("source"),
                annotation_source=record.get("annotation_source"),
                language=detection.metadata.get("language"),
                legibility=detection.metadata.get("legibility"),
            )
        )

    return PagePrediction(
        image_id=image_id,
        width=width,
        height=height,
        regions=sort_regions_reading_order(regions),
    )


def run_oracle_pipeline_on_record(
    record: dict[str, Any],
    recognizer: OCRRecognizer,
    crop_padding: int = 0,
) -> PagePrediction:
    image = ensure_rgb(record["image"])
    reference_page = page_from_rukopys_record(record)
    predicted_regions: list[RegionPrediction] = []

    for region in reference_page.regions:
        if region.type in {"image", "graph"}:
            ocr_text = ""
            ocr_score = None
        else:
            crop = crop_region(image, region.bbox, padding=crop_padding)
            result = recognizer.recognize(crop, region)
            ocr_text = result.text
            ocr_score = result.score

        predicted_regions.append(
            RegionPrediction(
                image_id=region.image_id,
                bbox=region.bbox,
                type=region.type,
                text=ocr_text,
                score=ocr_score,
                source=region.source,
                annotation_source=region.annotation_source,
                language=region.language,
                legibility=region.legibility,
            )
        )

    return PagePrediction(
        image_id=reference_page.image_id,
        width=reference_page.width,
        height=reference_page.height,
        regions=sort_regions_reading_order(predicted_regions),
    )
