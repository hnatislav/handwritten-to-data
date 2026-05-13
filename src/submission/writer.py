from __future__ import annotations

import csv
import json
from pathlib import Path

from src.data.schema import PagePrediction
from src.ordering.reading_order import sort_regions_reading_order


def format_regions_for_submission(page: PagePrediction) -> str:
    ordered_regions = sort_regions_reading_order(page.regions)
    payload = [region.to_submission_dict() for region in ordered_regions]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def validate_page_prediction(page: PagePrediction) -> None:
    if not page.image_id:
        raise ValueError("PagePrediction.image_id must be non-empty")
    for region in page.regions:
        x1, y1, x2, y2 = region.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox for image {page.image_id}: {region.bbox!r}")
        if not region.type:
            raise ValueError(f"Missing region type for image {page.image_id}: {region!r}")


def write_submission(pages: list[PagePrediction] | tuple[PagePrediction, ...], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "regions"])
        writer.writeheader()
        for page in sorted(pages, key=lambda item: item.image_id):
            validate_page_prediction(page)
            writer.writerow(
                {
                    "image": page.image_id,
                    "regions": format_regions_for_submission(page),
                }
            )
    return output
