from __future__ import annotations

import difflib
import html
from pathlib import Path

from PIL import Image, ImageDraw

from src.data.schema import PagePrediction, RegionPrediction
from src.evaluation.metrics import PageTextEvaluation, assemble_page_text
from src.ordering.reading_order import sort_regions_reading_order


def text_diff(reference: str, prediction: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            reference.splitlines(),
            prediction.splitlines(),
            fromfile="gt",
            tofile="pred",
            lineterm="",
        )
    )


def write_page_text_comparison(
    evaluation: PageTextEvaluation,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"# {evaluation.image_id}\n\n"
        f"- PageCER: {evaluation.page_cer:.6f}\n"
        f"- Edit distance: {evaluation.edit_distance}\n"
        f"- Reference length: {evaluation.reference_length}\n\n"
        "## Reference\n\n"
        f"```text\n{evaluation.reference_text}\n```\n\n"
        "## Prediction\n\n"
        f"```text\n{evaluation.predicted_text}\n```\n\n"
        "## Diff\n\n"
        f"```diff\n{text_diff(evaluation.reference_text, evaluation.predicted_text)}\n```\n"
    )
    output.write_text(payload, encoding="utf-8")
    return output


def draw_region_errors(
    image: Image.Image,
    reference_page: PagePrediction,
    prediction_page: PagePrediction,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    pred_by_bbox = {region.bbox: region for region in prediction_page.regions}

    for ref in sort_regions_reading_order(reference_page.regions):
        pred = pred_by_bbox.get(ref.bbox)
        is_match = pred is not None and ref.text == pred.text
        color = "green" if is_match else "red"
        draw.rectangle(ref.bbox, outline=color, width=4)
        label = "ok" if is_match else "err"
        draw.text((ref.bbox[0], max(0, ref.bbox[1] - 14)), label, fill=color)

    canvas.save(output)
    return output


def write_hardest_examples(
    evaluations: list[PageTextEvaluation],
    output_dir: str | Path,
    top_k: int = 5,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in sorted(evaluations, key=lambda eval_item: eval_item.page_cer, reverse=True)[:top_k]:
        safe_id = Path(item.image_id).stem
        paths.append(write_page_text_comparison(item, output / f"{safe_id}.md"))
    return paths


def page_text_payload(page: PagePrediction) -> dict[str, str]:
    return {
        "image_id": page.image_id,
        "page_text": assemble_page_text(page.regions),
    }
