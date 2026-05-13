from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from src.analysis.qualitative import page_text_payload
from src.data.schema import PagePrediction, RegionPrediction
from src.detection.simple import detect_text_regions
from src.ocr.dummy import EmptyOCRRecognizer
from src.ocr.recognizer import OCRRecognizer
from src.ocr.trocr import TrOCRConfig, TrOCRRecognizer
from src.ordering.reading_order import sort_regions_reading_order
from src.preprocessing.image_ops import crop_region, ensure_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current OCR pipeline on one external image.")
    parser.add_argument("--image", default="text_photo.jpg")
    parser.add_argument("--output-dir", default="outputs/demo/external_image")
    parser.add_argument("--recognizer", choices=["empty", "trocr"], default="trocr")
    parser.add_argument("--trocr-model", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--crop-padding", type=int, default=8)
    return parser.parse_args()


def build_recognizer(args: argparse.Namespace) -> OCRRecognizer:
    if args.recognizer == "empty":
        return EmptyOCRRecognizer()
    return TrOCRRecognizer(
        TrOCRConfig(
            model_name=args.trocr_model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    )


def draw_detections(image: Image.Image, regions: tuple[RegionPrediction, ...], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = ensure_rgb(image).copy()
    draw = ImageDraw.Draw(canvas)
    for index, region in enumerate(regions, start=1):
        draw.rectangle(region.bbox, outline="red", width=4)
        x1, y1, _, _ = region.bbox
        label = str(index)
        label_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(
            (label_bbox[0] - 3, label_bbox[1] - 3, label_bbox[2] + 3, label_bbox[3] + 3),
            fill="red",
        )
        draw.text((x1, y1), label, fill="white")
    canvas.save(output_path)
    return output_path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    for path in crops_dir.glob("region_*.jpg"):
        path.unlink()
    for path in (
        output_dir / "detected_boxes.jpg",
        output_dir / "metadata.json",
        output_dir / "page_text.txt",
        output_dir / "page_texts.jsonl",
        output_dir / "predictions.jsonl",
        output_dir / "regions.json",
    ):
        if path.exists():
            path.unlink()

    image = ensure_rgb(Image.open(image_path))
    detections = detect_text_regions(image)
    recognizer = build_recognizer(args)

    regions: list[RegionPrediction] = []
    prediction_rows: list[dict] = []
    for index, detection in enumerate(detections, start=1):
        region = RegionPrediction(
            image_id=image_path.name,
            bbox=detection.bbox,
            type=detection.type,
            text="",
            score=detection.score,
        )
        crop = crop_region(image, region.bbox, padding=args.crop_padding)
        crop_path = crops_dir / f"region_{index:03d}.jpg"
        crop.save(crop_path)
        result = recognizer.recognize(crop, region)
        predicted_region = RegionPrediction(
            image_id=image_path.name,
            bbox=region.bbox,
            type=region.type,
            text=result.text,
            score=result.score,
        )
        regions.append(predicted_region)
        prediction_rows.append(
            {
                "image_id": image_path.name,
                "region_index": index,
                "bbox": list(region.bbox),
                "crop_path": str(crop_path),
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata,
            }
        )

    page = PagePrediction(
        image_id=image_path.name,
        width=image.width,
        height=image.height,
        regions=sort_regions_reading_order(regions),
    )
    page_text = page_text_payload(page)["page_text"]

    draw_detections(image, page.regions, output_dir / "detected_boxes.jpg")
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(output_dir / "page_texts.jsonl", [{"image_id": image_path.name, "predicted_text": page_text}])
    (output_dir / "page_text.txt").write_text(page_text + "\n", encoding="utf-8")
    (output_dir / "regions.json").write_text(
        json.dumps(page.to_submission_row(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "image_size": [image.width, image.height],
                "output_dir": str(output_dir),
                "recognizer": args.recognizer,
                "trocr_model": args.trocr_model if args.recognizer == "trocr" else None,
                "detector": "simple_cv",
                "regions": len(page.regions),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"image={image_path}")
    print(f"regions={len(page.regions)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
