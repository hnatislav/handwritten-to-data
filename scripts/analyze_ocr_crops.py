from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import TrOCRProcessor

from src.experiments.logging import write_json
from src.ocr.crop_manifest import read_manifest, resolve_crop_path, text_length_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exported OCR crop manifests.")
    parser.add_argument("--train-manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/val.jsonl")
    parser.add_argument("--model-name", default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--output-dir", default="data/ocr_crops/rukopys_gt_only_train_v1/stats")
    return parser.parse_args()


def percentiles(values: list[int | float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max", "mean"]}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def split_stats(rows: list[dict[str, Any]], manifest_path: str, tokenizer) -> dict[str, Any]:
    text_lengths = [len(row["text"]) for row in rows]
    token_lengths = [len(tokenizer(row["text"], add_special_tokens=False)["input_ids"]) for row in rows]
    unk_id = tokenizer.unk_token_id
    unk_rows = 0
    unk_tokens = 0
    for row in rows:
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        count = sum(1 for token_id in ids if token_id == unk_id)
        unk_tokens += count
        unk_rows += int(count > 0)

    crop_widths: list[int] = []
    crop_heights: list[int] = []
    crop_areas: list[int] = []
    for row in rows:
        with Image.open(resolve_crop_path(manifest_path, row)) as image:
            width, height = image.size
        crop_widths.append(width)
        crop_heights.append(height)
        crop_areas.append(width * height)

    return {
        "rows": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "text_length": percentiles(text_lengths),
        "token_length": percentiles(token_lengths),
        "tokenizer_unk": {
            "rows_with_unk": unk_rows,
            "unk_tokens": unk_tokens,
        },
        "crop_width": percentiles(crop_widths),
        "crop_height": percentiles(crop_heights),
        "crop_area": percentiles(crop_areas),
        "source": distribution(rows, "source"),
        "region_type": distribution(rows, "region_type"),
        "language": distribution(rows, "language"),
        "legibility": distribution(rows, "legibility"),
        "annotation_source": distribution(rows, "annotation_source"),
        "text_length_bucket": dict(sorted(Counter(text_length_bucket(row["text"]) for row in rows).items())),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# OCR Crop Analysis",
        "",
        f"Model/tokenizer: `{payload['model_name']}`",
        "",
        "## Summary",
        "",
        f"- Train rows: {payload['splits']['train']['rows']}",
        f"- Val rows: {payload['splits']['val']['rows']}",
        f"- Exact train/val text overlap strings: {payload['text_overlap']['unique_text_overlap_count']}",
        f"- Val rows whose text appears in train: {payload['text_overlap']['val_rows_with_train_text']}",
        "",
        "## Token Length Percentiles",
        "",
        "```json",
        json.dumps(
            {
                "train": payload["splits"]["train"]["token_length"],
                "val": payload["splits"]["val"]["token_length"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest)

    train_texts = Counter(row["text"] for row in train_rows)
    val_texts = Counter(row["text"] for row in val_rows)
    overlap = set(train_texts) & set(val_texts)
    val_rows_with_train_text = sum(count for text, count in val_texts.items() if text in overlap)

    payload = {
        "model_name": args.model_name,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "unk_token": tokenizer.unk_token,
            "unk_token_id": tokenizer.unk_token_id,
        },
        "manifests": {
            "train": args.train_manifest,
            "val": args.val_manifest,
        },
        "splits": {
            "train": split_stats(train_rows, args.train_manifest, tokenizer),
            "val": split_stats(val_rows, args.val_manifest, tokenizer),
        },
        "text_overlap": {
            "unique_text_overlap_count": len(overlap),
            "val_rows_with_train_text": val_rows_with_train_text,
            "top_overlaps": [
                {"text": text, "train_count": train_texts[text], "val_count": val_texts[text]}
                for text in sorted(overlap, key=lambda item: (val_texts[item] + train_texts[item], item), reverse=True)[:50]
            ],
        },
    }

    output_dir = Path(args.output_dir)
    write_json(output_dir / "ocr_crop_analysis.json", payload)
    write_markdown(output_dir / "ocr_crop_analysis.md", payload)
    print(f"analysis_json={output_dir / 'ocr_crop_analysis.json'}")
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)} overlap={len(overlap)}")


if __name__ == "__main__":
    main()
