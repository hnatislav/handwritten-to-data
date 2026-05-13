from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from src.evaluation.cer import character_error_rate, levenshtein_distance
from src.experiments.logging import write_experiment_metadata, write_json
from src.ocr.crop_manifest import read_manifest, resolve_crop_path, text_length_bucket
from src.ocr.trocr import TrOCRConfig, TrOCRRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR recognizer on exported crop manifest.")
    parser.add_argument("--manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/experiments/003_frozen_cyrillic_trocr_val_crops")
    parser.add_argument("--model-name", default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--hardest-k", type=int, default=50)
    return parser.parse_args()


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "mean_cer": None}
    edits = sum(row["edit_distance"] for row in rows)
    ref_len = sum(row["reference_length"] for row in rows)
    return {
        "rows": len(rows),
        "mean_cer": edits / ref_len if ref_len else 0.0,
        "unweighted_mean_cer": sum(row["cer"] for row in rows) / len(rows),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_hardest_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Hardest OCR Crop Examples", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['crop_id']}",
                "",
                f"- CER: {row['cer']:.6f}",
                f"- Source: {row.get('source')}",
                f"- Type: {row.get('region_type')}",
                f"- Crop: `{row['crop_path']}`",
                "",
                "Reference:",
                "",
                f"```text\n{row['reference_text']}\n```",
                "",
                "Prediction:",
                "",
                f"```text\n{row['predicted_text']}\n```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest, limit=args.limit)
    recognizer = TrOCRRecognizer(
        TrOCRConfig(
            model_name=args.model_name,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    )

    predictions: list[dict[str, Any]] = []
    for row in rows:
        with Image.open(resolve_crop_path(args.manifest, row)) as image:
            result = recognizer.recognize(image.convert("RGB"), region=None)  # type: ignore[arg-type]
        reference = row["text"]
        prediction = result.text
        edit_distance = levenshtein_distance(reference, prediction)
        predictions.append(
            {
                **row,
                "reference_text": reference,
                "predicted_text": prediction,
                "edit_distance": edit_distance,
                "reference_length": len(reference),
                "cer": character_error_rate(reference, prediction),
                "text_length_bucket": text_length_bucket(reference),
            }
        )

    grouped: dict[str, dict[str, Any]] = {}
    for key in ["source", "region_type", "language", "annotation_source", "text_length_bucket"]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            buckets[str(row.get(key))].append(row)
        grouped[key] = {bucket: summarize_group(items) for bucket, items in sorted(buckets.items())}

    metrics = {
        "rows": len(predictions),
        "model_name": args.model_name,
        "manifest": args.manifest,
        "overall": summarize_group(predictions),
        "grouped": grouped,
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    hardest = sorted(predictions, key=lambda item: item["cer"], reverse=True)[: args.hardest_k]
    write_jsonl(output_dir / "hardest_examples.jsonl", hardest)
    write_hardest_markdown(output_dir / "hardest_examples.md", hardest[: min(args.hardest_k, 20)])
    write_experiment_metadata(
        output_dir,
        config=vars(args),
        dataset_info={"manifest": args.manifest},
        metrics=metrics,
    )
    print(f"rows={metrics['rows']}")
    print(f"mean_cer={metrics['overall']['mean_cer']:.6f}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
