from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from transformers import TrOCRProcessor

from src.experiments.logging import write_json
from src.ocr.crop_manifest import read_manifest, resolve_crop_path, text_length_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic stage-1 OCR training artifact.")
    parser.add_argument("--train-manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/val.jsonl")
    parser.add_argument("--output-dir", default="data/ocr_stage1/rukopys_ocr_stage1_v1")
    parser.add_argument("--model-name", default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--max-token-length", type=int, default=128)
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["crop_id"])
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def token_len(row: dict[str, Any], tokenizer) -> int:
    return len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])


def exact_text(row: dict[str, Any]) -> str:
    return str(row.get("text", "")).strip()


def annotate_rows(rows: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        updated = dict(row)
        updated["exact_text"] = exact_text(row)
        updated["token_length"] = token_len(row, tokenizer)
        updated["raw_text_length"] = len(str(row.get("text", "")))
        updated["text_length_bucket"] = text_length_bucket(str(row.get("text", "")))
        annotated.append(updated)
    return annotated


def percentiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def filter_rows(rows: list[dict[str, Any]], max_token_length: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    dropped = []
    for row in rows:
        if not row["exact_text"]:
            dropped.append({**row, "drop_reason": "empty_text"})
        elif row["token_length"] > max_token_length:
            dropped.append({**row, "drop_reason": "over_max_token_length"})
        else:
            kept.append(row)
    return kept, dropped


def select_validation(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    selected = []
    used_texts: set[str] = set()
    for row in stable_shuffle(rows, seed):
        text = row["exact_text"]
        if text in used_texts:
            continue
        selected.append(row)
        used_texts.add(text)
        if len(selected) >= size:
            break
    return sorted(selected, key=lambda row: row["crop_id"])


def select_train(rows: list[dict[str, Any]], size: int, seed: int, blocked_texts: set[str]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["exact_text"] not in blocked_texts]
    return sorted(stable_shuffle(eligible, seed)[:size], key=lambda row: row["crop_id"])


def copy_rows(rows: list[dict[str, Any]], source_manifest: str, output_dir: Path, split: str) -> list[dict[str, Any]]:
    manifest_rows = []
    for row in rows:
        source_path = resolve_crop_path(source_manifest, row)
        target_rel = Path("crops") / split / Path(row["crop_path"]).name
        target_path = output_dir / target_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        updated = dict(row)
        updated["selected_split"] = split
        updated["source_manifest"] = source_manifest
        updated["crop_path"] = target_rel.as_posix()
        manifest_rows.append(updated)
    return sorted(manifest_rows, key=lambda row: row["crop_id"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def zip_dataset(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))
    return zip_path


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "token_length": percentiles([row["token_length"] for row in rows]),
        "raw_text_length": percentiles([row["raw_text_length"] for row in rows]),
        "source_counts": count_by(rows, "source"),
        "region_type_counts": count_by(rows, "region_type"),
        "language_counts": count_by(rows, "language"),
        "annotation_source_counts": count_by(rows, "annotation_source"),
        "text_length_bucket_counts": count_by(rows, "text_length_bucket"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer

    raw_train = annotate_rows(read_manifest(args.train_manifest), tokenizer)
    raw_val = annotate_rows(read_manifest(args.val_manifest), tokenizer)
    eligible_train, dropped_train = filter_rows(raw_train, args.max_token_length)
    eligible_val, dropped_val = filter_rows(raw_val, args.max_token_length)

    selected_val = select_validation(eligible_val, args.val_size, args.seed + 1)
    val_texts = {row["exact_text"] for row in selected_val}
    selected_train = select_train(eligible_train, args.train_size, args.seed, val_texts)

    if len(selected_train) < args.train_size:
        raise ValueError(f"Selected {len(selected_train)} train rows, requested {args.train_size}")
    if len(selected_val) < args.val_size:
        raise ValueError(f"Selected {len(selected_val)} val rows, requested {args.val_size}")

    train_rows = copy_rows(selected_train, args.train_manifest, output_dir, "train")
    val_rows = copy_rows(selected_val, args.val_manifest, output_dir, "val")
    write_jsonl(output_dir / "manifests" / "train.jsonl", train_rows)
    write_jsonl(output_dir / "manifests" / "val.jsonl", val_rows)
    write_jsonl(output_dir / "manifests" / "all.jsonl", train_rows + val_rows)

    dropped_for_length = [
        row
        for row in dropped_train + dropped_val
        if row["drop_reason"] == "over_max_token_length"
    ]
    write_jsonl(
        output_dir / "stats" / "dropped_overlength_examples.jsonl",
        sorted(dropped_for_length, key=lambda row: (-row["token_length"], row["crop_id"]))[:200],
    )

    selected_train_texts = {row["exact_text"] for row in train_rows}
    selected_val_texts = {row["exact_text"] for row in val_rows}
    raw_overlap = {row["exact_text"] for row in raw_train if row["exact_text"]} & {
        row["exact_text"] for row in raw_val if row["exact_text"]
    }
    eligible_overlap = {row["exact_text"] for row in eligible_train} & {row["exact_text"] for row in eligible_val}
    selected_overlap = selected_train_texts & selected_val_texts

    stats = {
        "name": output_dir.name,
        "model_name": args.model_name,
        "seed": args.seed,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "requested_train_size": args.train_size,
        "requested_val_size": args.val_size,
        "selected_train_size": len(train_rows),
        "selected_val_size": len(val_rows),
        "target_length_policy": {
            "max_token_length": args.max_token_length,
            "silent_truncation_allowed": False,
            "dropped_over_max_token_length": len(dropped_for_length),
            "train_coverage_pct": 100.0 * len(eligible_train) / len(raw_train) if raw_train else None,
            "val_coverage_pct": 100.0 * len(eligible_val) / len(raw_val) if raw_val else None,
            "selected_train_overflow_count": sum(row["token_length"] > args.max_token_length for row in train_rows),
            "selected_val_overflow_count": sum(row["token_length"] > args.max_token_length for row in val_rows),
        },
        "overlap": {
            "raw_exact_text_overlap_count": len(raw_overlap),
            "eligible_exact_text_overlap_count": len(eligible_overlap),
            "selected_exact_text_overlap_count": len(selected_overlap),
            "train_rows_removed_by_selected_val_texts": sum(row["exact_text"] in val_texts for row in eligible_train),
        },
        "input": {
            "train_rows": len(raw_train),
            "val_rows": len(raw_val),
            "train_dropped": count_by(dropped_train, "drop_reason"),
            "val_dropped": count_by(dropped_val, "drop_reason"),
            "train_token_length": percentiles([row["token_length"] for row in raw_train]),
            "val_token_length": percentiles([row["token_length"] for row in raw_val]),
        },
        "selected": {
            "train": split_stats(train_rows),
            "val": split_stats(val_rows),
        },
    }
    write_json(output_dir / "stats" / "stage1_dataset_stats.json", stats)
    write_json(
        output_dir / "dataset_info.json",
        {
            "name": output_dir.name,
            "description": "Deterministic stage-1 OCR subset with train/val exact-text deduplication.",
            "train_manifest": "manifests/train.jsonl",
            "val_manifest": "manifests/val.jsonl",
            "stats": "stats/stage1_dataset_stats.json",
            "filtering_policy": {
                "max_token_length": args.max_token_length,
                "empty_text": "drop",
                "over_max_token_length": "drop",
                "val_selection": "deterministic unique exact text",
                "train_selection": "deterministic, excluding selected validation exact texts",
            },
        },
    )
    (output_dir / "FILTERING_POLICY.md").write_text(
        "\n".join(
            [
                "# Stage-1 OCR Filtering Policy",
                "",
                f"- Tokenizer/model: `{args.model_name}`",
                f"- Max target token length: `{args.max_token_length}`",
                "- Empty text rows are dropped.",
                "- Rows above max target token length are dropped, not truncated.",
                "- Validation rows are selected first with unique exact text.",
                "- Training rows exclude exact texts selected for validation.",
                "- Sampling is deterministic by `crop_id` plus fixed seed.",
            ]
        ),
        encoding="utf-8",
    )

    checksums = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    write_json(output_dir / "checksums.json", checksums)
    zip_path = zip_dataset(output_dir) if args.zip else None
    if zip_path is not None:
        print(f"zip_path={zip_path}")
        print(f"zip_sha256={sha256_file(zip_path)}")
    print(f"output_dir={output_dir}")
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}")
    print(f"selected_exact_text_overlap_count={len(selected_overlap)}")


if __name__ == "__main__":
    main()
