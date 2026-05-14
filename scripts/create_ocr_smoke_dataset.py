from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from transformers import TrOCRProcessor

from src.experiments.logging import write_json
from src.ocr.crop_manifest import read_manifest, resolve_crop_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic lightweight OCR smoke dataset artifact.")
    parser.add_argument("--train-manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/val.jsonl")
    parser.add_argument("--output-dir", default="data/ocr_smoke/rukopys_ocr_smoke_v1")
    parser.add_argument("--model-name", default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--max-token-length", type=int, default=96)
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_len(row: dict[str, Any], tokenizer) -> int:
    return len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])


def select_rows(rows: list[dict[str, Any]], count: int, seed: int, tokenizer, max_token_length: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if token_len(row, tokenizer) <= max_token_length]
    eligible = sorted(eligible, key=lambda row: row["crop_id"])
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:count]


def copy_rows(rows: list[dict[str, Any]], source_manifest: str, output_dir: Path, split: str) -> list[dict[str, Any]]:
    manifest_rows = []
    for row in rows:
        source_path = resolve_crop_path(source_manifest, row)
        target_rel = Path("crops") / split / Path(row["crop_path"]).name
        target_path = output_dir / target_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        updated = dict(row)
        updated["split"] = split
        updated["crop_path"] = target_rel.as_posix()
        manifest_rows.append(updated)
    return sorted(manifest_rows, key=lambda row: row["crop_id"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def percentiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def zip_dataset(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))
    return zip_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    train_source = read_manifest(args.train_manifest)
    val_source = read_manifest(args.val_manifest)
    train_selected = select_rows(train_source, args.train_size, args.seed, tokenizer, args.max_token_length)
    val_selected = select_rows(val_source, args.val_size, args.seed + 1, tokenizer, args.max_token_length)

    train_rows = copy_rows(train_selected, args.train_manifest, output_dir, "train")
    val_rows = copy_rows(val_selected, args.val_manifest, output_dir, "val")
    write_jsonl(output_dir / "manifests" / "train.jsonl", train_rows)
    write_jsonl(output_dir / "manifests" / "val.jsonl", val_rows)
    write_jsonl(output_dir / "manifests" / "all.jsonl", train_rows + val_rows)

    all_rows = train_rows + val_rows
    token_lengths = [token_len(row, tokenizer) for row in all_rows]
    text_lengths = [len(row["text"]) for row in all_rows]
    stats = {
        "model_name": args.model_name,
        "seed": args.seed,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "max_token_length_policy": args.max_token_length,
        "token_length": percentiles(token_lengths),
        "text_length": percentiles(text_lengths),
        "source_counts": {
            split: {
                source: sum(1 for row in rows if row["source"] == source)
                for source in sorted({row["source"] for row in rows})
            }
            for split, rows in {"train": train_rows, "val": val_rows}.items()
        },
        "region_type_counts": {
            split: {
                region_type: sum(1 for row in rows if row["region_type"] == region_type)
                for region_type in sorted({row["region_type"] for row in rows})
            }
            for split, rows in {"train": train_rows, "val": val_rows}.items()
        },
    }
    write_json(output_dir / "stats" / "smoke_dataset_stats.json", stats)
    write_json(
        output_dir / "dataset_info.json",
        {
            "name": output_dir.name,
            "description": "Deterministic lightweight OCR smoke dataset derived from RUKOPYS gt_only/train crop export.",
            "train_manifest": "manifests/train.jsonl",
            "val_manifest": "manifests/val.jsonl",
            "stats": "stats/smoke_dataset_stats.json",
        },
    )

    zip_path = zip_dataset(output_dir) if args.zip else None
    checksums = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    write_json(output_dir / "checksums.json", checksums)
    if zip_path is not None:
        print(f"zip_path={zip_path}")
        print(f"zip_sha256={sha256_file(zip_path)}")
    print(f"output_dir={output_dir}")
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}")


if __name__ == "__main__":
    main()
