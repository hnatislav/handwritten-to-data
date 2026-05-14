from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

from transformers import TrOCRProcessor

from src.analysis.ocr_experiment import unicode_summary, write_json, write_jsonl
from src.ocr.crop_manifest import read_manifest, resolve_crop_path, text_length_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic tiny OCR overfit diagnostic dataset.")
    parser.add_argument("--source-manifest", default="data/ocr_stage1/rukopys_ocr_stage1_v1/manifests/train.jsonl")
    parser.add_argument("--output-dir", default="data/ocr_overfit/rukopys_ocr_overfit_v1")
    parser.add_argument("--model-name", default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--val-size", type=int, default=8)
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


def token_ids(text: str, tokenizer) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def annotate_row(row: dict[str, Any], tokenizer) -> dict[str, Any]:
    text = str(row.get("text", ""))
    ids = token_ids(text, tokenizer)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    return {
        **row,
        "token_length": len(ids),
        "raw_text_length": len(text),
        "text_length_bucket": text_length_bucket(text),
        "unicode_replacement_char_count": unicode_summary(text)["replacement_char_count"],
        "tokenizer_roundtrip_equal": decoded == text,
        "tokenizer_roundtrip_decoded": decoded,
    }


def copy_rows(rows: list[dict[str, Any]], source_manifest: str, output_dir: Path, split: str) -> list[dict[str, Any]]:
    output_rows = []
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
        output_rows.append(updated)
    return sorted(output_rows, key=lambda item: item["crop_id"])


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
    rows = [
        annotate_row(row, tokenizer)
        for row in read_manifest(args.source_manifest)
        if str(row.get("text", "")).strip()
    ]
    rows = [row for row in rows if row["token_length"] <= args.max_token_length]
    selected_train = stable_shuffle(rows, args.seed)[: args.train_size]
    selected_val = selected_train[: args.val_size]

    if len(selected_train) < args.train_size:
        raise ValueError(f"Selected {len(selected_train)} train rows, requested {args.train_size}")
    if len(selected_val) < args.val_size:
        raise ValueError(f"Selected {len(selected_val)} val rows, requested {args.val_size}")

    train_rows = copy_rows(selected_train, args.source_manifest, output_dir, "train")
    val_rows = copy_rows(selected_val, args.source_manifest, output_dir, "val")
    write_jsonl(output_dir / "manifests" / "train.jsonl", train_rows)
    write_jsonl(output_dir / "manifests" / "val.jsonl", val_rows)
    write_jsonl(output_dir / "manifests" / "all.jsonl", train_rows + val_rows)

    reference_rows = []
    roundtrip_rows = []
    for split, split_rows in (("train", train_rows), ("val", val_rows)):
        for row in split_rows:
            text = row["text"]
            ids = token_ids(text, tokenizer)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            reference_rows.append(
                {
                    "split": split,
                    "crop_id": row["crop_id"],
                    "text": text,
                    "text_repr": repr(text),
                    "unicode": unicode_summary(text),
                }
            )
            roundtrip_rows.append(
                {
                    "split": split,
                    "crop_id": row["crop_id"],
                    "text": text,
                    "text_repr": repr(text),
                    "token_ids": ids,
                    "decoded": decoded,
                    "decoded_repr": repr(decoded),
                    "roundtrip_equal": decoded == text,
                    "decoded_unicode": unicode_summary(decoded),
                }
            )
    write_jsonl(output_dir / "diagnostics" / "reference_integrity.jsonl", reference_rows)
    write_jsonl(output_dir / "diagnostics" / "tokenizer_roundtrip.jsonl", roundtrip_rows)

    train_token_lengths = [row["token_length"] for row in train_rows]
    val_token_lengths = [row["token_length"] for row in val_rows]
    stats = {
        "name": output_dir.name,
        "model_name": args.model_name,
        "seed": args.seed,
        "source_manifest": args.source_manifest,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "val_is_train_subset": True,
        "max_token_length": args.max_token_length,
        "train_token_length": {
            "min": min(train_token_lengths),
            "max": max(train_token_lengths),
            "mean": sum(train_token_lengths) / len(train_token_lengths),
        },
        "val_token_length": {
            "min": min(val_token_lengths),
            "max": max(val_token_lengths),
            "mean": sum(val_token_lengths) / len(val_token_lengths),
        },
        "reference_replacement_char_count": sum(row["unicode"]["replacement_char_count"] for row in reference_rows),
        "tokenizer_roundtrip_failures": sum(not row["roundtrip_equal"] for row in roundtrip_rows),
        "special_tokens": {
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
            "decoder_start_token_id": getattr(processor, "decoder_start_token_id", None),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "vocab_size": len(tokenizer),
        },
    }
    write_json(output_dir / "stats" / "overfit_dataset_stats.json", stats)
    write_json(
        output_dir / "dataset_info.json",
        {
            "name": output_dir.name,
            "description": "Tiny deterministic OCR overfit diagnostic dataset. Validation intentionally reuses train rows.",
            "train_manifest": "manifests/train.jsonl",
            "val_manifest": "manifests/val.jsonl",
            "stats": "stats/overfit_dataset_stats.json",
        },
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
    print(f"reference_replacement_char_count={stats['reference_replacement_char_count']}")
    print(f"tokenizer_roundtrip_failures={stats['tokenizer_roundtrip_failures']}")


if __name__ == "__main__":
    main()
