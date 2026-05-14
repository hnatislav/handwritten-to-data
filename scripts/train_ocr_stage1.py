from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.evaluation.cer import character_error_rate, levenshtein_distance
from src.experiments.logging import runtime_environment, write_experiment_metadata, write_json
from src.ocr.crop_manifest import read_manifest, resolve_crop_path, text_length_bucket
from src.ocr.trocr import TrOCRConfig, TrOCRRecognizer
from src.analysis.ocr_experiment import enrich_prediction, grouped_metrics, summarize_group


class OCRCropDataset(Dataset):
    def __init__(self, manifest_path: str, rows: list[dict[str, Any]], processor: TrOCRProcessor, max_target_length: int) -> None:
        self.manifest_path = manifest_path
        self.rows = rows
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(resolve_crop_path(self.manifest_path, row)) as image:
            pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": row["text"],
            "crop_id": row["crop_id"],
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "text": [item["text"] for item in batch],
        "crop_id": [item["crop_id"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled stage-1 OCR fine-tuning experiment.")
    parser.add_argument("--train-manifest", default="data/ocr_stage1/rukopys_ocr_stage1_v1/manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="data/ocr_stage1/rukopys_ocr_stage1_v1/manifests/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/experiments/010_ocr_stage1_controlled")
    parser.add_argument("--model-name", default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--hardest-k", type=int, default=50)
    parser.add_argument("--sample-k", type=int, default=40)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def token_lengths(rows: list[dict[str, Any]], processor: TrOCRProcessor) -> list[int]:
    return [len(processor.tokenizer(row["text"], add_special_tokens=False)["input_ids"]) for row in rows]


def percentile_summary(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    def pct(q: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
        return float(ordered[index])
    return {
        "min": float(ordered[0]),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": float(ordered[-1]),
    }


def validate_target_lengths(
    rows: list[dict[str, Any]],
    processor: TrOCRProcessor,
    max_target_length: int,
    allow_truncation: bool,
) -> dict[str, Any]:
    lengths = token_lengths(rows, processor)
    too_long = [
        {"crop_id": row["crop_id"], "token_length": length, "text": row["text"]}
        for row, length in zip(rows, lengths, strict=True)
        if length > max_target_length
    ]
    if too_long and not allow_truncation:
        raise ValueError(
            f"{len(too_long)} rows exceed max_target_length={max_target_length}. "
            f"Use the stage-1 dataset exporter to drop overlength rows, or pass --allow-truncation only for diagnostics."
        )
    return {
        "rows": len(rows),
        "max_target_length": max_target_length,
        "token_length": percentile_summary(lengths),
        "truncated_samples": len(too_long) if allow_truncation else 0,
        "dropped_samples": 0,
        "coverage_pct": 100.0 * (len(rows) - len(too_long)) / len(rows) if rows else None,
        "truncation_allowed": allow_truncation,
        "overflow_examples": too_long[:20],
    }


def fixed_batch_loss(model, dataset: OCRCropDataset, device: torch.device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    with torch.inference_mode():
        outputs = model(pixel_values=batch["pixel_values"].to(device), labels=batch["labels"].to(device))
    model.train()
    return float(outputs.loss.detach().cpu())


def has_repeated_token_failure(text: str) -> bool:
    tokens = re.findall(r"\S+", text)
    if len(tokens) >= 4:
        token_counts = Counter(tokens)
        if any(count >= 3 for count in token_counts.values()):
            return True
    chars = re.sub(r"\s+", "", text)
    return bool(re.search(r"(.)\1{5,}", chars))


def infer_handwriting_bucket(row: dict[str, Any]) -> str:
    fields = " ".join(str(row.get(key, "")) for key in ("source", "region_type", "annotation_source", "legibility")).lower()
    if "print" in fields or "printed" in fields:
        return "printed"
    if "hand" in fields or "written" in fields or "manuscript" in fields:
        return "handwritten"
    return "unknown"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evaluate_checkpoint(
    checkpoint_dir: Path,
    val_manifest: str,
    val_rows: list[dict[str, Any]],
    device: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    recognizer = TrOCRRecognizer(
        TrOCRConfig(
            model_name=str(checkpoint_dir),
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )
    )
    predictions = []
    for row in val_rows:
        with Image.open(resolve_crop_path(val_manifest, row)) as image:
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
                "handwriting_bucket": infer_handwriting_bucket(row),
                "predicted_repeated_token_failure": has_repeated_token_failure(prediction),
                "reference_token_length": int(row.get("token_length", 0)),
            }
        )
    return predictions


def write_error_analysis(
    output_dir: Path,
    predictions: list[dict[str, Any]],
    hardest_k: int,
    sample_k: int,
    max_target_length: int,
) -> dict[str, Any]:
    enriched = [enrich_prediction(row, max_target_length=max_target_length) for row in predictions]
    hardest = sorted(enriched, key=lambda row: (row["cer"], row["reference_length"]), reverse=True)[:hardest_k]
    longest_failures = sorted(
        [row for row in enriched if row["cer"] > 0],
        key=lambda row: (row["reference_length"], row["cer"]),
        reverse=True,
    )[:hardest_k]
    repeated_failures = [
        row for row in enriched if "repeated_token_degeneration" in row["error_buckets"]
    ][:hardest_k]
    truncation_failures = [
        row
        for row in enriched
        if "truncation_failure" in row["error_buckets"]
    ][:hardest_k]
    representative = sorted(enriched, key=lambda row: (str(row.get("source")), row["crop_id"]))[:sample_k]
    write_jsonl(output_dir / "predictions.jsonl", enriched)
    write_jsonl(output_dir / "hardest_examples.jsonl", hardest)
    write_jsonl(output_dir / "longest_failures.jsonl", longest_failures)
    write_jsonl(output_dir / "repeated_token_failures.jsonl", repeated_failures)
    write_jsonl(output_dir / "truncation_failures.jsonl", truncation_failures)
    write_jsonl(output_dir / "sample_predictions.jsonl", representative)
    write_jsonl(output_dir / "representative_samples.jsonl", representative)
    return {
        "hardest_examples": len(hardest),
        "longest_failures": len(longest_failures),
        "repeated_token_failures": len(repeated_failures),
        "truncation_failures": len(truncation_failures),
        "representative_samples": len(representative),
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()
    set_determinism(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"

    train_rows = read_manifest(args.train_manifest, limit=args.train_limit)
    val_rows = read_manifest(args.val_manifest, limit=args.eval_limit)
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.to(device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    target_policy = {
        "train": validate_target_lengths(train_rows, processor, args.max_target_length, args.allow_truncation),
        "val": validate_target_lengths(val_rows, processor, args.max_target_length, args.allow_truncation),
    }
    train_dataset = OCRCropDataset(args.train_manifest, train_rows, processor, args.max_target_length)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    probe_loss_before = fixed_batch_loss(model, train_dataset, device, args.batch_size)
    epoch_metrics = []
    all_losses: list[float] = []
    for epoch_index in range(args.epochs):
        epoch_losses = []
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=batch["pixel_values"].to(device), labels=batch["labels"].to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            epoch_losses.append(loss_value)
            all_losses.append(loss_value)
        epoch_metrics.append(
            {
                "epoch": epoch_index + 1,
                "batches": len(epoch_losses),
                "mean_train_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else None,
                "last_train_loss": epoch_losses[-1] if epoch_losses else None,
            }
        )

    probe_loss_after = fixed_batch_loss(model, train_dataset, device, args.batch_size)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    predictions = evaluate_checkpoint(checkpoint_dir, args.val_manifest, val_rows, args.device, args.max_new_tokens)
    analysis_counts = write_error_analysis(
        output_dir,
        predictions,
        args.hardest_k,
        args.sample_k,
        args.max_target_length,
    )
    runtime = runtime_environment()
    if device.type == "cuda":
        runtime["cuda_memory"] = {
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "max_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        }

    enriched_predictions = [
        enrich_prediction(row, max_target_length=args.max_target_length)
        for row in predictions
    ]
    metrics = {
        "train_rows": len(train_rows),
        "val_rows_evaluated": len(val_rows),
        "epochs": args.epochs,
        "first_train_batch_loss": all_losses[0] if all_losses else None,
        "last_train_batch_loss": all_losses[-1] if all_losses else None,
        "probe_loss_before": probe_loss_before,
        "probe_loss_after": probe_loss_after,
        "loss_decreased": probe_loss_after < probe_loss_before,
        "epoch_metrics": epoch_metrics,
        "target_length_policy": target_policy,
        "overall": summarize_group(enriched_predictions),
        "grouped": grouped_metrics(
            enriched_predictions,
            (
                "source",
                "region_type",
                "language",
                "annotation_source",
                "text_length_bucket",
                "handwriting_bucket",
                "error_buckets",
            ),
        ),
        "error_analysis": analysis_counts,
        "runtime_seconds": time.time() - start_time,
        "runtime": runtime,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "runtime.json", runtime)
    write_json(output_dir / "config_snapshot.json", vars(args))
    write_json(output_dir / "environment_snapshot.json", runtime)
    write_experiment_metadata(
        output_dir,
        config=vars(args),
        dataset_info={
            "train_manifest": args.train_manifest,
            "val_manifest": args.val_manifest,
            "train_rows": len(train_rows),
            "val_rows_evaluated": len(val_rows),
        },
        metrics=metrics,
    )
    print(f"train_rows={metrics['train_rows']}")
    print(f"val_rows_evaluated={metrics['val_rows_evaluated']}")
    print(f"probe_loss_before={metrics['probe_loss_before']}")
    print(f"probe_loss_after={metrics['probe_loss_after']}")
    print(f"loss_decreased={metrics['loss_decreased']}")
    print(f"mean_cer={metrics['overall']['mean_cer']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
