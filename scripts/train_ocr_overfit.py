from __future__ import annotations

import argparse
import copy
import json
import random
import time
import unicodedata
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.analysis.ocr_experiment import (
    enrich_prediction,
    grouped_metrics,
    has_repeated_token_failure,
    summarize_group,
    unicode_summary,
    write_json,
    write_jsonl,
)
from src.evaluation.cer import character_error_rate, levenshtein_distance
from src.experiments.logging import runtime_environment, write_experiment_metadata
from src.ocr.crop_manifest import read_manifest, resolve_crop_path, text_length_bucket


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
    parser = argparse.ArgumentParser(description="Tiny OCR overfit diagnostic with tokenizer/decode inspection.")
    parser.add_argument("--train-manifest", default="data/ocr_overfit/rukopys_ocr_overfit_v1/manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="data/ocr_overfit/rukopys_ocr_overfit_v1/manifests/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/experiments/020_ocr_overfit_v1")
    parser.add_argument("--model-name", default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--sample-k", type=int, default=8)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generation_config(model, max_new_tokens: int):
    config = copy.deepcopy(model.generation_config)
    config.max_length = max_new_tokens
    config.max_new_tokens = None
    config.num_beams = 1
    config.do_sample = False
    config.early_stopping = False
    config.length_penalty = None
    model.generation_config = copy.deepcopy(config)
    return config


def tokenizer_diagnostics(processor: TrOCRProcessor, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    probes = [
        "Україна",
        "Є ї І і Ґ ґ",
        "абвгдеєжзиіїйклмнопрстуфхцчшщьюя",
        "1234567890.,:;!?-",
    ]
    probe_rows = []
    for text in probes:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        probe_rows.append(
            {
                "text": text,
                "text_repr": repr(text),
                "token_ids": token_ids,
                "decoded": decoded,
                "decoded_repr": repr(decoded),
                "roundtrip_equal": decoded == text,
                "unicode": unicode_summary(decoded),
            }
        )
    row_checks = []
    for row in rows:
        text = row["text"]
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        row_checks.append(
            {
                "crop_id": row["crop_id"],
                "text": text,
                "text_repr": repr(text),
                "token_ids": token_ids,
                "decoded": decoded,
                "decoded_repr": repr(decoded),
                "roundtrip_equal": decoded == text,
                "reference_unicode": unicode_summary(text),
                "decoded_unicode": unicode_summary(decoded),
            }
        )
    return {
        "special_tokens": {
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "vocab_size": len(tokenizer),
        },
        "probe_roundtrip": probe_rows,
        "manifest_roundtrip_failures": [row for row in row_checks if not row["roundtrip_equal"]],
        "reference_replacement_char_count": sum(
            item["reference_unicode"]["replacement_char_count"] for item in row_checks
        ),
    }


def evaluate_rows(
    model,
    processor: TrOCRProcessor,
    manifest_path: str,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    max_target_length: int,
) -> list[dict[str, Any]]:
    model.eval()
    gen_config = generation_config(model, max_new_tokens)
    predictions = []
    with torch.inference_mode():
        for row in rows:
            with Image.open(resolve_crop_path(manifest_path, row)) as image:
                pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
            generated = model.generate(pixel_values, generation_config=gen_config)
            token_ids = generated[0].detach().cpu().tolist()
            decoded_raw = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
            decoded_clean = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            decoded_nfc = unicodedata.normalize("NFC", decoded_clean).strip()
            reference = row["text"]
            prediction = decoded_nfc
            predictions.append(
                enrich_prediction(
                    {
                        **row,
                        "reference_text": reference,
                        "predicted_text": prediction,
                        "edit_distance": levenshtein_distance(reference, prediction),
                        "reference_length": len(reference),
                        "cer": character_error_rate(reference, prediction),
                        "text_length_bucket": text_length_bucket(reference),
                        "reference_token_length": int(row.get("token_length", 0)),
                        "raw_generated_token_ids": token_ids,
                        "decoded_raw": decoded_raw,
                        "decoded_raw_repr": repr(decoded_raw),
                        "decoded_clean": decoded_clean,
                        "decoded_clean_repr": repr(decoded_clean),
                        "decoded_nfc": decoded_nfc,
                        "decoded_nfc_repr": repr(decoded_nfc),
                        "decoded_raw_unicode": unicode_summary(decoded_raw),
                        "decoded_clean_unicode": unicode_summary(decoded_clean),
                        "decoded_nfc_unicode": unicode_summary(decoded_nfc),
                        "reference_unicode": unicode_summary(reference),
                        "decoder_degenerated": has_repeated_token_failure(prediction),
                        "empty_prediction": not prediction.strip(),
                    },
                    max_target_length=max_target_length,
                )
            )
    model.train()
    return predictions


def fixed_loss(model, dataset: OCRCropDataset, device: torch.device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    with torch.inference_mode():
        outputs = model(pixel_values=batch["pixel_values"].to(device), labels=batch["labels"].to(device))
    model.train()
    return float(outputs.loss.detach().cpu())


def write_prediction_artifacts(output_dir: Path, prefix: str, predictions: list[dict[str, Any]], sample_k: int) -> None:
    write_jsonl(output_dir / f"{prefix}_predictions.jsonl", predictions)
    write_jsonl(
        output_dir / f"{prefix}_problematic_token_dumps.jsonl",
        [
            row
            for row in predictions
            if row["decoded_nfc_unicode"]["replacement_char_count"]
            or row["decoded_raw_unicode"]["replacement_char_count"]
            or row["decoder_degenerated"]
            or row["empty_prediction"]
        ],
    )
    write_jsonl(
        output_dir / f"{prefix}_sample_predictions.jsonl",
        sorted(predictions, key=lambda row: str(row["crop_id"]))[:sample_k],
    )


def main() -> None:
    args = parse_args()
    start = time.time()
    set_determinism(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest)
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    write_json(output_dir / "tokenizer_diagnostics.json", tokenizer_diagnostics(processor, train_rows + val_rows))

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.to(device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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

    probe_loss_before = fixed_loss(model, train_dataset, device, args.batch_size)
    epoch_metrics = []
    eval_epochs = {1, args.epochs}
    eval_epochs.update(range(args.eval_every, args.epochs + 1, args.eval_every))
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=batch["pixel_values"].to(device), labels=batch["labels"].to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        item: dict[str, Any] = {
            "epoch": epoch,
            "mean_train_loss": sum(losses) / len(losses) if losses else None,
            "last_train_loss": losses[-1] if losses else None,
        }
        if epoch in eval_epochs:
            train_predictions = evaluate_rows(
                model, processor, args.train_manifest, train_rows, device, args.max_new_tokens, args.max_target_length
            )
            val_predictions = evaluate_rows(
                model, processor, args.val_manifest, val_rows, device, args.max_new_tokens, args.max_target_length
            )
            item["train"] = summarize_group(train_predictions)
            item["val"] = summarize_group(val_predictions)
            write_prediction_artifacts(output_dir / "epoch_dumps", f"epoch_{epoch:03d}_train", train_predictions, args.sample_k)
            write_prediction_artifacts(output_dir / "epoch_dumps", f"epoch_{epoch:03d}_val", val_predictions, args.sample_k)
        epoch_metrics.append(item)

    probe_loss_after = fixed_loss(model, train_dataset, device, args.batch_size)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    final_train_predictions = evaluate_rows(
        model, processor, args.train_manifest, train_rows, device, args.max_new_tokens, args.max_target_length
    )
    final_val_predictions = evaluate_rows(
        model, processor, args.val_manifest, val_rows, device, args.max_new_tokens, args.max_target_length
    )
    write_prediction_artifacts(output_dir, "train", final_train_predictions, args.sample_k)
    write_prediction_artifacts(output_dir, "val", final_val_predictions, args.sample_k)
    write_jsonl(output_dir / "predictions.jsonl", final_val_predictions)
    write_jsonl(output_dir / "sample_predictions.jsonl", sorted(final_val_predictions, key=lambda row: row["crop_id"])[: args.sample_k])
    write_jsonl(output_dir / "hardest_examples.jsonl", sorted(final_val_predictions, key=lambda row: row["cer"], reverse=True))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reload_model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    reload_model.to(device)
    reload_predictions = evaluate_rows(
        reload_model, processor, args.val_manifest, val_rows, device, args.max_new_tokens, args.max_target_length
    )
    write_jsonl(output_dir / "reload_val_predictions.jsonl", reload_predictions)

    runtime = runtime_environment()
    if device.type == "cuda":
        runtime["cuda_memory"] = {
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "max_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        }
    metrics = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "epochs": args.epochs,
        "probe_loss_before": probe_loss_before,
        "probe_loss_after": probe_loss_after,
        "loss_decreased": probe_loss_after < probe_loss_before,
        "epoch_metrics": epoch_metrics,
        "final_train": summarize_group(final_train_predictions),
        "final_val": summarize_group(final_val_predictions),
        "reload_val": summarize_group(reload_predictions),
        "final_train_grouped": grouped_metrics(final_train_predictions, ("error_buckets", "text_length_bucket", "source", "region_type")),
        "final_val_grouped": grouped_metrics(final_val_predictions, ("error_buckets", "text_length_bucket", "source", "region_type")),
        "decoder_diagnostics": {
            "train_repeated_token_failures": sum(row["decoder_degenerated"] for row in final_train_predictions),
            "val_repeated_token_failures": sum(row["decoder_degenerated"] for row in final_val_predictions),
            "train_empty_predictions": sum(row["empty_prediction"] for row in final_train_predictions),
            "val_empty_predictions": sum(row["empty_prediction"] for row in final_val_predictions),
            "train_replacement_char_outputs": sum(row["decoded_nfc_unicode"]["replacement_char_count"] > 0 for row in final_train_predictions),
            "val_replacement_char_outputs": sum(row["decoded_nfc_unicode"]["replacement_char_count"] > 0 for row in final_val_predictions),
            "reference_replacement_chars": sum(row["reference_unicode"]["replacement_char_count"] for row in final_train_predictions + final_val_predictions),
        },
        "runtime_seconds": time.time() - start,
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
            "val_rows": len(val_rows),
        },
        metrics=metrics,
    )
    print(f"train_mean_cer={metrics['final_train']['mean_cer']}")
    print(f"val_mean_cer={metrics['final_val']['mean_cer']}")
    print(f"reload_val_mean_cer={metrics['reload_val']['mean_cer']}")
    print(f"probe_loss_before={probe_loss_before}")
    print(f"probe_loss_after={probe_loss_after}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
