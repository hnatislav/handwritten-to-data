from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.evaluation.cer import character_error_rate
from src.experiments.logging import write_experiment_metadata, write_json
from src.ocr.crop_manifest import read_manifest, resolve_crop_path
from src.ocr.trocr import TrOCRConfig, TrOCRRecognizer


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
    parser = argparse.ArgumentParser(description="Tiny OCR fine-tuning smoke test.")
    parser.add_argument("--train-manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="data/ocr_crops/rukopys_gt_only_train_v1/manifests/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/experiments/004_ocr_train_smoke")
    parser.add_argument("--model-name", default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--val-limit", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--eval-limit", type=int, default=8)
    return parser.parse_args()


def deterministic_rows(path: str, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = read_manifest(path)
    rows = sorted(rows, key=lambda row: row["crop_id"])
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def evaluate_reloaded_checkpoint(checkpoint_dir: Path, val_manifest: str, val_rows: list[dict[str, Any]], device: str, max_new_tokens: int) -> dict[str, Any]:
    recognizer = TrOCRRecognizer(
        TrOCRConfig(
            model_name=str(checkpoint_dir),
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )
    )
    rows = []
    for row in val_rows:
        with Image.open(resolve_crop_path(val_manifest, row)) as image:
            result = recognizer.recognize(image.convert("RGB"), region=None)  # type: ignore[arg-type]
        rows.append(
            {
                "crop_id": row["crop_id"],
                "reference_text": row["text"],
                "predicted_text": result.text,
                "cer": character_error_rate(row["text"], result.text),
            }
        )
    return {
        "rows": rows,
        "mean_cer": sum(row["cer"] for row in rows) / len(rows) if rows else None,
    }


def fixed_batch_loss(model, dataset: OCRCropDataset, device: torch.device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        collate_fn=collate,
    )
    batch = next(iter(loader))
    with torch.inference_mode():
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            labels=batch["labels"].to(device),
        )
    model.train()
    return float(outputs.loss.detach().cpu())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.to(device)
    model.train()

    train_rows = deterministic_rows(args.train_manifest, args.train_limit, args.seed)
    val_rows = deterministic_rows(args.val_manifest, args.val_limit, args.seed)
    train_dataset = OCRCropDataset(args.train_manifest, train_rows, processor, args.max_target_length)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    probe_loss_before = fixed_batch_loss(model, train_dataset, device, args.batch_size)
    losses: list[float] = []
    for _epoch in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    probe_loss_after = fixed_batch_loss(model, train_dataset, device, args.batch_size)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    eval_result = evaluate_reloaded_checkpoint(
        checkpoint_dir=checkpoint_dir,
        val_manifest=args.val_manifest,
        val_rows=val_rows[: args.eval_limit],
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    metrics = {
        "losses": losses,
        "first_train_batch_loss": losses[0] if losses else None,
        "last_train_batch_loss": losses[-1] if losses else None,
        "probe_loss_before": probe_loss_before,
        "probe_loss_after": probe_loss_after,
        "loss_decreased": probe_loss_after < probe_loss_before,
        "reloaded_checkpoint_eval": {
            "rows": len(eval_result["rows"]),
            "mean_cer": eval_result["mean_cer"],
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "reloaded_predictions.json", {"rows": eval_result["rows"]})
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
    print(f"probe_loss_before={metrics['probe_loss_before']}")
    print(f"probe_loss_after={metrics['probe_loss_after']}")
    print(f"loss_decreased={metrics['loss_decreased']}")
    print(f"reloaded_mean_cer={metrics['reloaded_checkpoint_eval']['mean_cer']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
