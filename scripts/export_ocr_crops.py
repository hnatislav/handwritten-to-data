from __future__ import annotations

import argparse
from pathlib import Path

from src.data.ocr_crop_exporter import OCRCropExportConfig, export_ocr_crops
from src.experiments.logging import write_experiment_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic OCR crops from RUKOPYS gt_only/train.")
    parser.add_argument("--output-dir", default="data/ocr_crops/rukopys_gt_only_train_v1")
    parser.add_argument("--dataset-config", default="gt_only")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--crop-padding", type=int, default=8)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--max-text-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OCRCropExportConfig(
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_revision=args.dataset_revision,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        crop_padding=args.crop_padding,
        limit_images=args.limit_images,
        max_text_length=args.max_text_length,
    )
    stats = export_ocr_crops(config)
    write_experiment_metadata(
        experiment_dir=Path(args.output_dir) / "logs",
        config=stats["config"],
        dataset_info={
            "dataset_id": stats["config"]["dataset_id"],
            "dataset_config": stats["config"]["dataset_config"],
            "dataset_split": stats["config"]["dataset_split"],
            "dataset_revision_requested": stats["config"]["dataset_revision"],
            "dataset_revision_observed": stats["dataset_revision_observed"],
        },
        metrics={
            "images_seen": stats["images_seen"],
            "regions_seen": stats["regions_seen"],
            "regions_exported": stats["regions_exported"],
            "exported_by_split": stats["exported_by_split"],
        },
    )
    print(f"output_dir={args.output_dir}")
    print(f"images_seen={stats['images_seen']}")
    print(f"regions_seen={stats['regions_seen']}")
    print(f"regions_exported={stats['regions_exported']}")
    print(f"exported_by_split={stats['exported_by_split']}")
    print(f"dataset_revision_observed={stats['dataset_revision_observed']}")


if __name__ == "__main__":
    main()
