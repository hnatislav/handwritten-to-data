from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.rukopys_dataset import DATASET_ID, iter_records, load_rukopys
from src.data.schema import BBox, coerce_bbox
from src.data.validation_split import stable_hash
from src.pipeline.oracle import image_id_from_record
from src.preprocessing.image_ops import crop_region, ensure_rgb


DEFAULT_INCLUDED_TYPES = ("handwritten", "printed")


@dataclass(frozen=True)
class OCRCropExportConfig:
    output_dir: str
    dataset_id: str = DATASET_ID
    dataset_config: str = "gt_only"
    dataset_split: str = "train"
    dataset_revision: str | None = None
    seed: int = 42
    validation_fraction: float = 0.2
    crop_padding: int = 8
    included_types: tuple[str, ...] = DEFAULT_INCLUDED_TYPES
    included_languages: tuple[str, ...] = ("uk",)
    included_legibilities: tuple[str, ...] = ("legible",)
    min_text_length: int = 1
    max_text_length: int | None = None
    limit_images: int | None = None


@dataclass(frozen=True)
class RegionFilterDecision:
    keep: bool
    reason: str


@dataclass(frozen=True)
class OCRCropManifestRow:
    crop_id: str
    split: str
    crop_path: str
    image_id: str
    region_index: int
    text: str
    region_type: str
    language: str | None
    legibility: str | None
    bbox: BBox
    image_width: int
    image_height: int
    source: str | None
    annotation_source: str | None
    year: int | None
    grade: int | None
    subject: str | None
    group: str | None
    fund: str | None
    content_type: str | None


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def infer_dataset_revision_from_record(record: dict[str, Any]) -> str | None:
    image = record.get("image")
    filename = getattr(image, "filename", None)
    if not filename:
        return None
    parts = Path(filename).parts
    if "snapshots" not in parts:
        return None
    snapshot_index = parts.index("snapshots")
    if snapshot_index + 1 >= len(parts):
        return None
    return parts[snapshot_index + 1]


def split_for_image(image_id: str, validation_fraction: float, seed: int) -> str:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    threshold = int(validation_fraction * 10_000)
    bucket = stable_hash(image_id, seed=seed) % 10_000
    return "val" if bucket < threshold else "train"


def filter_region(region: dict[str, Any], config: OCRCropExportConfig) -> RegionFilterDecision:
    region_type = region.get("type")
    if region_type not in config.included_types:
        return RegionFilterDecision(False, f"excluded_type:{region_type}")

    language = region.get("language")
    if config.included_languages and language not in config.included_languages:
        return RegionFilterDecision(False, f"excluded_language:{language}")

    legibility = region.get("legibility")
    if config.included_legibilities and legibility not in config.included_legibilities:
        return RegionFilterDecision(False, f"excluded_legibility:{legibility}")

    text = str(region.get("text") or "")
    if len(text) < config.min_text_length:
        return RegionFilterDecision(False, "empty_or_too_short_text")

    if config.max_text_length is not None and len(text) > config.max_text_length:
        return RegionFilterDecision(False, "too_long_text")

    try:
        coerce_bbox(region["bbox"])
    except (KeyError, TypeError, ValueError):
        return RegionFilterDecision(False, "invalid_bbox")

    return RegionFilterDecision(True, "keep")


def crop_filename(image_id: str, region_index: int) -> str:
    stem = Path(image_id).stem
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return f"{safe_stem}__r{region_index:04d}.png"


def row_to_json(row: OCRCropManifestRow) -> dict[str, Any]:
    payload = asdict(row)
    payload["bbox"] = list(row.bbox)
    return payload


def write_jsonl(path: Path, rows: list[OCRCropManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row_to_json(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[OCRCropManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_to_json(rows[0]).keys()) if rows else list(OCRCropManifestRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = row_to_json(row)
            payload["bbox"] = json.dumps(payload["bbox"], separators=(",", ":"))
            payload["text"] = str(payload["text"]).replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
            writer.writerow(payload)


def export_ocr_crops(config: OCRCropExportConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    crops_dir = output_dir / "crops"
    manifests_dir = output_dir / "manifests"
    stats_dir = output_dir / "stats"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_rukopys(
        config=config.dataset_config,
        split=config.dataset_split,
        revision=config.dataset_revision,
        streaming=False,
    )

    rows: list[OCRCropManifestRow] = []
    split_manifest: dict[str, str] = {}
    stats: dict[str, Any] = {
        "config": asdict(config),
        "dataset_revision_observed": None,
        "images_seen": 0,
        "images_with_exported_regions": 0,
        "regions_seen": 0,
        "regions_exported": 0,
        "skip_reasons": Counter(),
        "exported_by_split": Counter(),
        "images_by_split": Counter(),
        "images_with_exported_regions_by_split": Counter(),
        "exported_by_type": Counter(),
        "exported_by_source": Counter(),
        "image_area": {
            "max": 0,
            "max_image_id": None,
        },
        "text_length": {
            "min": None,
            "max": None,
            "sum": 0,
        },
    }

    for record in iter_records(dataset, limit=config.limit_images):
        image_id = image_id_from_record(record)
        split = split_for_image(image_id, config.validation_fraction, config.seed)
        split_manifest[image_id] = split

        observed_revision = infer_dataset_revision_from_record(record)
        if stats["dataset_revision_observed"] is None and observed_revision is not None:
            stats["dataset_revision_observed"] = observed_revision

        image = ensure_rgb(record["image"])
        image_width = int(record.get("image_width") or image.width)
        image_height = int(record.get("image_height") or image.height)
        exported_for_image = 0
        stats["images_seen"] += 1
        stats["images_by_split"][split] += 1
        image_area = image_width * image_height
        if image_area > stats["image_area"]["max"]:
            stats["image_area"]["max"] = image_area
            stats["image_area"]["max_image_id"] = image_id

        for region_index, region in enumerate(record.get("regions", [])):
            stats["regions_seen"] += 1
            decision = filter_region(region, config)
            if not decision.keep:
                stats["skip_reasons"][decision.reason] += 1
                continue

            bbox = coerce_bbox(region["bbox"])
            try:
                crop = crop_region(image, bbox, padding=config.crop_padding)
            except ValueError:
                stats["skip_reasons"]["invalid_bbox_after_padding"] += 1
                continue

            filename = crop_filename(image_id, region_index)
            relative_crop_path = Path("crops") / split / filename
            crop_path = output_dir / relative_crop_path
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path, format="PNG", compress_level=6)

            text = str(region.get("text") or "")
            row = OCRCropManifestRow(
                crop_id=f"{Path(image_id).stem}::r{region_index:04d}",
                split=split,
                crop_path=relative_crop_path.as_posix(),
                image_id=image_id,
                region_index=region_index,
                text=text,
                region_type=str(region.get("type")),
                language=normalize_optional_str(region.get("language")),
                legibility=normalize_optional_str(region.get("legibility")),
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
                source=normalize_optional_str(record.get("source")),
                annotation_source=normalize_optional_str(record.get("annotation_source")),
                year=record.get("year"),
                grade=record.get("grade"),
                subject=normalize_optional_str(record.get("subject")),
                group=normalize_optional_str(record.get("group")),
                fund=normalize_optional_str(record.get("fund")),
                content_type=normalize_optional_str(record.get("content_type")),
            )
            rows.append(row)
            exported_for_image += 1
            stats["regions_exported"] += 1
            stats["exported_by_split"][split] += 1
            stats["exported_by_type"][row.region_type] += 1
            stats["exported_by_source"][row.source or "unknown"] += 1
            text_len = len(text)
            stats["text_length"]["sum"] += text_len
            stats["text_length"]["min"] = text_len if stats["text_length"]["min"] is None else min(stats["text_length"]["min"], text_len)
            stats["text_length"]["max"] = text_len if stats["text_length"]["max"] is None else max(stats["text_length"]["max"], text_len)

        if exported_for_image:
            stats["images_with_exported_regions"] += 1
            stats["images_with_exported_regions_by_split"][split] += 1

    rows = sorted(rows, key=lambda row: (row.split, row.image_id, row.region_index))
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]

    write_jsonl(manifests_dir / "train.jsonl", train_rows)
    write_jsonl(manifests_dir / "val.jsonl", val_rows)
    write_jsonl(manifests_dir / "all.jsonl", rows)
    write_csv(manifests_dir / "all.csv", rows)

    split_payload = {
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "split_unit": "image_id",
        "assignments": dict(sorted(split_manifest.items())),
    }
    (manifests_dir / "split_manifest.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    text_count = stats["regions_exported"]
    stats["text_length"]["mean"] = stats["text_length"]["sum"] / text_count if text_count else 0.0
    stats["skip_reasons"] = dict(sorted(stats["skip_reasons"].items()))
    stats["exported_by_split"] = dict(sorted(stats["exported_by_split"].items()))
    stats["images_by_split"] = dict(sorted(stats["images_by_split"].items()))
    stats["images_with_exported_regions_by_split"] = dict(sorted(stats["images_with_exported_regions_by_split"].items()))
    stats["exported_by_type"] = dict(sorted(stats["exported_by_type"].items()))
    stats["exported_by_source"] = dict(sorted(stats["exported_by_source"].items()))
    stats["manifest_paths"] = {
        "train_jsonl": "manifests/train.jsonl",
        "val_jsonl": "manifests/val.jsonl",
        "all_jsonl": "manifests/all.jsonl",
        "all_csv": "manifests/all.csv",
        "split_manifest": "manifests/split_manifest.json",
    }
    (stats_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return stats
