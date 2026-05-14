from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.evaluation.cer import character_error_rate, levenshtein_distance
from src.ocr.crop_manifest import resolve_crop_path, text_length_bucket


PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
REPLACEMENT_CHAR = "\ufffd"


@dataclass(frozen=True)
class EditProfile:
    insertions: int
    deletions: int
    substitutions: int
    equal: int
    cyrillic_substitutions: int

    @property
    def total_errors(self) -> int:
        return self.insertions + self.deletions + self.substitutions


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return output


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def pct(q: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": float(ordered[-1]),
    }


def cer_histogram(rows: list[dict[str, Any]], bins: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0, math.inf)) -> dict[str, int]:
    counts: dict[str, int] = {}
    for left, right in zip(bins, bins[1:], strict=False):
        label = f"{left:.2f}_{right:.2f}" if math.isfinite(right) else f"{left:.2f}_inf"
        counts[label] = 0
    for row in rows:
        cer = float(row.get("cer", 0.0))
        for left, right in zip(bins, bins[1:], strict=False):
            if left <= cer < right or (right == math.inf and cer >= left):
                label = f"{left:.2f}_{right:.2f}" if math.isfinite(right) else f"{left:.2f}_inf"
                counts[label] += 1
                break
    return counts


def edit_profile(reference: str, prediction: str) -> EditProfile:
    matcher = difflib.SequenceMatcher(a=reference, b=prediction, autojunk=False)
    insertions = 0
    deletions = 0
    substitutions = 0
    equal = 0
    cyrillic_substitutions = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_span = reference[i1:i2]
        pred_span = prediction[j1:j2]
        if tag == "equal":
            equal += i2 - i1
        elif tag == "insert":
            insertions += j2 - j1
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "replace":
            substitutions += max(i2 - i1, j2 - j1)
            for ref_char, pred_char in zip(ref_span, pred_span, strict=False):
                if CYRILLIC_RE.search(ref_char) and CYRILLIC_RE.search(pred_char) and ref_char != pred_char:
                    cyrillic_substitutions += 1
    return EditProfile(
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        equal=equal,
        cyrillic_substitutions=cyrillic_substitutions,
    )


def has_repeated_token_failure(text: str) -> bool:
    tokens = re.findall(r"\S+", text)
    if len(tokens) >= 4 and any(count >= 3 for count in Counter(tokens).values()):
        return True
    compact = re.sub(r"\s+", "", text)
    return bool(re.search(r"(.)\1{5,}", compact))


def punctuation_only_drift(reference: str, prediction: str) -> bool:
    return bool(prediction.strip()) and bool(PUNCTUATION_RE.match(prediction.strip())) and reference.strip() != prediction.strip()


def unicode_codepoints(text: str) -> list[dict[str, Any]]:
    rows = []
    for index, char in enumerate(text):
        codepoint = ord(char)
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "<unnamed>"
        rows.append(
            {
                "index": index,
                "char": char,
                "repr": repr(char),
                "codepoint": f"U+{codepoint:04X}",
                "category": unicodedata.category(char),
                "name": name,
                "is_replacement_char": char == REPLACEMENT_CHAR,
                "is_control": unicodedata.category(char).startswith("C"),
            }
        )
    return rows


def unicode_summary(text: str) -> dict[str, Any]:
    codepoints = unicode_codepoints(text)
    return {
        "length": len(text),
        "repr": repr(text),
        "replacement_char_count": sum(item["is_replacement_char"] for item in codepoints),
        "control_char_count": sum(item["is_control"] for item in codepoints),
        "non_ascii_count": sum(ord(char) > 127 for char in text),
        "cyrillic_count": len(CYRILLIC_RE.findall(text)),
        "codepoints": codepoints,
    }


def classify_error(row: dict[str, Any], max_target_length: int | None = None) -> list[str]:
    reference = str(row.get("reference_text", row.get("text", "")))
    prediction = str(row.get("predicted_text", ""))
    profile = edit_profile(reference, prediction)
    ref_len = max(1, len(reference))
    pred_len = len(prediction)
    labels: list[str] = []
    token_length = row.get("reference_token_length", row.get("token_length"))
    if max_target_length is not None and token_length is not None and int(token_length) > max_target_length:
        labels.append("truncation_failure")
    if not prediction.strip():
        labels.append("empty_prediction")
    if has_repeated_token_failure(prediction):
        labels.append("repeated_token_degeneration")
    if punctuation_only_drift(reference, prediction):
        labels.append("punctuation_only_drift")
    if REPLACEMENT_CHAR in prediction:
        labels.append("invalid_unicode_output")
    if REPLACEMENT_CHAR in reference:
        labels.append("invalid_unicode_reference")
    if profile.cyrillic_substitutions:
        labels.append("cyrillic_character_confusion")
    if len(reference) >= 80 and pred_len <= max(10, int(0.4 * len(reference))):
        labels.append("long_line_collapse")
    if profile.insertions / ref_len >= 0.35 and profile.insertions > profile.deletions + profile.substitutions:
        labels.append("insertion_heavy_failure")
    if profile.deletions / ref_len >= 0.35 and profile.deletions > profile.insertions + profile.substitutions:
        labels.append("deletion_heavy_failure")
    if not labels and float(row.get("cer", 0.0)) > 0:
        labels.append("other_ocr_error")
    if not labels:
        labels.append("exact_match")
    return labels


def enrich_prediction(row: dict[str, Any], max_target_length: int | None = None) -> dict[str, Any]:
    reference = str(row.get("reference_text", row.get("text", "")))
    prediction = str(row.get("predicted_text", ""))
    profile = edit_profile(reference, prediction)
    cer = float(row.get("cer", character_error_rate(reference, prediction)))
    edit_distance = int(row.get("edit_distance", levenshtein_distance(reference, prediction)))
    enriched = {
        **row,
        "reference_text": reference,
        "predicted_text": prediction,
        "cer": cer,
        "edit_distance": edit_distance,
        "reference_length": int(row.get("reference_length", len(reference))),
        "prediction_length": len(prediction),
        "text_length_bucket": row.get("text_length_bucket", text_length_bucket(reference)),
        "edit_insertions": profile.insertions,
        "edit_deletions": profile.deletions,
        "edit_substitutions": profile.substitutions,
        "edit_cyrillic_substitutions": profile.cyrillic_substitutions,
    }
    enriched["error_buckets"] = classify_error(enriched, max_target_length=max_target_length)
    return enriched


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "mean_cer": None, "unweighted_mean_cer": None, "cer_percentiles": percentile_summary([])}
    edits = sum(int(row.get("edit_distance", 0)) for row in rows)
    ref_len = sum(int(row.get("reference_length", len(str(row.get("reference_text", ""))))) for row in rows)
    return {
        "rows": len(rows),
        "mean_cer": edits / ref_len if ref_len else 0.0,
        "unweighted_mean_cer": sum(float(row.get("cer", 0.0)) for row in rows) / len(rows),
        "cer_percentiles": percentile_summary([float(row.get("cer", 0.0)) for row in rows]),
    }


def grouped_metrics(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for key in keys:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(key)
            if isinstance(value, list):
                for item in value:
                    buckets[str(item)].append(row)
            else:
                buckets[str(value)].append(row)
        grouped[key] = {bucket: summarize_group(items) for bucket, items in sorted(buckets.items())}
    return grouped


def load_experiment(experiment_dir: str | Path, max_target_length: int | None = None) -> dict[str, Any]:
    root = Path(experiment_dir)
    metrics_path = root / "metrics.json"
    metadata_path = root / "metadata.json"
    config_path = root / "config_snapshot.json"
    environment_path = root / "environment_snapshot.json"
    predictions_path = root / "predictions.jsonl"
    rows = read_jsonl(predictions_path) if predictions_path.exists() else []
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    if max_target_length is None:
        max_target_length = (
            metrics.get("target_length_policy", {})
            .get("val", {})
            .get("max_target_length")
        )
    predictions = [enrich_prediction(row, max_target_length=max_target_length) for row in rows]
    return {
        "root": root,
        "metrics": metrics,
        "metadata": read_json(metadata_path) if metadata_path.exists() else {},
        "config": read_json(config_path) if config_path.exists() else {},
        "environment": read_json(environment_path) if environment_path.exists() else {},
        "predictions": predictions,
    }


def summarize_experiment(experiment: dict[str, Any], worst_k: int = 25) -> dict[str, Any]:
    predictions = experiment["predictions"]
    bucket_counts = Counter(bucket for row in predictions for bucket in row["error_buckets"])
    return {
        "experiment_dir": str(experiment["root"]),
        "rows": len(predictions),
        "overall": summarize_group(predictions),
        "cer_histogram": cer_histogram(predictions),
        "error_bucket_counts": dict(sorted(bucket_counts.items())),
        "grouped": grouped_metrics(
            predictions,
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
        "worst_k": [
            {
                "crop_id": row.get("crop_id"),
                "cer": row.get("cer"),
                "reference_length": row.get("reference_length"),
                "error_buckets": row.get("error_buckets"),
                "reference_text": row.get("reference_text"),
                "predicted_text": row.get("predicted_text"),
            }
            for row in sorted(predictions, key=lambda item: (item["cer"], item["reference_length"]), reverse=True)[:worst_k]
        ],
    }


def compare_experiments(base: dict[str, Any], candidate: dict[str, Any], worst_k: int = 50) -> dict[str, Any]:
    base_by_id = {row["crop_id"]: row for row in base["predictions"] if "crop_id" in row}
    candidate_by_id = {row["crop_id"]: row for row in candidate["predictions"] if "crop_id" in row}
    shared_ids = sorted(set(base_by_id) & set(candidate_by_id))
    deltas = []
    for crop_id in shared_ids:
        base_row = base_by_id[crop_id]
        candidate_row = candidate_by_id[crop_id]
        deltas.append(
            {
                "crop_id": crop_id,
                "base_cer": base_row["cer"],
                "candidate_cer": candidate_row["cer"],
                "delta_cer": candidate_row["cer"] - base_row["cer"],
                "reference_text": candidate_row["reference_text"],
                "base_prediction": base_row["predicted_text"],
                "candidate_prediction": candidate_row["predicted_text"],
                "base_buckets": base_row["error_buckets"],
                "candidate_buckets": candidate_row["error_buckets"],
            }
        )
    regressions = sorted([row for row in deltas if row["delta_cer"] > 0], key=lambda row: row["delta_cer"], reverse=True)
    improvements = sorted([row for row in deltas if row["delta_cer"] < 0], key=lambda row: row["delta_cer"])
    base_hard = {row["crop_id"] for row in sorted(base["predictions"], key=lambda item: item["cer"], reverse=True)[:worst_k]}
    candidate_hard = {row["crop_id"] for row in sorted(candidate["predictions"], key=lambda item: item["cer"], reverse=True)[:worst_k]}
    return {
        "base_dir": str(base["root"]),
        "candidate_dir": str(candidate["root"]),
        "shared_rows": len(shared_ids),
        "base_only_rows": len(set(base_by_id) - set(candidate_by_id)),
        "candidate_only_rows": len(set(candidate_by_id) - set(base_by_id)),
        "mean_delta_cer": sum(row["delta_cer"] for row in deltas) / len(deltas) if deltas else None,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "hardest_overlap_at_k": len(base_hard & candidate_hard),
        "largest_regressions": regressions[:worst_k],
        "largest_improvements": improvements[:worst_k],
    }


def render_prediction_grid(
    rows: list[dict[str, Any]],
    manifest_path: str | Path,
    output_path: str | Path,
    columns: int = 2,
    crop_width: int = 420,
    text_height: int = 150,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        canvas = Image.new("RGB", (crop_width, text_height), "white")
        canvas.save(output)
        return output
    font = ImageFont.load_default()
    cell_width = crop_width
    cell_height = crop_width // 3 + text_height
    rows_count = math.ceil(len(rows) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        col = index % columns
        grid_row = index // columns
        x0 = col * cell_width
        y0 = grid_row * cell_height
        try:
            with Image.open(resolve_crop_path(manifest_path, row)) as image:
                crop = image.convert("RGB")
        except FileNotFoundError:
            crop = Image.new("RGB", (crop_width, crop_width // 3), "white")
        crop.thumbnail((crop_width - 8, crop_width // 3), Image.Resampling.LANCZOS)
        canvas.paste(crop, (x0 + 4, y0 + 4))
        text_y = y0 + crop_width // 3 + 8
        lines = [
            f"crop_id: {row.get('crop_id')}",
            f"CER: {float(row.get('cer', 0.0)):.4f} buckets: {','.join(row.get('error_buckets', []))}",
            f"REF: {row.get('reference_text', '')}",
            f"PRED: {row.get('predicted_text', '')}",
        ]
        for offset, line in enumerate(lines):
            draw.text((x0 + 6, text_y + offset * 16), line[:110], fill="black", font=font)
    canvas.save(output)
    return output
