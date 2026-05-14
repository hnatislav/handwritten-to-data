from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

from src.analysis.ocr_experiment import (
    read_jsonl,
    unicode_summary,
    utf8_file_diagnostics,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw OCR artifact text without image rendering.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fields", nargs="+", default=["reference_text", "predicted_text", "decoded_raw", "decoded_clean", "decoded_nfc"])
    return parser.parse_args()


def inspect_row(row: dict, fields: list[str]) -> dict:
    inspected = {"crop_id": row.get("crop_id")}
    for field in fields:
        if field not in row:
            continue
        text = str(row.get(field, ""))
        inspected[field] = {
            "raw": text,
            "repr": repr(text),
            "unicode": unicode_summary(text),
            "nfc_equal_raw": unicodedata.normalize("NFC", text) == text,
            "nfd_equal_raw": unicodedata.normalize("NFD", text) == text,
        }
    return inspected


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl)
    rows = read_jsonl(path)
    inspected = [inspect_row(row, args.fields) for row in rows[: args.limit]]
    diagnostics = {
        "file": utf8_file_diagnostics(path),
        "rows_total": len(rows),
        "rows_inspected": len(inspected),
        "replacement_counts": {
            field: sum(str(row.get(field, "")).count("\ufffd") for row in rows)
            for field in args.fields
        },
        "control_counts": {
            field: sum(
                sum(item["is_control"] for item in unicode_summary(str(row.get(field, "")))["codepoints"])
                for row in rows
            )
            for field in args.fields
        },
    }
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True))
    for item in inspected:
        print(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output_dir:
        output_dir = Path(args.output_dir)
        write_json(output_dir / "unicode_file_diagnostics.json", diagnostics)
        write_jsonl(output_dir / "unicode_row_diagnostics.jsonl", inspected)


if __name__ == "__main__":
    main()
