from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.ocr_experiment import load_experiment, render_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render static OCR crop/reference/prediction grids.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["hardest", "sample", "truncation"], default="hardest")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--font", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.experiment_dir)
    predictions = experiment["predictions"]
    if args.mode == "hardest":
        rows = sorted(predictions, key=lambda row: (row["cer"], row["reference_length"]), reverse=True)[: args.limit]
    elif args.mode == "truncation":
        rows = [row for row in predictions if "truncation_failure" in row["error_buckets"]][: args.limit]
    else:
        rows = sorted(predictions, key=lambda row: str(row.get("crop_id")))[: args.limit]
    output = Path(args.output) if args.output else Path(args.experiment_dir) / "analysis" / f"{args.mode}_grid.png"
    render_prediction_grid(rows, args.manifest, output, columns=args.columns, font_path=args.font)
    print(f"rows={len(rows)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
