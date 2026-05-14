from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.ocr_experiment import (
    load_experiment,
    summarize_experiment,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect OCR experiment artifacts.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--worst-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.experiment_dir)
    summary = summarize_experiment(experiment, worst_k=args.worst_k)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.experiment_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "worst_k.jsonl", summary["worst_k"])
    print(f"experiment_dir={args.experiment_dir}")
    print(f"rows={summary['rows']}")
    print(f"mean_cer={summary['overall']['mean_cer']}")
    print(f"p50_cer={summary['overall']['cer_percentiles']['p50']}")
    print(f"p95_cer={summary['overall']['cer_percentiles']['p95']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
