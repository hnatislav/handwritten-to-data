from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.ocr_experiment import compare_experiments, load_experiment, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two OCR experiment output directories.")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worst-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_experiment(args.base_dir)
    candidate = load_experiment(args.candidate_dir)
    comparison = compare_experiments(base, candidate, worst_k=args.worst_k)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", comparison)
    write_jsonl(output_dir / "largest_regressions.jsonl", comparison["largest_regressions"])
    write_jsonl(output_dir / "largest_improvements.jsonl", comparison["largest_improvements"])
    print(f"shared_rows={comparison['shared_rows']}")
    print(f"mean_delta_cer={comparison['mean_delta_cer']}")
    print(f"regression_count={comparison['regression_count']}")
    print(f"improvement_count={comparison['improvement_count']}")
    print(f"hardest_overlap_at_k={comparison['hardest_overlap_at_k']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
