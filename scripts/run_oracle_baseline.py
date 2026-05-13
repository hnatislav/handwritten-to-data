from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.qualitative import draw_region_errors, page_text_payload, write_hardest_examples
from src.data.rukopys_dataset import DATASET_ID, iter_records, load_rukopys
from src.evaluation.metrics import evaluate_pages, mean_page_cer
from src.experiments.logging import write_experiment_metadata, write_json
from src.ocr.dummy import EmptyOCRRecognizer, OracleTextRecognizer
from src.ocr.trocr import TrOCRConfig, TrOCRRecognizer
from src.pipeline.oracle import page_from_rukopys_record, run_oracle_pipeline_on_record
from src.submission.writer import write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal oracle-region baseline.")
    parser.add_argument("--config", default="gt_only")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--recognizer", choices=["empty", "oracle_text", "trocr"], default="empty")
    parser.add_argument("--trocr-model", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--qualitative-top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/experiments/000_oracle_baseline")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def build_recognizer(args: argparse.Namespace):
    name = args.recognizer
    if name == "empty":
        return EmptyOCRRecognizer()
    if name == "oracle_text":
        return OracleTextRecognizer()
    if name == "trocr":
        return TrOCRRecognizer(
            TrOCRConfig(
                model_name=args.trocr_model,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        )
    raise ValueError(f"Unknown recognizer: {name}")


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_rukopys(
        config=args.config,
        split=args.split,
        revision=args.revision,
        streaming=args.streaming,
    )
    recognizer = build_recognizer(args)

    references = []
    predictions = []
    source_records = []
    for record in iter_records(dataset, limit=args.limit):
        source_records.append(record)
        references.append(page_from_rukopys_record(record))
        predictions.append(run_oracle_pipeline_on_record(record, recognizer=recognizer))

    evaluations = evaluate_pages(references, predictions)
    metrics = {
        "pages": len(evaluations),
        "mean_page_cer": mean_page_cer(evaluations),
        "page_cer_by_image": {item.image_id: item.page_cer for item in evaluations},
    }

    write_submission(predictions, output_dir / "submission.csv")
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "predictions.jsonl",
        [page.to_submission_row() for page in predictions],
    )
    write_jsonl(
        output_dir / "page_texts.jsonl",
        [
            {
                "image_id": reference.image_id,
                "reference_text": page_text_payload(reference)["page_text"],
                "predicted_text": page_text_payload(prediction)["page_text"],
            }
            for reference, prediction in zip(references, predictions, strict=True)
        ],
    )
    write_hardest_examples(evaluations, output_dir / "qualitative" / "text", top_k=args.qualitative_top_k)

    for record, reference, prediction, evaluation in zip(source_records, references, predictions, evaluations, strict=True):
        if evaluation.page_cer <= 0:
            continue
        safe_id = Path(reference.image_id).stem
        draw_region_errors(
            record["image"],
            reference,
            prediction,
            output_dir / "qualitative" / "images" / f"{safe_id}.jpg",
        )

    write_experiment_metadata(
        experiment_dir=output_dir,
        config=vars(args),
        dataset_info={
            "dataset_id": DATASET_ID,
            "config": args.config,
            "split": args.split,
            "revision": args.revision,
            "limit": args.limit,
        },
        metrics=metrics,
    )

    print(f"pages={metrics['pages']}")
    print(f"mean_page_cer={metrics['mean_page_cer']:.6f}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
