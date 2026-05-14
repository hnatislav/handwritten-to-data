# OCR Experiment Analysis

## Purpose

Provide deterministic post-run inspection for OCR experiments.

This is file-based tooling only. It does not introduce dashboards, external tracking services, or LLM-based error classification.

## Standard Output Files

Stage-1 OCR runs should contain:

```text
metrics.json
runtime.json
metadata.json
config_snapshot.json
environment_snapshot.json
predictions.jsonl
hardest_examples.jsonl
sample_predictions.jsonl
longest_failures.jsonl
repeated_token_failures.jsonl
truncation_failures.jsonl
```

Older runs may lack some files. The inspector tolerates missing config/environment snapshots.

## Inspect One Run

```bash
python -m scripts.inspect_ocr_experiment \
  --experiment-dir /kaggle/working/outputs/ocr_stage1_v1 \
  --output-dir /kaggle/working/outputs/ocr_stage1_v1/analysis \
  --worst-k 50
```

Outputs:

```text
analysis/summary.json
analysis/worst_k.jsonl
```

Summary includes:

- overall CER;
- CER percentiles;
- CER histogram;
- grouped CER by source/domain, text length bucket, region type, language, annotation source, handwriting bucket, and error bucket;
- worst-k examples.

## Compare Two Runs

```bash
python -m scripts.compare_ocr_experiments \
  --base-dir /kaggle/working/outputs/ocr_stage1_baseline \
  --candidate-dir /kaggle/working/outputs/ocr_stage1_v1 \
  --output-dir /kaggle/working/outputs/ocr_stage1_compare \
  --worst-k 50
```

Outputs:

```text
comparison.json
largest_regressions.jsonl
largest_improvements.jsonl
```

Comparison requires shared `crop_id` values.

## Render Static Grids

```bash
python -m scripts.render_ocr_samples \
  --experiment-dir /kaggle/working/outputs/ocr_stage1_v1 \
  --manifest /kaggle/input/rukopys-ocr-stage1-v1/rukopys_ocr_stage1_v1/manifests/val.jsonl \
  --mode hardest \
  --limit 20 \
  --columns 2 \
  --output /kaggle/working/outputs/ocr_stage1_v1/analysis/hardest_grid.png
```

Modes:

- `hardest`;
- `sample`;
- `truncation`.

## Error Buckets

Implemented deterministically in `src.analysis.ocr_experiment`.

Buckets:

- `truncation_failure`;
- `repeated_token_degeneration`;
- `empty_prediction`;
- `punctuation_only_drift`;
- `cyrillic_character_confusion`;
- `long_line_collapse`;
- `insertion_heavy_failure`;
- `deletion_heavy_failure`;
- `other_ocr_error`;
- `exact_match`.

These are heuristic debugging labels, not scorer labels.

## Interpretation Constraints

Use the tools to identify failure modes and regressions.

Do not use a single aggregate CER value as sufficient evidence of progress. Inspect:

- median vs p95/p99 CER;
- source/domain grouped CER;
- long-line failures;
- repeated-token failures;
- regression cases against the prior run.
