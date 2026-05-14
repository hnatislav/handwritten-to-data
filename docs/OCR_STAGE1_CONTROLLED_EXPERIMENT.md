# OCR Stage-1 Controlled Experiment

## Purpose

Create the first non-smoke OCR experiment.

The objective is to obtain a meaningful OCR learning signal while keeping validation controls explicit.

This is still not a leaderboard optimization run.

## Dataset Artifact

Default artifact:

```text
data/ocr_stage1/rukopys_ocr_stage1_v1
```

Target size:

- train: 3,000 crops;
- validation: 400 crops.

Config:

```text
configs/ocr_stage1_subset_v1.json
```

Create locally:

```bash
.venv/bin/python -m scripts.create_ocr_stage1_dataset \
  --train-manifest data/ocr_crops/rukopys_gt_only_train_v1/manifests/train.jsonl \
  --val-manifest data/ocr_crops/rukopys_gt_only_train_v1/manifests/val.jsonl \
  --output-dir data/ocr_stage1/rukopys_ocr_stage1_v1 \
  --model-name cyrillic-trocr/trocr-handwritten-cyrillic \
  --seed 42 \
  --train-size 3000 \
  --val-size 400 \
  --max-token-length 128 \
  --zip
```

Upload the generated zip as a Kaggle Dataset named:

```text
rukopys-ocr-stage1-v1
```

Generated local artifact:

```text
path: data/ocr_stage1/rukopys_ocr_stage1_v1.zip
size: ~1.0 GB
sha256: 4ee4dd461521488f4611bec83c89eb44a6acfae9e91780f79a2d5186d4ec951b
```

Expected Kaggle root:

```text
/kaggle/input/rukopys-ocr-stage1-v1/rukopys_ocr_stage1_v1
```

## Filtering Policy

Implemented by `scripts/create_ocr_stage1_dataset.py`.

Policy:

- drop empty text rows;
- compute token lengths with `cyrillic-trocr/trocr-handwritten-cyrillic`;
- drop rows above `max_token_length=128`;
- select validation first with unique exact text;
- select training rows after excluding selected validation exact texts;
- deterministic ordering by `crop_id` plus fixed seed.

This removes selected exact-text overlap between train and validation.

Known limitation:

- exact-text deduplication does not remove semantic duplicates or near-duplicates.

## Dataset Outputs

Expected structure:

```text
rukopys_ocr_stage1_v1/
  crops/
    train/*.png
    val/*.png
  manifests/
    train.jsonl
    val.jsonl
    all.jsonl
  stats/
    stage1_dataset_stats.json
    dropped_overlength_examples.jsonl
  FILTERING_POLICY.md
  dataset_info.json
  checksums.json
```

Key statistics to inspect before training:

```bash
cat data/ocr_stage1/rukopys_ocr_stage1_v1/stats/stage1_dataset_stats.json
```

Required checks:

- `selected_exact_text_overlap_count` must be `0`;
- selected train/val overflow counts must be `0`;
- token-length coverage must be recorded;
- source/type distributions must be inspected.

Observed for the generated local artifact:

```text
selected_train_size=3000
selected_val_size=400
selected_exact_text_overlap_count=0
raw_exact_text_overlap_count=677
eligible_exact_text_overlap_count=677
train_rows_removed_by_selected_val_texts=768
max_token_length=128
dropped_over_max_token_length=5
train_coverage_pct=99.97164729231642
val_coverage_pct=100.0
selected_train_token_length_max=112
selected_val_token_length_max=100
```

## Kaggle Setup

Clone repository:

```bash
git clone git@github.com:hnatislav/handwritten-to-data.git /kaggle/working/handwritten-to-data
cd /kaggle/working/handwritten-to-data
```

If SSH is unavailable:

```bash
git clone https://github.com/hnatislav/handwritten-to-data.git /kaggle/working/handwritten-to-data
cd /kaggle/working/handwritten-to-data
```

Install pinned smoke/stage-1 runtime:

```bash
pip install -r requirements-kaggle-ocr-smoke.lock.txt
```

Verify GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

## Training Command

Config:

```text
configs/ocr_stage1_train_kaggle_v1.json
```

Run:

```bash
python -m scripts.train_ocr_stage1 \
  --train-manifest /kaggle/input/rukopys-ocr-stage1-v1/rukopys_ocr_stage1_v1/manifests/train.jsonl \
  --val-manifest /kaggle/input/rukopys-ocr-stage1-v1/rukopys_ocr_stage1_v1/manifests/val.jsonl \
  --output-dir /kaggle/working/outputs/ocr_stage1_v1 \
  --model-name cyrillic-trocr/trocr-handwritten-cyrillic \
  --device cuda \
  --seed 42 \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 5e-6 \
  --weight-decay 0.0 \
  --max-target-length 128 \
  --max-new-tokens 128 \
  --hardest-k 50 \
  --sample-k 40
```

Do not pass `--allow-truncation` for the controlled run.

The literal command line should omit `--train-limit` for the full stage-1 run. Use `--train-limit` only for diagnostic dry runs.

## Output Artifacts

Expected output:

```text
/kaggle/working/outputs/ocr_stage1_v1/
  checkpoint/
  metadata.json
  metrics.json
  predictions.jsonl
  hardest_examples.jsonl
  longest_failures.jsonl
  repeated_token_failures.jsonl
  truncation_failures.jsonl
  representative_samples.jsonl
```

`metrics.json` includes:

- train loss summaries;
- fixed probe loss before/after;
- validation CER;
- CER grouped by source, region type, language, annotation source, text length bucket, and handwriting bucket;
- target-length policy;
- package/CUDA runtime metadata;
- peak CUDA memory fields when running on CUDA.

## Interpretation Rules

Acceptable conclusions:

- whether the model learns on a controlled subset;
- whether checkpoint reload and validation evaluation remain stable;
- which source/type/length buckets fail most obviously.

Invalid conclusions:

- full-dataset OCR quality;
- leaderboard readiness;
- page-level performance;
- detector or reading-order quality.

## Stop Conditions

Stop and inspect artifacts if:

- target-length overflow is nonzero;
- selected exact-text overlap is nonzero;
- repeated-token failures dominate predictions;
- validation CER is driven by a small number of very long samples;
- GPU memory is near T4 capacity.
