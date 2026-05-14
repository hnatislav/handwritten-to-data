# OCR Overfit Diagnostic

## Purpose

Run a controlled tiny-data overfit test to determine whether the current OCR path can memorize a small handwritten subset.

This diagnostic is specifically designed to separate model-capacity issues from:

- tokenizer/model mismatch;
- Unicode/reference corruption;
- decoding configuration errors;
- special-token handling bugs;
- generation degeneration;
- serialization/rendering artifacts.

This is not a generalization benchmark.

## Dataset

Create a tiny deterministic dataset:

```bash
python -m scripts.create_ocr_overfit_dataset \
  --source-manifest data/ocr_stage1/rukopys_ocr_stage1_v1/manifests/train.jsonl \
  --output-dir data/ocr_overfit/rukopys_ocr_overfit_v1 \
  --model-name cyrillic-trocr/trocr-handwritten-cyrillic \
  --seed 42 \
  --train-size 32 \
  --val-size 8 \
  --max-token-length 128 \
  --zip
```

Validation rows intentionally duplicate the first 8 selected train rows.

Reason:

- the goal is memorization and decode-path verification;
- validation text/image overlap is allowed for this diagnostic only.

Expected structure:

```text
rukopys_ocr_overfit_v1/
  crops/
    train/*.png
    val/*.png
  manifests/
    train.jsonl
    val.jsonl
    all.jsonl
  diagnostics/
    reference_integrity.jsonl
    tokenizer_roundtrip.jsonl
  stats/
    overfit_dataset_stats.json
  dataset_info.json
  checksums.json
```

Generated local artifact:

```text
path: data/ocr_overfit/rukopys_ocr_overfit_v1.zip
size: ~12 MB
sha256: d69f1afac4609aa3592b92caa663d5a1a6f2239cf913255bb1c6ed0990e88d7c
train_rows=32
val_rows=8
reference_replacement_char_count=0
tokenizer_roundtrip_failures=0
```

Before training inspect:

```bash
cat data/ocr_overfit/rukopys_ocr_overfit_v1/stats/overfit_dataset_stats.json
```

Required checks:

- reference replacement-character count;
- tokenizer roundtrip failures;
- tokenizer special tokens;
- token length range.

## Kaggle Setup

Upload the zip as:

```text
rukopys-ocr-overfit-v1
```

Install runtime:

```bash
pip install --no-deps -r requirements-kaggle-ocr.txt
python -m scripts.verify_kaggle_ocr_environment --require-cuda
```

## Training Command

Config:

```text
configs/ocr_overfit_v1.json
```

Run:

```bash
python -m scripts.train_ocr_overfit \
  --train-manifest /kaggle/input/rukopys-ocr-overfit-v1/rukopys_ocr_overfit_v1/manifests/train.jsonl \
  --val-manifest /kaggle/input/rukopys-ocr-overfit-v1/rukopys_ocr_overfit_v1/manifests/val.jsonl \
  --output-dir /kaggle/working/outputs/ocr_overfit_v1 \
  --model-name cyrillic-trocr/trocr-handwritten-cyrillic \
  --device cuda \
  --seed 42 \
  --epochs 40 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --weight-decay 0.0 \
  --max-target-length 128 \
  --max-new-tokens 128 \
  --eval-every 5 \
  --sample-k 8
```

Greedy decoding is used. Sampling and beam search are not used.

## Output Artifacts

Expected output:

```text
metrics.json
metadata.json
runtime.json
config_snapshot.json
environment_snapshot.json
tokenizer_diagnostics.json
train_predictions.jsonl
val_predictions.jsonl
reload_val_predictions.jsonl
train_problematic_token_dumps.jsonl
val_problematic_token_dumps.jsonl
predictions.jsonl
sample_predictions.jsonl
hardest_examples.jsonl
epoch_dumps/
  epoch_001_train_predictions.jsonl
  epoch_001_val_predictions.jsonl
  ...
```

Each prediction row includes:

- raw generated token IDs;
- decoded string with special tokens;
- decoded string without special tokens;
- NFC-normalized decoded string;
- `repr()` strings;
- Unicode codepoint dumps;
- replacement-character counts;
- repeated-token/empty-prediction flags;
- deterministic error buckets.

## Success Criteria

Expected successful behavior:

- train CER approaches near zero;
- validation CER approaches near zero because validation rows are duplicated from train;
- predictions become readable Cyrillic;
- checkpoint reload predictions remain consistent;
- tokenizer roundtrip diagnostics do not show corruption for normal Cyrillic probes.

## Failure Interpretation

If overfit fails, do not assume the visual OCR model is the only cause.

Likely failure classes:

- tokenizer/model mismatch;
- corrupted references before training;
- broken token-to-text decoding;
- decoder special-token configuration issue;
- generation configuration issue;
- preprocessing/label pipeline bug;
- optimizer or training-loop bug.

Primary files to inspect first:

```text
tokenizer_diagnostics.json
train_problematic_token_dumps.jsonl
val_problematic_token_dumps.jsonl
epoch_dumps/epoch_001_train_sample_predictions.jsonl
epoch_dumps/epoch_040_train_sample_predictions.jsonl
```

Local zero-epoch dry run observation:

- reference replacement-character count was `0`;
- tokenizer roundtrip failures were `0`;
- initial model outputs already contained replacement characters for some rows.

This supports treating replacement-glyph output as a generation/decode/model-path issue unless a future dataset diagnostic contradicts it.

## Post-Run Inspection

```bash
python -m scripts.inspect_ocr_experiment \
  --experiment-dir /kaggle/working/outputs/ocr_overfit_v1 \
  --output-dir /kaggle/working/outputs/ocr_overfit_v1/analysis \
  --worst-k 20
```

Render samples:

```bash
python -m scripts.render_ocr_samples \
  --experiment-dir /kaggle/working/outputs/ocr_overfit_v1 \
  --manifest /kaggle/input/rukopys-ocr-overfit-v1/rukopys_ocr_overfit_v1/manifests/val.jsonl \
  --mode hardest \
  --limit 8 \
  --columns 2 \
  --output /kaggle/working/outputs/ocr_overfit_v1/analysis/hardest_grid.png
```
