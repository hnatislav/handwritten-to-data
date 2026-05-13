# Pre-Training Validation Hardening Results

Date: 2026-05-13

## Implemented Utilities

Added:

- `scripts/analyze_ocr_crops.py`
- `scripts/evaluate_ocr_crops.py`
- `scripts/train_ocr_smoke.py`
- `src/ocr/crop_manifest.py`

These utilities operate on exported OCR crop manifests and do not change detection or submission logic.

## OCR Crop Analysis

Command:

```bash
.venv/bin/python -m scripts.analyze_ocr_crops
```

Artifacts:

- `data/ocr_crops/rukopys_gt_only_train_v1/stats/ocr_crop_analysis.json`
- `data/ocr_crops/rukopys_gt_only_train_v1/stats/ocr_crop_analysis.md`

Key findings:

- tokenizer: `cyrillic-trocr/trocr-handwritten-cyrillic`
- train rows: `17635`
- val rows: `3950`
- tokenizer `<unk>` count: `0` on both train and val manifests
- train token length p95/p99/max: `64 / 80 / 571`
- val token length p95/p99/max: `68 / 80.51 / 127`
- exact train/val text overlap strings: `677`
- val rows whose exact text appears in train: `1148`

Interpretation:

- tokenizer coverage is suitable for this crop dataset;
- at least one train target exceeds normal model length limits and requires truncation/filtering policy;
- exact text overlap is substantial, especially for repeated dictation/school strings, so validation CER may overestimate generalization.

## Frozen Region-Level OCR Evaluation

Command:

```bash
.venv/bin/python -m scripts.evaluate_ocr_crops \
  --limit 32 \
  --output-dir outputs/experiments/003_frozen_cyrillic_trocr_val_crops_32
```

Result:

```text
rows = 32
mean region CER = 0.408491
```

Grouped findings on this tiny subset:

- school: `0.570370`
- university: `0.384865`
- all rows were handwritten and Ukrainian in this subset

Interpretation:

- frozen Cyrillic TrOCR is a meaningful baseline, but still far from sufficient;
- the evaluation utility works and writes hardest examples;
- this 32-row subset is a smoke reference, not a stable validation score.

## Tiny Training Smoke Test

Command:

```bash
.venv/bin/python -m scripts.train_ocr_smoke \
  --train-limit 1 \
  --val-limit 1 \
  --eval-limit 1 \
  --batch-size 1 \
  --epochs 5 \
  --learning-rate 1e-6 \
  --output-dir outputs/experiments/004_ocr_train_smoke_cpu_overfit1
```

Result:

```text
probe_loss_before = 0.284878
probe_loss_after  = 0.124371
loss_decreased    = true
```

The script also:

- builds dataloaders;
- tokenizes labels;
- runs forward/backward/optimizer steps;
- saves a checkpoint;
- reloads checkpoint through the inference wrapper;
- runs post-reload inference.

Interpretation:

- training plumbing is functional;
- checkpoint save/reload path is functional;
- CPU smoke is not representative of real training stability;
- a 64-256 crop GPU smoke should be run before a serious baseline fine-tune.

## Current Validation Risks

Remaining risks:

- current split is image-level but not source/group/family isolated;
- exact text overlap between train and val is high;
- printed val coverage is too small for meaningful printed OCR evaluation;
- repeated dictation text may make validation easier than private test behavior;
- long target outliers need a clear max-length policy.

## Recommendation

The project is ready for a GPU smoke fine-tuning run, not yet a serious baseline run.

Next controlled step:

1. add max target length policy, probably filter or truncate above a documented percentile;
2. run `scripts.train_ocr_smoke.py` on GPU with `64-256` train crops and `32-64` val crops;
3. confirm fixed probe loss decreases on a small mixed subset;
4. run frozen-vs-smoke checkpoint region CER on the same validation subset.

Only after that should we run the first 2k-5k crop baseline fine-tune.
