# OCR Smoke Run 001

## Purpose

Validate the OCR fine-tuning execution path on Kaggle GPU.

This run is not a model-quality benchmark and must not be used for OCR model selection.

## Status

Successful.

Validated:

- Kaggle GPU execution;
- smoke dataset mount;
- CUDA forward/backward;
- checkpoint save;
- checkpoint reload;
- inference after reload;
- CER evaluation path.

## Input Dataset

Kaggle Dataset:

```text
rukopys-ocr-smoke-v1
```

Expected root:

```text
/kaggle/input/rukopys-ocr-smoke-v1/rukopys_ocr_smoke_v1
```

Local source artifact:

```text
data/ocr_smoke/rukopys_ocr_smoke_v1.zip
sha256: d31d6572307dedfe9a318bf209dde85aace09295d7d681c7433a1a98d272eefb
```

Subset:

- train rows: 128;
- val rows: 32;
- eval rows after reload: 8;
- deterministic seed: 42;
- max token length policy: 96;
- truncation allowed: false.

## Command

```bash
python -m scripts.train_ocr_smoke \
  --train-manifest /kaggle/input/rukopys-ocr-smoke-v1/rukopys_ocr_smoke_v1/manifests/train.jsonl \
  --val-manifest /kaggle/input/rukopys-ocr-smoke-v1/rukopys_ocr_smoke_v1/manifests/val.jsonl \
  --output-dir /kaggle/working/outputs/ocr_smoke_gpu \
  --device cuda \
  --train-limit 128 \
  --val-limit 32 \
  --eval-limit 8 \
  --batch-size 4 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --max-target-length 96 \
  --max-new-tokens 96
```

Config file:

```text
configs/ocr_smoke_kaggle_gpu_v1.json
```

## Observed Metrics

```text
probe_loss_before=0.6510433554649353
probe_loss_after=0.1226029321551323
loss_decreased=True
reloaded_mean_cer=0.12944897048598875
```

Interpretation:

- loss decrease validates gradient/update plumbing on the fixed smoke subset;
- reloaded CER validates checkpoint reload and inference path;
- CER is not a trustworthy benchmark because the subset is tiny and not designed for model selection.

## Environment

Observed from the successful Kaggle run:

- GPU: Tesla T4 x2;
- CUDA: available;
- HuggingFace model download: successful;
- model: `cyrillic-trocr/trocr-handwritten-cyrillic`.

Runtime package strategy:

```text
requirements-kaggle-ocr.txt
```

The original smoke run used a heavy lock file. That file has been removed from the recommended workflow because it replaced the Kaggle torch/CUDA stack and pinned core numerical libraries.

The training script writes runtime metadata into:

```text
/kaggle/working/outputs/ocr_smoke_gpu/metadata.json
/kaggle/working/outputs/ocr_smoke_gpu/metrics.json
```

Use those files as the source of truth for exact package/CUDA versions from each future Kaggle run.

Also run:

```bash
python -m scripts.verify_kaggle_ocr_environment --require-cuda
```

## Output Structure

Expected files:

```text
/kaggle/working/outputs/ocr_smoke_gpu/
  checkpoint/
  metadata.json
  metrics.json
  reloaded_predictions.json
```

`checkpoint/` contains the saved model and processor used for reload validation.

## Known Warnings

Previously observed:

```text
The following generation flags are not valid...
```

Mitigation:

- TrOCR inference now uses `max_new_tokens` instead of `max_length`;
- the wrapper no longer mutates `model.generation_config.max_length`;
- greedy decoding sets `early_stopping=False`.

If this warning reappears, record the full warning text in the run notes and inspect `model.generation_config`.

## Limitations

- The validation subset is too small for model-quality claims.
- The smoke dataset is filtered by max token length, so it is not distribution-complete.
- Validation trust remains weak until larger image-isolated validation subsets are analyzed.
- No detector, ordering, or submission-level evaluation is exercised by this run.
