# Kaggle OCR Smoke Training Workflow

## Purpose

Validate OCR fine-tuning infrastructure on Kaggle GPU before any serious training run.

This workflow is intentionally small. It is not a leaderboard experiment.

## Inputs

Repository:

```text
git@github.com:hnatislav/handwritten-to-data.git
```

Smoke dataset artifact:

```text
rukopys_ocr_smoke_v1.zip
```

The artifact is generated locally and is not committed to git because `/data/` and archive files are ignored. Upload this zip as a Kaggle Dataset named `rukopys-ocr-smoke-v1`.

Local artifact size and checksum:

```text
size: ~44 MB
sha256: d31d6572307dedfe9a318bf209dde85aace09295d7d681c7433a1a98d272eefb
```

Expected unzipped structure:

```text
/kaggle/input/rukopys-ocr-smoke-v1/rukopys_ocr_smoke_v1/
  crops/
    train/*.png
    val/*.png
  manifests/
    train.jsonl
    val.jsonl
    all.jsonl
  stats/smoke_dataset_stats.json
  dataset_info.json
  checksums.json
```

## Kaggle Notebook Cells

### 1. Clone repository

```bash
!git clone git@github.com:hnatislav/handwritten-to-data.git /kaggle/working/handwritten-to-data
%cd /kaggle/working/handwritten-to-data
```

If SSH is not configured in the notebook, use the HTTPS URL for read-only clone.

### 2. Install dependencies

```bash
!pip install -r requirements-kaggle-ocr-smoke.lock.txt
```

### 3. Verify GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

Expected:

```text
True
2
Tesla T4
```

### 4. Set smoke dataset paths

```python
SMOKE_ROOT = "/kaggle/input/rukopys-ocr-smoke-v1/rukopys_ocr_smoke_v1"
TRAIN_MANIFEST = f"{SMOKE_ROOT}/manifests/train.jsonl"
VAL_MANIFEST = f"{SMOKE_ROOT}/manifests/val.jsonl"
```

### 5. Inspect smoke dataset stats

```bash
!cat /kaggle/input/rukopys-ocr-smoke-v1/rukopys_ocr_smoke_v1/stats/smoke_dataset_stats.json
```

### 6. Run OCR smoke training

```bash
!python -m scripts.train_ocr_smoke \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
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

Expected validation:

- dataloader works;
- tokenizer works;
- no target exceeds `max_target_length=96`;
- CUDA forward/backward works;
- checkpoint is saved;
- checkpoint reloads;
- inference after reload runs;
- `metrics.json` is written.

Do not interpret the CER as model quality. This run is an infrastructure validation.

Successful run 001 observed:

```text
probe_loss_before=0.6510433554649353
probe_loss_after=0.1226029321551323
loss_decreased=True
reloaded_mean_cer=0.12944897048598875
```

See `docs/OCR_SMOKE_RUN_001.md`.

### 7. Inspect output artifacts

```bash
!find /kaggle/working/outputs/ocr_smoke_gpu -maxdepth 2 -type f | sort
!cat /kaggle/working/outputs/ocr_smoke_gpu/metrics.json
```

## Target-Length Policy

The smoke dataset is constructed with:

```text
max_token_length <= 96
```

The training script now fails if selected rows exceed `--max-target-length`, unless `--allow-truncation` is explicitly passed.

For smoke training, do not pass `--allow-truncation`.

Reason:

- silent target truncation invalidates loss interpretation;
- max-length policy must be explicit before larger runs.

## Reproducibility Notes

The smoke dataset is deterministic:

- source manifests: exported OCR crop train/val manifests;
- seed: `42`;
- train rows: `128`;
- val rows: `32`;
- tokenizer/model used for filtering: `cyrillic-trocr/trocr-handwritten-cyrillic`;
- max token length: `96`.

Current local artifact stats:

```text
train rows: 128
val rows: 32
token_length max: 88
text_length max: 80
```

Local CPU path validation was run on a smaller subset:

```bash
.venv/bin/python -m scripts.train_ocr_smoke \
  --train-manifest data/ocr_smoke/rukopys_ocr_smoke_v1/manifests/train.jsonl \
  --val-manifest data/ocr_smoke/rukopys_ocr_smoke_v1/manifests/val.jsonl \
  --output-dir outputs/experiments/005_local_smoke_dataset_train_check \
  --device cpu \
  --train-limit 4 \
  --val-limit 2 \
  --eval-limit 1 \
  --batch-size 1 \
  --epochs 1 \
  --learning-rate 1e-6 \
  --max-target-length 96 \
  --max-new-tokens 96
```

Result:

```text
probe_loss_before=0.1101785
probe_loss_after=0.0685570
loss_decreased=True
checkpoint reload inference completed
```

Use `checksums.json` to verify uploaded dataset integrity if needed.

## Known Generation Warning

If this warning appears:

```text
The following generation flags are not valid...
```

record the full warning text in the run notes.

The current TrOCR wrapper uses `max_new_tokens` and does not mutate `generation_config.max_length`.

## Limitations

- The smoke subset is not a representative validation set.
- Loss decrease on smoke data is not evidence of generalization.
- CER after one epoch on 128 crops should not be used for model selection.
- This workflow validates infrastructure only.
