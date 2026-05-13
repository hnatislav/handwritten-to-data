# OCR Baseline Notes

## Current Baseline

Date: 2026-05-12

Recognizer:
- `microsoft/trocr-small-handwritten`
- frozen inference only;
- no fine-tuning;
- oracle detection mode using GT bounding boxes;
- streaming `gt_only/train` subset.

Command:

```bash
.venv/bin/python -m scripts.run_oracle_baseline \
  --limit 3 \
  --streaming \
  --recognizer trocr \
  --trocr-model microsoft/trocr-small-handwritten \
  --output-dir outputs/experiments/001_trocr_small_oracle_3pages
```

## Result

Local oracle-detection PageCER on 3 streaming pages:

```text
mean_page_cer = 1.415457
```

Per-page:

```text
stream_000000.jpg = 1.362874
stream_000001.jpg = 1.321212
stream_000002.jpg = 1.499603
```

## Qualitative Finding

The model does not produce usable Ukrainian handwriting OCR out of the box.

Observed behavior:
- English hallucinated phrases;
- repeated generic patterns such as alphabet-like sequences;
- no meaningful Ukrainian transcription;
- failure occurs even with perfect/oracle GT region boxes.

Interpretation:
- this is an OCR recognizer/domain-language failure, not a detection failure;
- the result should not be used to judge region detection or reading-order quality;
- frozen English-oriented HTR is not a useful final recognizer for RUKOPYS without adaptation.

## Implication

Next OCR candidates should prioritize Ukrainian/Cyrillic support or fine-tuning readiness.

Do not optimize preprocessing based on this baseline yet. The dominant failure is model-language/domain mismatch, not proven image preprocessing failure.
