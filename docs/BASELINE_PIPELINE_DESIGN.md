# Minimal End-to-End Baseline Pipeline

## Objective

Build the first reproducible end-to-end baseline for RUKOPYS.

The baseline is not intended to be SOTA. It is intended to:
- process images end-to-end;
- generate valid Kaggle submissions;
- support local evaluation;
- expose failure modes;
- provide a stable foundation for controlled iteration.

Primary optimization target:
- PageCER.

Primary engineering target:
- debuggability and reproducibility.

---

## Design Principles

- Keep the first baseline simple.
- Prefer explicit intermediate files over hidden state.
- Avoid giant VLM orchestration.
- Avoid premature model complexity.
- Make every pipeline stage independently runnable.
- Log inputs, outputs, config, and dataset revision.
- Treat train, silver, and test differently.
- Do not mix human GT and silver labels without an explicit experiment.

---

## Minimal Modular Architecture

```text
image
  -> dataset loader
  -> preprocessing
  -> region proposal / detection
  -> region cropping
  -> OCR / transcription
  -> reading-order reconstruction
  -> prediction assembly
  -> local evaluation
  -> submission writer
```

Recommended first implementation modules:

```text
src/
  config.py
  data/
    rukopys_dataset.py
    schema.py
    validation_split.py
  preprocessing/
    image_ops.py
  detection/
    oracle.py
    simple_regions.py
  ocr/
    recognizer.py
    dummy.py
  ordering/
    reading_order.py
  evaluation/
    cer.py
    metrics.py
  submission/
    writer.py
  experiments/
    logging.py
```

For the first baseline, implementation can be simpler than this full layout, but these ownership boundaries should be preserved.

---

## Pipeline Stages

### 1. Dataset Loader

Inputs:
- HuggingFace dataset id: `UkrainianCatholicUniversity/rukopys`;
- config/split selection;
- dataset revision if available.

Outputs:
- image object or image path;
- image id / file name;
- image width and height;
- source metadata;
- annotation_source;
- regions when available.

Requirements:
- preserve raw metadata;
- do not assume all splits have annotations;
- support `gt_only/train` and `test/test` first;
- handle `silver` only after a verified loading path exists.

### 2. Preprocessing

First baseline:
- no aggressive preprocessing;
- deterministic RGB conversion;
- optional max-size resize only if required by downstream model;
- exact bbox scaling when resizing.

Avoid initially:
- binarization;
- denoising;
- contrast enhancement;
- deskewing;
- crop heuristics.

Rationale:
preprocessing may improve or damage handwriting differently across archive scans, phone photos, pencil, stamps, formulas, and tables. It should become an experiment, not a hidden default.

### 3. Region Detection

Two baseline modes are needed:

#### Oracle Detection Mode

Use ground-truth bboxes and region types on validation data.

Purpose:
- isolate OCR and reading-order errors;
- estimate upper bound if detection were perfect;
- debug PageCER assembly.

Not valid for test submission.

#### Simple Test-Time Detection Mode

First valid submission should use the simplest available non-oracle detector.

Candidate options:
- lightweight open-source OCR/layout detector;
- simple line/region proposal baseline;
- trained detector only after validation infrastructure is ready.

Decision should be made after confirming installed tooling and runtime constraints.

### 4. Region Cropping

Inputs:
- image;
- bbox `[x1, y1, x2, y2]`;
- optional padding config.

Outputs:
- cropped region image;
- crop metadata;
- transformed bbox if any resize is applied.

Requirements:
- clamp bbox to image bounds;
- log invalid boxes;
- do not silently drop regions unless explicitly configured.

### 5. OCR / Transcription

First baseline should support two recognizer modes:

#### Dummy / Empty Recognizer

Returns empty text.

Purpose:
- validate submission format;
- validate evaluation code;
- validate pipeline wiring.

Expected performance:
- poor OCR/PageCER.

#### Simple OCR Recognizer

Use a lightweight open-source recognizer for initial text output.

Candidate families:
- TrOCR-like line recognizer;
- PaddleOCR/docTR-style recognizer;
- simple CTC recognizer after a small controlled training run.

Important:
do not assume Ukrainian handwritten quality before evaluation.

### 6. Reading Order

First baseline:
- sort regions top-to-bottom, then left-to-right;
- use configurable vertical grouping threshold;
- preserve deterministic ordering.

Need to evaluate separately because PageCER is highly sensitive to order.

### 7. Prediction Assembly

Each output row:
- `image`;
- `regions`.

Each region:
- `bbox`;
- `type`;
- `text`.

Rules:
- include every test image;
- use empty text for `image` and `graph`;
- serialize JSON deterministically;
- validate output schema before writing CSV.

---

## Data Flow

Internal region record:

```python
{
    "image_id": str,
    "bbox": [int, int, int, int],
    "type": str,
    "text": str,
    "score": float | None,
    "source": str | None,
    "annotation_source": str | None,
    "language": str | None,
    "legibility": str | None,
}
```

Internal page prediction:

```python
{
    "image_id": str,
    "width": int,
    "height": int,
    "regions": list[RegionPrediction],
}
```

Keep raw dataset records separate from normalized internal records.

---

## Validation Strategy Draft

Do not use a random region-level split as the main validation.

Initial strategy:
- split at image level;
- preserve source distribution where possible;
- report metrics per `source`, `annotation_source`, `region type`, `language`, and `legibility`;
- keep one small smoke validation subset for fast iteration;
- keep one stable main validation subset for experiment comparison.

Preferred next step:
- inspect available grouping metadata;
- design a deterministic grouped split based on `source`, `year`, `grade`, `subject`, `group`, `fund`, and file/document identifiers where available.

Validation must support:
- oracle detection evaluation;
- full pipeline evaluation;
- per-component failure analysis.

---

## Local Evaluation Approach

Implement local metrics in stages.

### Stage A — Text Assembly Evaluation

Using oracle regions:
- sort GT regions by chosen reading order;
- concatenate text;
- compute PageCER;
- compare different ordering heuristics.

Purpose:
- validate reading-order assumptions before training models.

### Stage B — OCR Evaluation

Using oracle regions:
- crop GT regions;
- run recognizer;
- compute region CER and PageCER.

Purpose:
- isolate OCR quality from detection quality.

### Stage C — Detection Evaluation

Using predicted regions:
- compute IoU matching at threshold 0.5;
- type-agnostic detection F1;
- classification accuracy on matched regions.

### Stage D — Full Competition Proxy

Combine:
- Detection F1;
- ClassAcc;
- Region CER;
- PageCER.

Use the documented metric weights:

```text
0.15 * Detection_F1
+ 0.05 * ClassAcc
+ 0.30 * (1 - CER)
+ 0.50 * (1 - PageCER)
```

Clearly mark this as a local proxy until verified against official scoring behavior.

---

## Experiment Logging Structure

Suggested output layout:

```text
outputs/
  experiments/
    000_baseline_dummy/
      config.json
      dataset_info.json
      metrics.json
      predictions_val.jsonl
      submission.csv
      failure_samples/
      notes.md
```

Minimum logged fields:
- experiment id;
- timestamp;
- git commit if repository is initialized;
- Python/package versions;
- dataset id and revision;
- config;
- validation split definition;
- metrics;
- runtime;
- known failures.

---

## Expected Bottlenecks

Likely bottlenecks, to be verified:

- OCR quality on Ukrainian handwriting;
- reading-order reconstruction;
- detection of dense handwritten lines;
- handling formulas and tables;
- domain shift between archive, dictation, school, and university sources;
- noisy `silver` labels;
- validation leakage if split strategy is weak.

Do not optimize any one bottleneck until local evaluation identifies it.

---

## Biggest Engineering Risks

- Building model complexity before local metrics are reliable.
- Using `silver` as clean supervision.
- Optimizing public leaderboard without stable validation.
- Silent bbox corruption during resize/crop.
- Dropping test images from submission.
- Treating PageCER as region CER.
- Overfitting preprocessing to a small visual sample.
- Creating notebook-only logic that cannot be reproduced from scripts.
- Depending on hosted proprietary inference in the final pipeline.

---

## First Implementation Milestones

1. Implement normalized dataset records from `gt_only/train`.
2. Implement deterministic validation split.
3. Implement submission writer and schema validator.
4. Implement CER and PageCER.
5. Implement oracle-region evaluation.
6. Implement dummy recognizer baseline.
7. Generate a valid local prediction artifact.
8. Add first non-oracle detector/OCR candidate only after the above works.
