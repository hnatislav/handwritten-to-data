# OCR Next-Stage Experiment Plan

## Objective

Run the first controlled OCR fine-tuning experiment after smoke validation.

Primary goal:

- validate training behavior on a slightly larger subset without weakening validation discipline.

Non-goals:

- leaderboard optimization;
- detector development;
- aggressive hyperparameter search;
- training on all crops.

## Proposed Scope

Dataset:

- train crops: 1,000 to 2,000;
- validation crops: 300 to 500;
- split source: existing deterministic image-level split artifacts;
- no random region-level split.

Model:

```text
cyrillic-trocr/trocr-handwritten-cyrillic
```

Training:

- epochs: 1 to 3;
- batch size: constrained by T4 memory;
- greedy decoding for evaluation;
- no beam search in the first controlled run.

## Target-Length Policy

Before launch:

- compute token length percentiles on selected train and validation rows;
- set `max_target_length` from observed percentile plus margin;
- fail on overflow by default;
- allow truncation only in explicitly labeled diagnostic runs.

Required report fields:

- max target length;
- max observed train token length;
- max observed validation token length;
- overflow count;
- truncation flag.

## Evaluation

Track:

- train loss curve;
- fixed probe-batch loss before/after;
- validation region CER;
- CER grouped by region type, source, language, and text-length bucket;
- hardest validation crops;
- prediction JSONL with crop IDs and references.

Do not treat mean CER as stable unless:

- validation sample is large enough;
- image-level split isolation is verified;
- duplicate/exact-text overlap is measured.

## Runtime Monitoring

Record:

- package versions;
- Python version;
- CUDA availability and version;
- GPU device names;
- peak allocated CUDA memory;
- peak reserved CUDA memory;
- wall-clock runtime.

`scripts.train_ocr_smoke` now records package and CUDA metadata. The next controlled training script should preserve the same fields.

## Leakage Checks

Before training:

- exact text overlap between train and validation;
- crop ID overlap;
- image ID overlap;
- source distribution comparison;
- handwritten/printed distribution comparison.

Current risk:

- validation trust is not yet strong enough for high-confidence model selection.

Mitigation:

- keep experiment size moderate;
- inspect grouped CER and hardest examples;
- compare against frozen OCR baseline on the same validation subset.

## Acceptance Criteria

The next run is acceptable if:

- launch is deterministic from committed config;
- target-length policy is explicit and logged;
- checkpoint reload evaluation runs;
- validation CER artifacts are written;
- runtime/CUDA metadata is captured;
- no region-level leakage is introduced.

The run is not acceptable if:

- silent truncation occurs;
- validation rows are sampled from train images;
- package/runtime versions are not recorded;
- only aggregate CER is saved.
