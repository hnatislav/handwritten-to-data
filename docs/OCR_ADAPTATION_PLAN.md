# OCR Adaptation Plan

## Current Evidence

The first frozen OCR baseline isolated OCR as the dominant bottleneck under oracle detection.

Observed local results:

| Recognizer | Setup | Pages | Mean PageCER | Main behavior |
| --- | --- | ---: | ---: | --- |
| `microsoft/trocr-small-handwritten` | oracle GT boxes, streaming train subset | 3 | 1.415457 | English hallucinations; unusable for Ukrainian HTR |
| `cyrillic-trocr/trocr-handwritten-cyrillic` | oracle GT boxes, streaming train subset | 1 | 0.319760 | Cyrillic/Ukrainian-like output with many substitutions |

These numbers are small-subset smoke results, not stable validation metrics.

## Research Findings

### TrOCR Fine-Tuning

TrOCR is a vision-encoder/text-decoder OCR model. HuggingFace documents it as a `VisionEncoderDecoderModel`; the official notes describe fine-tuning on labeled datasets and using `generate()` for OCR inference.

Implication:
- TrOCR fine-tuning is feasible with our current infrastructure;
- line/region crops from RUKOPYS can be used as image-to-text pairs;
- evaluation can use region CER and oracle-detection PageCER.

Risk:
- vanilla Microsoft TrOCR handwriting checkpoints are not Ukrainian/Cyrillic-specialized enough;
- decoder/tokenizer choice matters.

### Cyrillic TrOCR Checkpoints

The model `cyrillic-trocr/trocr-handwritten-cyrillic` is documented as a fine-tuned version of `kazars24/trocr-base-handwritten-ru`, itself based on `microsoft/trocr-base-handwritten`. The model card states intended use for Church Slavonic, Russian, and Ukrainian handwriting and reports validation CER `0.253278` on its own data.

Implication:
- this is the best currently identified TrOCR starting point;
- it already decodes Cyrillic and Ukrainian characters cleanly;
- adaptation to RUKOPYS should require less data than adapting English-oriented TrOCR from scratch.

Risk:
- its training data is not RUKOPYS;
- its reported CER is from its own dataset and cannot be assumed on RUKOPYS;
- it is 0.3B params, slower on CPU and should be trained/evaluated on GPU for real experiments.

### PaddleOCR Cyrillic Support

PaddleOCR PP-OCRv5 documents multilingual recognition, including Cyrillic/East-Slavic support. The documentation lists:
- East Slavic language model trained/evaluated on Russian, Belarusian, and Ukrainian text images;
- Cyrillic recognition model;
- CLI/Python `lang`-based usage.

Implication:
- PaddleOCR is a realistic non-TrOCR OCR candidate;
- it may provide a useful inference-only Cyrillic baseline;
- it may be stronger on printed/general OCR than handwritten RUKOPYS, so it must be tested under oracle crops before any conclusion.

Risk:
- PaddleOCR dependencies may be heavier than TrOCR in the current Python 3.14 environment;
- documented model accuracy is not handwritten Ukrainian RUKOPYS accuracy;
- integration complexity is medium because PaddleOCR has its own runtime and model format.

## Tokenizer Compatibility

Local tokenizer test strings:

```text
Магія голосу.
Україна єднає: ґанок, їжак, Європа, апострофʼ
Спочатку був "брехунець". З нього почи-
```

Results:

| Model | Tokenizer | Ukrainian test result |
| --- | --- | --- |
| `microsoft/trocr-small-handwritten` | `XLMRobertaTokenizer` | mostly works, but produced 2 `<unk>` tokens on `ґ/Є` sample and decoded with lost characters |
| `cyrillic-trocr/trocr-handwritten-cyrillic` | `RobertaTokenizer` | 0 `<unk>` tokens; decoded Ukrainian sample cleanly |

Interpretation:
- for Microsoft TrOCR small, failures are partly decoder/tokenizer-language related and strongly visual/domain related;
- for Cyrillic TrOCR, tokenizer coverage is not the immediate blocker;
- Cyrillic TrOCR errors on RUKOPYS are more likely visual-domain mismatch, handwriting style mismatch, and language/model adaptation issues.

## Candidate Ranking

### Candidate A — Fine-tune Cyrillic TrOCR

Priority: first.

Why:
- already Ukrainian/Cyrillic-capable;
- compatible with current `TrOCRRecognizer` wrapper;
- direct continuation of current oracle OCR evaluation;
- likely lowest implementation cost for a meaningful Ukrainian-capable baseline.

Expected compute:
- small experiment: single GPU, likely feasible in minutes to a few hours depending subset size;
- H100 is more than sufficient;
- local CPU is only suitable for smoke tests, not training.

Implementation complexity:
- medium-low.

### Candidate B — PaddleOCR Cyrillic / East-Slavic Inference

Priority: second, as an inference baseline.

Why:
- official multilingual/Cyrillic support;
- may be useful for printed regions or as an ensemble component later.

Expected compute:
- inference only should be modest;
- install/runtime compatibility must be tested.

Implementation complexity:
- medium.

### Candidate C — Fine-tune Microsoft TrOCR Small/Base

Priority: lower.

Why:
- current small checkpoint shows bad Ukrainian behavior;
- tokenizer has partial Ukrainian coverage issues;
- starting from Cyrillic TrOCR is better justified.

Use only if:
- Cyrillic TrOCR has licensing/runtime issues;
- we explicitly want a controlled comparison.

## First Minimal Fine-Tuning Experiment

### Hypothesis

Fine-tuning `cyrillic-trocr/trocr-handwritten-cyrillic` on clean RUKOPYS line/region crops will reduce oracle-detection region CER and PageCER versus frozen Cyrillic TrOCR.

### Scope

Train OCR recognizer only.

Do not change:
- detection;
- reading order;
- preprocessing, except deterministic crop extraction;
- postprocessing.

### Data

Use only human-labeled `gt_only/train` regions.

Initial filters:
- `type in {"handwritten", "printed"}`;
- `language == "uk"`;
- `legibility == "legible"`;
- non-empty `text`;
- bbox valid after clamping;
- exclude `image`, `graph`, `formula`, `table`, `annotation` for first OCR experiment.

Initial subset:
- train: 2,000-5,000 cropped regions;
- validation: 300-500 cropped regions;
- split by image, not by region.

Rationale:
- enough to verify learning signal;
- small enough to iterate quickly;
- avoids overfitting to a few pages as much as possible.

### Objective

Sequence-to-sequence cross-entropy over target text tokens.

Metrics:
- region CER on validation crops;
- oracle-detection PageCER on validation pages;
- per-source/per-region-type breakdown where available.

### Training Config Draft

Start conservative:
- model: `cyrillic-trocr/trocr-handwritten-cyrillic`;
- image processor/tokenizer from same checkpoint;
- max target length from dataset percentile, not arbitrary;
- batch size: start 8-16 depending GPU memory;
- epochs: 1-3 for first smoke experiment;
- learning rate: small, e.g. `1e-5` to `5e-5`;
- mixed precision on GPU;
- fixed seed;
- save best checkpoint by validation CER.

Do not use `silver` in first experiment.

### Evaluation Methodology

Run three comparisons on the same validation pages:

1. frozen Cyrillic TrOCR;
2. fine-tuned Cyrillic TrOCR;
3. oracle-text recognizer sanity check.

Report:
- region CER;
- PageCER;
- CER by text length bucket;
- CER by source;
- CER by region type;
- examples of best/worst changes.

### Overfitting Risks

- random region split leaks page/writer/document style;
- small validation subset can be unstable;
- national dictation text may repeat or be known;
- model may memorize common lines;
- aggressive training on narrow source distribution may hurt archive/school domains.

Mitigation:
- image-level validation split;
- log source distribution;
- keep smoke validation and main validation separate;
- inspect hardest samples, not only average CER.

## Immediate Next Implementation Steps

1. Build crop dataset exporter for filtered OCR regions.
2. Add tokenizer statistics over all training text.
3. Add region-level CER evaluation.
4. Add fine-tuning script for `VisionEncoderDecoderModel`.
5. Run a tiny overfit test on 32-64 crops to verify training loop.
6. Run the first controlled 2k-5k region fine-tune.
