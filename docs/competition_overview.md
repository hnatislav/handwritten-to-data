# Handwritten to Data — Technical Overview

## Competition Goal

Build an AI system for recognizing and extracting structured information from Ukrainian handwritten documents.

The task is not limited to OCR only.
The expected system may include:
- layout analysis,
- region detection,
- document understanding,
- handwriting recognition,
- structured extraction,
- postprocessing.

The competition focuses on:
- robustness across writing styles,
- real-world noisy data,
- reproducibility,
- deployment readiness.

The target use cases include:
- governmental forms,
- applications,
- certificates,
- archival materials,
- educational documents,
- handwritten notes,
- signatures,
- stamps.

---

# Core Constraints

## Inference Constraints

The full inference pipeline must:
- use only open-weight/open-source models,
- fit into a single NVIDIA H100 80GB GPU.

Forbidden during inference/submission:
- OpenAI APIs,
- Anthropic APIs,
- Gemini APIs,
- proprietary hosted inference services.

Training and data preparation may use external APIs.

---

# Dataset — RUKOPYS

Dataset:
UkrainianCatholicUniversity/rukopys

The dataset spans more than 100 years of Ukrainian handwriting.

## Document Sources

### National Dictation (2020–2025)
- phone photos,
- large handwriting diversity,
- canonical known text.

### State Archive (1919–1935)
- historical archival documents,
- pen & ink,
- archaic orthography,
- scanned materials.

### University / KNUTE (2022–2025)
- student exam work,
- formulas,
- chemistry,
- tables.

### School Homework (2022–2025)
- grades 5–11,
- 20+ subjects,
- phone photos.

---

# Dataset Splits

Known split counts from the competition site:

| Split | Images | GT Regions | annotation_source | Description |
| --- | ---: | ---: | --- | --- |
| train | 1,330 | 25,523 | annotator / volunteer | Human-annotated — full bboxes + verified transcription |
| silver | 8,207 | 161,065 | auto | Auto-annotated by Qwen3-VL 8B + Gemini — for self-training |
| test | 386 | hidden | — | Images only — submit predictions to Kaggle |
| private benchmark | 21 | hidden until June 15 | — | Held-out set withheld during the competition; published after the online stage closes as a reusable community benchmark |

## Train
Human-annotated:
- bounding boxes,
- region types,
- transcriptions.

## Silver
Auto-annotated data intended for:
- self-training,
- pseudo-labeling,
- curriculum learning.

## Test
Images only.
Predictions are submitted to Kaggle.

---

# Annotation Structure

Each region contains:
- bbox,
- type,
- text.

## Bounding Box Format

[x1, y1, x2, y2]

Pixel coordinates with top-left origin.

## Region Types

Allowed classes:
- handwritten
- printed
- formula
- table
- annotation
- image
- graph

## Text Rules

- image and graph regions should use empty text.
- all images must appear in submission.
- missing images cause scoring failure.

---

# Evaluation Metric

Final score:

Score =
0.15 × Detection_F1
+ 0.05 × ClassAcc
+ 0.30 × (1 − CER)
+ 0.50 × (1 − PageCER)

---

# Metric Components

## Detection F1 (15%)

Measures:
- region localization quality,
- IoU threshold = 0.5,
- type-agnostic detection.

Global precision/recall/F1 across all images.

## Classification Accuracy (5%)

Among IoU-matched regions:
- correct region type classification.

## Region CER (30%)

Character Error Rate on matched regions.

Uses:
- Levenshtein distance,
- normalized by GT text length.

Only evaluated on:
- language=uk,
- legible regions,
- non-image,
- non-graph regions.

## Page CER (50%)

The most important metric component.

Procedure:
- regions sorted in reading order,
- concatenated into full-page text,
- compared using Levenshtein distance.

Important implication:
exact bbox precision is less important than correct final page text ordering and transcription quality.

This strongly favors:
- robust OCR,
- reading-order consistency,
- postprocessing,
- language correction,
- text reconstruction quality.

---

# Text Normalization Rules

Normalization is applied before CER evaluation.

Includes:
- strikethrough normalization,
- LaTeX symbol conversion,
- Latin/Cyrillic lookalike normalization,
- dash normalization,
- quote normalization,
- superscript/subscript normalization,
- whitespace collapsing.

Implication:
the pipeline should avoid overfitting to raw formatting artifacts.

---

# Submission Format

CSV with columns:

- image
- regions

Example:

image,regions
abc.jpg,"[{""bbox"":[50,100,850,130],""type"":""handwritten"",""text"":""Доброго ранку""}]"

---

# Technical Implications

The task is effectively a hybrid of:
- OCR,
- document layout analysis,
- sequence recognition,
- structured extraction.

This is NOT a pure text-line OCR benchmark.

The dominant metric component is PageCER (50%).
This may favor:
- end-to-end document models,
- strong text reconstruction,
- language-aware decoding,
- aggressive postprocessing.

Detection quality matters less than transcription quality.

---

# Recommended Directions from Organizers

## Vision-Language Models

Examples:
- Qwen3-VL,
- Gemma 4,
- LLaMA-based VLMs.

## Agentic Pipelines

Multi-stage routing systems:
- classify region/document,
- dispatch specialized OCR strategies.

## Retrieval-Augmented Recognition

Using:
- dictionaries,
- templates,
- lexical priors,
- domain-specific retrieval.

## Synthetic Data

Strongly encouraged.

## Pseudo-labeling

Strongly encouraged.

---

# External Data

Allowed:
- public datasets,
- open-source pretrained models,
- synthetic data,
- pseudo-labeling.

Suggested datasets:
- IAM Handwriting DB,
- HKR dataset,
- historical Cyrillic datasets.

---

# Suggested Models

## End-to-End

- Qwen3-VL
- Gemma 4
- PaddleOCR
- Donut
- Florence
- LayoutLM-family

## Text-Line OCR

- TrOCR
- PARSeq
- CRNN-based systems
- Cyrillic HTR checkpoints

NOTE:
These are recommendations/hypotheses, not validated optimal solutions.

---

# Important Strategic Observations

## PageCER Dominance

Since PageCER has 50% weight:
- correct reading order is critical,
- robust transcription is more important than precise boxes,
- fragmented detection may hurt reconstruction quality.

## Historical Documents

Archive data introduces:
- archaic spelling,
- degraded scans,
- unusual handwriting,
- ink artifacts.

Generalization risk is high.

## Multi-Domain OCR

The system must handle:
- modern handwriting,
- historical handwriting,
- formulas,
- tables,
- mixed layouts.

This increases:
- distribution shift risk,
- overfitting risk,
- routing complexity.

---

# Likely High-Value Areas

Potentially high ROI areas:
- synthetic handwriting generation,
- self-training on silver split,
- pseudo-labeling,
- language-model correction,
- reading-order optimization,
- ensemble decoding,
- detector/OCR specialization,
- curriculum learning.

---

# Verification Risks

Potential disqualification risks:
- non-reproducible training,
- hidden dependencies,
- proprietary inference APIs,
- excessive VRAM usage,
- unstable preprocessing,
- nondeterministic inference.

---

# Open Questions

Unknowns requiring analysis:
- annotation consistency,
- bbox quality variance,
- region class imbalance,
- handwriting distribution,
- OCR normalization edge cases,
- test-time document distribution,
- prevalence of formulas/tables,
- optimal reading-order heuristics.

# Primary Optimization Target

The dominant optimization target is likely:
- low PageCER,
- stable reading order,
- robust transcription quality.

Pure object-detection optimization is unlikely to be sufficient.

Because PageCER has the highest metric weight, robust transcription and reading-order consistency are likely more important than perfect bbox precision.

Hypothesis:
excessively fragmented detection may negatively affect PageCER through reading-order instability.

---

# Leakage Risks

The National Dictation subset contains canonical publicly known texts.

Care must be taken to avoid:
- accidental memorization,
- benchmark contamination,
- unfair alignment strategies,
- implicit test leakage through text retrieval.

Any retrieval-based or alignment-based approach should be carefully validated against private leaderboard generalization.
