# Handwritten to Data — Engineering Workflow

## Core Objective

Build a reproducible, modular, competition-grade document understanding system for the RUKOPYS dataset.

Primary optimization target:
- low PageCER,
- robust transcription quality,
- stable reading order reconstruction,
- strong generalization to unseen documents.

The goal is NOT to build the most complex architecture.

The goal is:
- reliable evaluation,
- controlled experimentation,
- stable iteration,
- reproducible improvements.

---

## Core Engineering Principles

Rules:
- Never optimize architecture before identifying the current bottleneck.
- Every experiment must answer a specific question.
- Prefer simple reproducible baselines over complex unstable systems.
- Avoid premature optimization.
- Avoid uncontrolled architectural complexity.
- Do not trust leaderboard gains without validation evidence.
- Validate assumptions through experiments whenever possible.
- Treat public leaderboard improvements cautiously.
- Avoid hidden state and notebook spaghetti code.
- Build systems incrementally and verify each component independently.

---

## Development Workflow

The project should evolve through iterative engineering cycles:

baseline
→ evaluation
→ failure analysis
→ bottleneck identification
→ hypothesis
→ controlled experiment
→ evaluation
→ repeat

The system should improve through:
- understanding failures,
- controlled experiments,
- validation stability,
not through random architecture changes.

---

## Stage 1 — Project Infrastructure

Goals:
- establish reproducible project structure;
- configure environments;
- organize repository;
- establish deterministic workflows.

Requirements:
- modular codebase;
- deterministic preprocessing;
- fixed random seeds;
- configuration separation;
- experiment logging;
- reproducible inference.

Deliverables:
- repo structure;
- requirements/environment setup;
- dataset loaders;
- experiment organization;
- documentation structure.

---

## Stage 2 — Dataset Understanding

Goals:
- understand actual dataset structure and distribution;
- identify risks and edge cases;
- avoid assumptions.

Tasks:
- inspect images manually;
- inspect annotations;
- visualize bounding boxes;
- inspect region types;
- inspect metadata fields;
- analyze text lengths;
- analyze image resolutions;
- inspect document variability.

Important:
do NOT train large models before understanding the dataset.

Deliverables:
- dataset exploration notebook;
- dataset statistics;
- dataset observations;
- documented risks and anomalies.

---

## Stage 3 — Evaluation Infrastructure

Goals:
- establish reliable local evaluation;
- create stable validation methodology.

Tasks:
- reproduce competition metric locally;
- implement validation pipeline;
- design group-aware validation;
- minimize leakage risk;
- establish experiment comparability.

Critical:
validation quality is more important than early leaderboard gains.

Potential risks:
- writer leakage;
- source leakage;
- near-duplicate documents;
- public leaderboard overfitting.

Deliverables:
- local evaluation pipeline;
- validation split strategy;
- experiment evaluation framework.

---

## Stage 4 — Minimal End-to-End Baseline

Goals:
- build the first fully working pipeline;
- establish baseline performance;
- identify bottlenecks.

The first baseline should prioritize:
- simplicity,
- reproducibility,
- debuggability,
not SOTA performance.

Candidate baseline directions:
- simple detector + OCR recognizer;
- PaddleOCR-based pipeline;
- TrOCR-based recognizer;
- lightweight OCR stack.

These are candidate directions, not guaranteed optimal solutions.

Pipeline structure:

image
→ preprocessing
→ region detection
→ OCR/transcription
→ reading-order reconstruction
→ submission generation
→ evaluation

Deliverables:
- working submission pipeline;
- baseline metrics;
- reproducible inference;
- initial failure analysis.

---

## Stage 5 — Failure-Driven Development

Goals:
- identify real bottlenecks;
- avoid blind optimization.

The project should evolve through systematic failure analysis.

Analyze:
- OCR failures;
- detection failures;
- reading-order failures;
- preprocessing sensitivity;
- document-type failures;
- domain-specific failures;
- noisy-label behavior.

Important:
do not assume where the bottleneck is.

Potential bottlenecks:
- OCR quality,
- reading order,
- detection quality,
- preprocessing instability,
- noisy labels,
- validation leakage.

Deliverables:
- failure taxonomy;
- bottleneck identification;
- prioritized improvement targets.

---

## Stage 6 — Controlled Experimentation

Goals:
- improve performance through measurable experiments.

Possible experiment areas:
- preprocessing;
- augmentations;
- better OCR models;
- better detectors;
- pseudo-labeling;
- self-training;
- synthetic handwriting generation;
- language correction;
- postprocessing;
- ensembles.

Experiment Rules:
- change one important variable at a time;
- log every experiment;
- compare against baseline;
- track validation variance;
- document failure cases;
- verify reproducibility.

Each experiment must specify:
- hypothesis;
- expected effect;
- compute cost;
- implementation complexity;
- overfitting risk;
- evaluation methodology.

---

## Stage 7 — Data Expansion

Goals:
- increase robustness and generalization.

Possible directions:
- silver split self-training;
- pseudo-labeling;
- synthetic handwriting;
- external public datasets;
- curriculum learning.

Important:
silver labels are noisy.

Potential risks:
- propagating VLM errors;
- reinforcing incorrect transcriptions;
- confirmation bias;
- domain imbalance.

Deliverables:
- controlled data expansion pipeline;
- quality evaluation of added data.

---

## Stage 8 — Advanced Optimization

Goals:
- improve PageCER and robustness.

Potential directions:
- reading-order optimization;
- postprocessing;
- language-model correction;
- constrained decoding;
- ensemble strategies;
- specialized region handling.

Important:
advanced optimization should only begin after:
- stable baseline,
- reliable validation,
- identified bottlenecks.

---

## Stage 9 — Final Competition System

Goals:
- produce a stable competition-grade system.

Final system requirements:
- reproducible;
- modular;
- robust;
- efficient;
- validation-stable;
- H100-compatible.

Expected final system properties:
- strong generalization;
- stable inference;
- controlled preprocessing;
- reliable evaluation;
- traceable experiments;
- reproducible results.

Most final gains will likely come from:
- data understanding,
- validation quality,
- postprocessing,
- reading-order consistency,
- ensemble robustness,
not from blindly increasing model size.