# Architectural Hardening Layer

## Scope

This layer adds evaluator-aware internal representation and debugging support around the existing OCR pipeline.

It does not replace the current submission writer and does not introduce PAGE XML, layout-transformer stacks, or page-autoregressive models.

## Added Components

- `src/document/schema.py`
  - `DocumentPage`
  - `DocumentRegion`
  - `DocumentLine`
  - `OCRHypothesis`
  - confidence, lineage, source-coordinate, transform-chain, and order metadata records
- `src/ordering/evaluator_order.py`
  - deterministic line clustering
  - configurable vertical tolerance
  - duplicate suppression hook
  - ordering diagnostics
- `src/debug/geometry.py`
  - failure attribution scaffolding for detection, split/merge, duplicate/overlap, ordering, OCR, and postprocess failures
  - ordering overlay visualization
- `src/experiments/reproducibility.py`
  - config hashing
  - split artifact references
  - preprocessing manifest metadata
  - experiment trace serialization

## Design Constraints

The internal document objects are not submission JSON.

Submission JSON remains a terminal serialization format only. Internal objects preserve:

- stable IDs;
- original/source coordinates;
- transform chain metadata;
- confidence fields;
- OCR n-best hypotheses;
- ordering metadata;
- lineage fields.

## Current Limitations

- Ordering repair is heuristic and must be validated per source.
- Duplicate suppression is conservative and disabled only through config changes, not learned calibration.
- Failure attribution is diagnostic scaffolding, not a complete scorer clone.
- Split/merge attribution uses bbox coverage heuristics and should be treated as a triage signal.
- Confidence values are preserved but not calibrated.

## Intended Use

Use this layer for:

- oracle-detection debugging;
- detector/OCR comparison;
- reading-order diagnostics;
- PageCER-oriented failure analysis;
- reproducible experiment traces.

Do not use this layer to justify larger architecture changes until it identifies a validated bottleneck.
