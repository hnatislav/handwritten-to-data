# OCR Region Filtering Policy

## Purpose

Define the first deterministic OCR crop export policy for RUKOPYS `gt_only/train`.

This policy is for OCR recognizer adaptation only. It is not a final document-understanding policy.

## Split Policy

The first crop dataset uses an image-level deterministic split:

- split key: `image_id`;
- hash: SHA-256 with fixed seed;
- default seed: `42`;
- default validation fraction: `0.2`;
- no random region-level split.

Reason:
- OCR crops from the same source page share handwriting, scan quality, layout, and context;
- region-level random splitting would leak page/writer/document style into validation.

This is an initial isolation policy. Stronger group-aware splitting should be introduced after metadata distribution analysis.

## Included Regions

Default included region types:

- `handwritten`;
- `printed`.

Default included metadata:

- `language == "uk"`;
- `legibility == "legible"`;
- non-empty text.

Reason:
- first recognizer experiment should target plain OCR text;
- formulas/tables/annotations require specialized handling;
- illegible or non-Ukrainian regions would distort early CER interpretation.

## Excluded Regions

Excluded by default:

- `formula`;
- `table`;
- `annotation`;
- `image`;
- `graph`;
- non-`uk` language;
- non-`legible` regions;
- empty text;
- invalid bounding boxes.

Rationale:
- `image` and `graph` have empty text by definition;
- formulas use LaTeX-like targets and should be a separate experiment;
- tables need structural handling, not plain line OCR;
- annotations may be short marks/grades with different visual distribution;
- non-legible regions should not be used to teach a first recognizer baseline.

## Crop Policy

Default crop behavior:

- use GT bbox;
- clamp padded bbox to image bounds;
- default padding: `8` pixels;
- save deterministic PNG crops;
- preserve text exactly as annotated.

No preprocessing is applied beyond deterministic RGB conversion and crop extraction.

Avoided for the first export:

- binarization;
- deskewing;
- denoising;
- contrast enhancement;
- handwriting-specific augmentation.

Reason:
preprocessing must be evaluated as a controlled experiment after a baseline recognizer exists.

## Artifacts

The exporter writes:

- `crops/train/*.png`;
- `crops/val/*.png`;
- `manifests/train.jsonl`;
- `manifests/val.jsonl`;
- `manifests/all.jsonl`;
- `manifests/all.csv`;
- `manifests/split_manifest.json`;
- `stats/dataset_stats.json`;
- `logs/metadata.json`.

The manifests include:

- crop path;
- image id;
- region index;
- source bbox;
- text;
- region type;
- language;
- legibility;
- source metadata;
- annotation source;
- image dimensions.

## Reproducibility Requirements

For a valid export, record:

- dataset id;
- dataset config;
- dataset split;
- requested dataset revision if set;
- observed dataset snapshot/revision when inferable;
- seed;
- validation fraction;
- filter config;
- crop padding;
- output statistics.

The generated crop files and manifests are data artifacts and should not be committed to git.
