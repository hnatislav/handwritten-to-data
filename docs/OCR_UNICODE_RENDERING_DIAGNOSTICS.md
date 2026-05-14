# OCR Unicode and Rendering Diagnostics

## Purpose

Separate actual OCR text corruption from visualization/font rendering failures.

Known current evidence:

- tokenizer roundtrip tests pass on the overfit subset;
- reference replacement-character count is zero on the overfit subset;
- PNG grids can still show square glyphs if the renderer uses a font without Cyrillic glyphs.

Therefore rendered PNGs are not sufficient evidence of text corruption.

## Raw Text Inspection

Inspect JSONL text directly:

```bash
python -m scripts.inspect_unicode_text \
  --jsonl /kaggle/working/outputs/ocr_overfit_v1/val_predictions.jsonl \
  --limit 8 \
  --output-dir /kaggle/working/outputs/ocr_overfit_v1/unicode_diagnostics
```

This prints and saves:

- raw strings;
- `repr()` strings;
- Unicode codepoints;
- UTF-8 file validity;
- NFC/NFD normalization checks;
- replacement-character counts;
- control-character counts.

Use this before interpreting PNG output.

## Rendering Self-Test

Render known Cyrillic strings:

```bash
python -m scripts.render_unicode_self_test \
  --output /kaggle/working/outputs/unicode_render_self_test.png \
  --report /kaggle/working/outputs/unicode_render_self_test.json
```

The renderer now searches for common Unicode fonts, including:

- DejaVu Sans on Linux/Kaggle;
- Liberation Sans;
- Noto Sans;
- Arial/LucidaGrande on macOS.

If the self-test PNG displays Cyrillic correctly but OCR grids do not, inspect the OCR text diagnostics for actual replacement characters or control characters.

## OCR Grid Rendering

Render hardest samples:

```bash
python -m scripts.render_ocr_samples \
  --experiment-dir /kaggle/working/outputs/ocr_overfit_v1 \
  --manifest /kaggle/input/rukopys-ocr-overfit-v1/rukopys_ocr_overfit_v1/manifests/val.jsonl \
  --mode hardest \
  --limit 8 \
  --columns 2 \
  --output /kaggle/working/outputs/ocr_overfit_v1/analysis/hardest_grid.png
```

Optional explicit font:

```bash
python -m scripts.render_ocr_samples \
  --experiment-dir /kaggle/working/outputs/ocr_overfit_v1 \
  --manifest /kaggle/input/rukopys-ocr-overfit-v1/rukopys_ocr_overfit_v1/manifests/val.jsonl \
  --mode hardest \
  --limit 8 \
  --columns 2 \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  --output /kaggle/working/outputs/ocr_overfit_v1/analysis/hardest_grid.png
```

## Interpretation Rules

If JSONL diagnostics show:

- `utf8_valid=true`;
- reference replacement count is zero;
- predicted replacement count is zero;
- Cyrillic codepoints are present;

then square glyphs in PNG output are a rendering/font issue.

If JSONL diagnostics show `U+FFFD` in predictions, corruption exists before visualization.

If references contain `U+FFFD`, inspect the manifest/export path before changing OCR training.

If tokenizer roundtrip fails on clean Cyrillic probes, inspect tokenizer/model compatibility before interpreting model quality.
