# Kaggle Environment Strategy

## Policy

Use the native Kaggle CUDA/PyTorch stack.

Do not install or pin:

- `torch`;
- `torchvision`;
- `torchaudio`;
- CUDA wheels or NVIDIA runtime packages;
- `numpy`;
- `pandas`;
- `protobuf`.

Reason:

- Kaggle ships those packages as a tested runtime set;
- replacing torch can pull incompatible CUDA wheels;
- replacing numpy/pandas/protobuf can destabilize notebooks and transitive libraries;
- reproducibility should come from environment capture plus constrained OCR dependencies, not from replacing the base image.

## Runtime Files

Use:

```text
requirements-kaggle-ocr.txt
```

For local development only:

```text
requirements-dev.txt
requirements.txt
```

The previous heavy lock file was removed:

```text
requirements-kaggle-ocr-smoke.lock.txt
```

It installed `torch==2.11.0`, CUDA 13 wheels, `protobuf==7`, `pandas==3`, and `numpy==2.4`, which is not an acceptable long-term Kaggle strategy.

## Recommended Kaggle Install Cell

```bash
pip install --no-deps -r requirements-kaggle-ocr.txt
```

Then verify:

```bash
python -m scripts.verify_kaggle_ocr_environment --require-cuda
```

If `--no-deps` fails because a package dependency is missing, install only the missing dependency after inspecting the verification output. Do not run broad upgrade commands.

## Remaining Dependencies

`transformers>=4.45,<4.56`

- needed for `TrOCRProcessor`, `VisionEncoderDecoderModel`, checkpoint save/load, and generation;
- version 5.8.0 is not required;
- `<4.56` avoids the newer 4.x/5.x generation and dependency line until explicitly validated on Kaggle.

`tokenizers>=0.20,<0.22`

- required by supported Transformers 4.x releases;
- installed explicitly because Kaggle base images may carry an older tokenizer wheel.

`huggingface-hub>=0.24,<1`

- required for model download and cache resolution through `from_pretrained`;
- `<1` keeps the API line conservative for Transformers 4.x.

`safetensors>=0.4.3`

- needed for safe checkpoint serialization/deserialization used by modern Transformers models;
- not a CUDA package.

`sentencepiece>=0.1.99`

- included for tokenizer compatibility across HuggingFace OCR models;
- not always required by TrOCR itself, but low risk and common for OCR model variants.

`datasets>=2.18,<5`

- needed for project dataset-loading utilities and RUKOPYS exploration workflows;
- not needed for manifest-only stage-1 training, but required by the project pipeline.

## Compatibility Notes

Minimum required behavior:

- TrOCR model and processor load with `from_pretrained`;
- greedy generation works through `model.generate`;
- training loop supports CUDA forward/backward through native Kaggle torch;
- checkpoint save/reload works with `safetensors`;
- `datasets.load_dataset` remains available for RUKOPYS utilities.

The project does not currently require Transformers 5.x behavior. Current code avoids known generation warnings with the 4.x API by storing generation parameters in `generation_config`.

Current compatibility assumption:

- Kaggle provides a CUDA 12-class PyTorch runtime in the notebook image;
- this repository should use that native torch installation;
- compatibility is verified by `scripts.verify_kaggle_ocr_environment`, not by force-installing a replacement torch wheel.

This change has not run model training. It only stabilizes environment setup before the next OCR experiment.

## Migration Notes

For existing Kaggle notebooks:

1. Remove any cell that installs `requirements-kaggle-ocr-smoke.lock.txt`.
2. Restart the Kaggle session if torch/CUDA packages were already replaced.
3. Clone the latest repository state.
4. Install:

```bash
pip install --no-deps -r requirements-kaggle-ocr.txt
```

5. Verify:

```bash
python -m scripts.verify_kaggle_ocr_environment --require-cuda
```

If the verification output shows `transformers==5.x`, the notebook is not using the intended stage-1 environment.

## Verification Output

The verification script writes:

```text
outputs/kaggle_ocr_environment.json
```

Required checks:

- `torch_runtime.cuda_available == true` when `--require-cuda` is used;
- CUDA device count is nonzero;
- `transformers` satisfies `>=4.45,<4.56`;
- `tokenizers` satisfies `>=0.20,<0.22`;
- `huggingface-hub` satisfies `>=0.24,<1`;
- `safetensors` satisfies `>=0.4.3`;
- `sentencepiece` satisfies `>=0.1.99`;
- `datasets` satisfies `>=2.18,<5`.

Also inspect:

- `torch`;
- `torchvision`;
- `torchaudio`;
- `numpy`;
- `pandas`;
- `protobuf`;
- CUDA version;
- GPU names.

## Residual Risks

- Kaggle base images can change across sessions.
- Native Kaggle package versions are not fully controlled by this repository.
- Installing even lightweight packages can still update transitive dependencies if `--no-deps` is not used.
- Exact reproducibility requires saving `outputs/kaggle_ocr_environment.json` with each run.

Mitigation:

- use `--no-deps` for OCR runtime packages;
- run `python -m scripts.verify_kaggle_ocr_environment --require-cuda` before training;
- keep `metadata.json` and `metrics.json` from each experiment;
- do not compare runs unless environment metadata is available.
