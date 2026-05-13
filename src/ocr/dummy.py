from __future__ import annotations

from PIL import Image

from src.data.schema import OCRResult, RegionPrediction


class EmptyOCRRecognizer:
    """Recognizer used to validate pipeline wiring and submission formatting."""

    def recognize(self, image: Image.Image, region: RegionPrediction) -> OCRResult:
        return OCRResult(text="", score=None, metadata={"recognizer": "empty"})


class OracleTextRecognizer:
    """Recognizer for oracle experiments that copies GT text from region metadata."""

    def recognize(self, image: Image.Image, region: RegionPrediction) -> OCRResult:
        return OCRResult(text=region.text, score=1.0, metadata={"recognizer": "oracle_text"})
