from __future__ import annotations

from typing import Protocol

from PIL import Image

from src.data.schema import OCRResult, RegionPrediction


class OCRRecognizer(Protocol):
    def recognize(self, image: Image.Image, region: RegionPrediction) -> OCRResult:
        """Return OCR text for a cropped or full image region."""
