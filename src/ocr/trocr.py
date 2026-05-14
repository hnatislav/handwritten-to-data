from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.data.schema import OCRResult, RegionPrediction


@dataclass(frozen=True)
class TrOCRConfig:
    model_name: str = "microsoft/trocr-small-handwritten"
    device: str = "cpu"
    max_new_tokens: int = 128
    num_beams: int = 1


class TrOCRRecognizer:
    """Zero-shot/frozen TrOCR inference wrapper for line/region crops.

    This is an inference baseline only. It is not expected to be strong on
    Ukrainian handwriting without adaptation; use it to measure failure modes.
    """

    def __init__(self, config: TrOCRConfig | None = None) -> None:
        self.config = config or TrOCRConfig()
        self.processor = TrOCRProcessor.from_pretrained(self.config.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.model_name)
        self.generation_config = deepcopy(self.model.generation_config)
        if self.config.num_beams == 1:
            self.generation_config.early_stopping = False
        self.model.to(self.config.device)
        self.model.eval()

    @torch.inference_mode()
    def recognize(self, image: Image.Image, region: RegionPrediction) -> OCRResult:
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.config.device)
        generated_ids = self.model.generate(
            pixel_values,
            generation_config=self.generation_config,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return OCRResult(
            text=text.strip(),
            score=None,
            metadata={
                "recognizer": "trocr",
                "model_name": self.config.model_name,
                "device": self.config.device,
                "max_new_tokens": self.config.max_new_tokens,
                "num_beams": self.config.num_beams,
            },
        )
