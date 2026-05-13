from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.data.schema import DetectionResult
from src.preprocessing.image_ops import ensure_rgb


def detect_text_regions(
    image: Image.Image,
    min_area: int = 500,
    max_area_ratio: float = 0.35,
    horizontal_kernel_ratio: float = 0.045,
    vertical_kernel_ratio: float = 0.012,
) -> list[DetectionResult]:
    """Detect coarse text regions in a single external image.

    This is a lightweight demo detector for visual inspection. It finds dark
    connected ink-like regions and merges nearby strokes into line/phrase boxes.
    """
    rgb = np.array(ensure_rgb(image))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        11,
    )

    height, width = binary.shape
    horizontal_kernel = max(15, int(width * horizontal_kernel_ratio))
    vertical_kernel = max(5, int(height * vertical_kernel_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel, vertical_kernel))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    merged = cv2.dilate(merged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = width * height * max_area_ratio
    detections: list[DetectionResult] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        touches_image_edge = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
        is_thin_edge_artifact = touches_image_edge and (h < height * 0.035 or w < width * 0.035)
        if area < min_area or area > max_area:
            continue
        if is_thin_edge_artifact:
            continue
        if w < 12 or h < 8:
            continue
        detections.append(
            DetectionResult(
                bbox=(x, y, x + w, y + h),
                type="text",
                score=None,
                metadata={"detector": "simple_cv"},
            )
        )

    return sorted(detections, key=lambda item: (item.bbox[1], item.bbox[0], item.bbox[2], item.bbox[3]))
