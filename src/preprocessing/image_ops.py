from __future__ import annotations

from PIL import Image

from src.data.schema import BBox


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    clamped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise ValueError(f"Clamped bbox has non-positive area: original={bbox!r}, clamped={clamped!r}")
    return clamped


def crop_region(image: Image.Image, bbox: BBox, padding: int = 0) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    padded = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
    return image.crop(clamp_bbox(padded, width=width, height=height))
