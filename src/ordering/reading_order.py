from __future__ import annotations

from collections.abc import Sequence

from src.data.schema import RegionPrediction


def sort_regions_reading_order(
    regions: Sequence[RegionPrediction],
    line_y_tolerance: float = 0.5,
) -> tuple[RegionPrediction, ...]:
    """Sort regions top-to-bottom, then left-to-right within approximate text lines.

    The grouping threshold is based on each region height. This is intentionally simple
    and deterministic; better ordering heuristics should be introduced only after
    local PageCER failure analysis.
    """
    remaining = sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0], region.bbox[2], region.bbox[3]))
    ordered: list[RegionPrediction] = []

    while remaining:
        anchor = remaining.pop(0)
        ax1, ay1, ax2, ay2 = anchor.bbox
        anchor_center_y = (ay1 + ay2) / 2
        anchor_height = max(1, ay2 - ay1)

        line = [anchor]
        next_remaining: list[RegionPrediction] = []
        for region in remaining:
            x1, y1, x2, y2 = region.bbox
            center_y = (y1 + y2) / 2
            height = max(1, y2 - y1)
            tolerance = line_y_tolerance * max(anchor_height, height)
            if abs(center_y - anchor_center_y) <= tolerance:
                line.append(region)
            else:
                next_remaining.append(region)

        ordered.extend(sorted(line, key=lambda region: (region.bbox[0], region.bbox[1], region.bbox[2], region.bbox[3])))
        remaining = next_remaining

    return tuple(ordered)
