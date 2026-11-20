"""One-to-one table matching by intersection over union."""

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLD = 0.5


def iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - intersection
    )
    return intersection / union if union else 0


def match_tables(predicted: list[dict], truth: list[dict]) -> list[tuple[int, int]]:
    """Maximize eligible match count first, then total IoU."""
    if not predicted or not truth:
        return []

    overlaps = np.array([
        [iou(pred['bbox'], target['bbox']) for target in truth]
        for pred in predicted
    ])
    eligible = overlaps >= IOU_THRESHOLD
    weights = np.where(eligible, overlaps + min(overlaps.shape) + 1, 0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if eligible[r, c]]
