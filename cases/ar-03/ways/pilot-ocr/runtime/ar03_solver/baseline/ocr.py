import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ar03_solver.common.postprocess import normalize


@lru_cache(maxsize=1)
def ocr_model():
    default_dir = (
        Path(__file__).resolve().parents[1]
        / "models/PP-OCRv6_small_rec"
    )

    model_dir = Path(
        os.environ.get(
            "AR03_OCR_MODEL_DIR",
            str(default_dir),
        )
    ).expanduser()

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"OCR weights not found: {model_dir}"
        )

    from paddlex import create_model

    return create_model(
        model_name="PP-OCRv6_small_rec",
        model_dir=str(model_dir),
        device="cpu",
    )


def cell_image(image, box):
    """Extract a cell in deskewed coordinates."""

    x0, y0, x1, y1 = box
    height, width = image.shape[:2]

    x0 = max(0, int(np.floor(x0)))
    y0 = max(0, int(np.floor(y0)))
    x1 = min(width, int(np.ceil(x1)))
    y1 = min(height, int(np.ceil(y1)))

    if x0 >= x1 or y0 >= y1:
        return None

    crop = image[y0:y1, x0:x1].copy()

    return np.ascontiguousarray(crop)


def line_images(crop):
    """Conservatively split clear text rows; leave ambiguous crops untouched.

    The binary image is used only for segmentation. OCR receives slices of the
    original image, with the split in the middle of each interline whitespace.
    """

    height, width = crop.shape[:2]
    if min(height, width) < 4:
        return [crop]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if int(gray.max()) - int(gray.min()) < 16:
        return [crop]

    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size * 0.6:
        return [crop]

    # Long cell borders otherwise bridge all text rows in the projection.
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((1, max(12, int(width * 0.85))), dtype=np.uint8)
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((max(12, int(height * 0.85)), 1), dtype=np.uint8)
    )
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    components = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 2]
    if not components:
        return [crop]

    char_height = float(np.percentile(stats[components, cv2.CC_STAT_HEIGHT], 75))
    foreground = np.isin(labels, components)
    active = foreground.any(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    bands = []
    merge_gap = max(2, int(round(char_height * 0.3)))
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        if bands and start - bands[-1][1] <= merge_gap:
            bands[-1][1] = int(end)
        else:
            bands.append([int(start), int(end)])

    # Isolated punctuation and graphic fragments are not evidence of a line.
    if len(bands) < 2 or any(
        end - start < max(3, char_height * 0.45)
        or end - start > char_height * 2.5
        or np.count_nonzero(foreground[start:end]) < max(3, char_height)
        for start, end in bands
    ):
        return [crop]

    boundaries = [0] + [
        (upper[1] + lower[0]) // 2 for upper, lower in zip(bands[:-1], bands[1:], strict=True)
    ] + [height]
    return [
        np.ascontiguousarray(crop[start:end])
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def recognize(image, table):
    """Recognize text rows once using the unchanged PP-OCRv6_small_rec."""

    crops = []
    cells = []

    for cell in table["cells"]:
        cell["text"] = ""

        crop = cell_image(image, cell["bbox"])

        if crop is None:
            continue

        for line in line_images(crop):
            crops.append(line)
            cells.append(cell)

    if not crops:
        return

    results = list(
        ocr_model().predict(
            crops,
            batch_size=8,
        )
    )

    if len(results) != len(crops):
        raise ValueError(
            "OCR returned an unexpected result count: "
            f"{len(results)} instead of {len(crops)}"
        )

    for cell, result in zip(cells, results, strict=True):
        cell["text"] = normalize(cell["text"] + " " + result["rec_text"])
