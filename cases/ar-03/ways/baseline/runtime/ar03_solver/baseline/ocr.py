import os
from functools import lru_cache
from pathlib import Path

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


def recognize(image, table):
    """Recognize cell text using PP-OCRv6_small_rec."""

    crops = []
    cells = []

    for cell in table["cells"]:
        cell["text"] = ""

        crop = cell_image(image, cell["bbox"])

        if crop is None:
            continue

        crops.append(crop)
        cells.append(cell)

    if not crops:
        return

    results = list(
        ocr_model().predict(
            crops,
            batch_size=8,
        )
    )

    if len(results) != len(cells):
        raise ValueError(
            "OCR returned an unexpected result count: "
            f"{len(results)} instead of {len(cells)}"
        )

    for cell, result in zip(cells, results):
        cell["text"] = normalize(result["rec_text"])
