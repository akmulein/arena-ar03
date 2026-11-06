import tempfile

import cv2
import numpy as np
import pytesseract
from PIL import Image

from solution.postprocess import normalize


def cell_image(gray, box):
    x0, y0, x1, y1 = box
    crop = gray[
        max(0, int(np.floor(y0))) : min(gray.shape[0], int(np.ceil(y1))),
        max(0, int(np.floor(x0))) : min(gray.shape[1], int(np.ceil(x1))),
    ].copy()
    if crop.size == 0:
        crop = np.full((10, 10), 255, np.uint8)
    dark = crop < 160
    for row in [0, 1, crop.shape[0] - 2, crop.shape[0] - 1]:
        if 0 <= row < crop.shape[0] and dark[row].mean() > 0.7:
            crop[row] = 255
    for col in [0, 1, crop.shape[1] - 2, crop.shape[1] - 1]:
        if 0 <= col < crop.shape[1] and dark[:, col].mean() > 0.85:
            crop[:, col] = 255
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(
        crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255
    )
    return Image.fromarray(crop)


def recognize(gray, table):
    pages = [cell_image(gray, cell["bbox"]) for cell in table["cells"]]
    with tempfile.NamedTemporaryFile(suffix=".tiff") as temporary:
        pages[0].save(temporary.name, save_all=True, append_images=pages[1:])
        text = pytesseract.image_to_string(
            temporary.name,
            config="--psm 6",
            lang="eng",
            timeout=max(10, 2 * len(pages)),
        )
    texts = text.split("\x0c")
    if len(texts) != len(pages):
        raise ValueError("Tesseract returned an unexpected page count")
    for cell, text in zip(table["cells"], texts):
        cell["text"] = normalize(text)
