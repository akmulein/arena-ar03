import tempfile

import cv2
import numpy as np
import pytesseract
from PIL import Image

from solution.postprocess import normalize


def recognize(gray, table):
    pages = []
    for cell in table["cells"]:
        x0, y0, x1, y1 = map(int, cell["bbox"])
        crop = gray[y0 + 2 : y1 - 1, x0 + 2 : x1 - 1]
        if crop.size == 0:
            crop = np.full((10, 10), 255, np.uint8)
        crop = cv2.copyMakeBorder(
            crop, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255
        )
        pages.append(Image.fromarray(crop))
    # Each TIFF page is one cell, including empty cells.
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
