import copy
import hashlib

import cv2
import numpy as np

from solution.geometry import transform_box


def distort(image, truth, page_id, seed=42):
    key = hashlib.sha256(f"{seed}:{page_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "big"))
    height, width = image.shape[:2]
    angle = float(rng.uniform(-2, 2))
    scale = float(rng.uniform(0.65, 1))
    new_w, new_h = round(width * scale), round(height * scale)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    matrix[0] *= new_w / width
    matrix[1] *= new_h / height
    image = cv2.warpAffine(
        image, matrix, (new_w, new_h), borderValue=(255, 255, 255)
    )
    sigma = float(rng.uniform(0, 0.8))
    image = cv2.GaussianBlur(image, (3, 3), max(sigma, 0.01))
    noise = float(rng.uniform(0, 7))
    image = np.clip(
        image.astype(float) + rng.normal(0, noise, image.shape), 0, 255
    ).astype("uint8")
    quality = int(rng.integers(55, 96))
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    truth = copy.deepcopy(truth)
    for table in truth["tables"]:
        table["bbox"] = transform_box(table["bbox"], matrix, new_w, new_h)
        for cell in table["cells"]:
            cell["bbox"] = transform_box(cell["bbox"], matrix, new_w, new_h)
    return (
        encoded,
        truth,
        dict(
            angle=angle,
            scale=scale,
            blur=sigma,
            noise=noise,
            jpeg=quality,
            matrix=matrix.tolist(),
        ),
    )
