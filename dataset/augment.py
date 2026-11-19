"""Deterministic simulation of scanning distortions."""

import copy
import hashlib

import cv2
import numpy as np

from common.geometry import transform_box
from dataset.config import DEFAULT_SEED


def page_rng(page_id: str, seed: int) -> np.random.Generator:
    key = hashlib.sha256(f"{seed}:{page_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big"))


def transform_truth(truth: dict, matrix: np.ndarray, width: int, height: int) -> dict:
    result = copy.deepcopy(truth)
    for table in result["tables"]:
        table["bbox"] = transform_box(table["bbox"], matrix, width, height)
        for cell in table["cells"]:
            cell["bbox"] = transform_box(cell["bbox"], matrix, width, height)
    return result


def distort(
    image: np.ndarray, truth: dict, page_id: str, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, dict, dict]:
    """Return JPEG bytes, transformed annotations, and distortion parameters."""
    rng = page_rng(page_id, seed)
    height, width = image.shape[:2]

    # Rotation and scaling. Keep the random draw order fixed.
    angle = float(rng.uniform(-2, 2))
    scale = float(rng.uniform(0.65, 1))
    new_width, new_height = round(width * scale), round(height * scale)

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    matrix[0] *= new_width / width
    matrix[1] *= new_height / height
    image = cv2.warpAffine(
        image, matrix, (new_width, new_height), borderValue=(255, 255, 255)
    )

    # Blur, sensor noise, and JPEG compression.
    sigma = float(rng.uniform(0, 0.8))
    image = cv2.GaussianBlur(image, (3, 3), max(sigma, 0.01))
    noise = float(rng.uniform(0, 7))
    image = np.clip(
        image.astype(float) + rng.normal(0, noise, image.shape), 0, 255
    ).astype(np.uint8)

    quality = int(rng.integers(55, 96))
    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise ValueError("JPEG encoding failed")

    transformed = transform_truth(truth, matrix, new_width, new_height)
    profile = {
        "angle": angle,
        "scale": scale,
        "blur": sigma,
        "noise": noise,
        "jpeg": quality,
        "matrix": matrix.tolist(),
    }
    return encoded, transformed, profile
