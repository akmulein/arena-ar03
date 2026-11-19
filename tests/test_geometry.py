"""Coordinate restoration and deskew contracts."""

import cv2
import numpy as np
import pytest

from baseline.preprocessing import preprocess
from common.geometry import transform_box


def test_affine_translation_and_scale():
    matrix = np.array([[2, 0, 3], [0, 2, 5]])
    assert transform_box([1, 2, 5, 6], matrix, 100, 100) == [5, 9, 13, 17]


def test_rotation_uses_all_four_corners():
    matrix = np.array([[1, 1, 0], [-1, 1, 10]])
    assert transform_box([1, 2, 5, 6], matrix, 20, 20) == [3, 7, 11, 15]


def test_coordinates_are_clipped():
    matrix = np.array([[1, 0, -5], [0, 1, 6]])
    assert transform_box([1, 2, 15, 18], matrix, 20, 20) == [0, 8, 10, 20]


def test_deskew():
    image = np.full((300, 400, 3), 255, np.uint8)
    for y in [40, 140, 240]:
        cv2.line(image, (30, y), (330, y), (0, 0, 0), 2)
    for x in [30, 180, 330]:
        cv2.line(image, (x, 40), (x, 240), (0, 0, 0), 2)
    matrix = cv2.getRotationMatrix2D((200, 150), 2, 1)
    skewed = cv2.warpAffine(image, matrix, (400, 300), borderValue=(255, 255, 255))

    deskewed, inverse = preprocess(skewed)

    assert deskewed.shape == image.shape
    assert deskewed.dtype == image.dtype
    assert inverse.shape == (2, 3)
    recovered = np.degrees(np.arctan2(inverse[0, 1], inverse[0, 0]))
    assert recovered == pytest.approx(2, abs=0.5)
