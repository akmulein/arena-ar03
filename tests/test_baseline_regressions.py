"""Regressions for ruled tables, photo rejection and tiny OCR crops."""

import cv2
import numpy as np

from solution.detection import detect
from solution.ocr import recognize
from solution.preprocessing import preprocess
from solution.structure import recover


def test_horizontal_rules_with_text():
    image = np.full((350, 500, 3), 255, np.uint8)
    for y in [40, 80, 160, 240]:
        cv2.line(image, (40, y), (450, y), (0, 0, 0), 1)
    for y in [67, 110, 150, 190, 225]:
        for x, value in [(55, "Alpha"), (210, "123"), (340, "456")]:
            cv2.putText(
                image,
                value,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
    _, binary, _ = preprocess(image)
    boxes, horizontal, vertical = detect(binary)
    assert len(boxes) == 1
    table = recover(boxes[0], horizontal, vertical, binary)
    assert table is not None
    assert table["rows"] >= 4 and table["cols"] >= 3


def test_dense_picture_inside_frame_is_rejected():
    rng = np.random.default_rng(42)
    binary = np.zeros((350, 500), np.uint8)
    binary[60:280, 50:450] = rng.integers(0, 2, (220, 400), np.uint8) * 255
    cv2.rectangle(binary, (48, 58), (452, 282), 255, 2)
    assert detect(binary)[0] == []


def test_small_numeric_ocr():
    image = np.full((35, 90), 255, np.uint8)
    cv2.putText(
        image,
        "12345",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        0,
        1,
        cv2.LINE_AA,
    )
    table = {"cells": [{"bbox": [4, 10, 48, 22], "text": ""}]}
    recognize(image, table)
    assert table["cells"][0]["text"] == "12345"
