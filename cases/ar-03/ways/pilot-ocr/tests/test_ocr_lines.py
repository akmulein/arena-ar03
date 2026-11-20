"""Synthetic-only development tests: no weights, real images, or model calls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

WAY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WAY / "runtime"))
SPEC = importlib.util.spec_from_file_location(
    "pilot_ocr", WAY / "runtime/ar03_solver/baseline/ocr.py"
)
assert SPEC is not None and SPEC.loader is not None
ocr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ocr)


def canvas(height=40, width=100):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def text_row(image, y, height=8):
    # Disconnected glyph-like components, rather than a solid horizontal rule.
    for x in range(8, 72, 8):
        image[y:y + height, x:x + 4] = 0


def test_single_line_is_unchanged():
    image = canvas()
    text_row(image, 15)
    result = ocr.line_images(image)
    assert len(result) == 1
    assert result[0] is image


def test_two_lines_are_split_in_reading_order_without_rescaling():
    image = canvas()
    text_row(image, 4)
    text_row(image, 24)
    result = ocr.line_images(image)
    assert len(result) == 2
    assert np.array_equal(result[0], image[:18])
    assert np.array_equal(result[1], image[18:])
    assert all(part.flags.c_contiguous for part in result)


def test_frame_rules_do_not_prevent_multiline_detection():
    image = canvas()
    text_row(image, 5)
    text_row(image, 25)
    image[0:2] = 0
    image[-2:] = 0
    image[:, 0:2] = 0
    image[:, -2:] = 0
    assert len(ocr.line_images(image)) == 2


def test_disconnected_dots_stay_with_their_text_line():
    image = canvas()
    text_row(image, 12)
    image[8:10, 9:11] = 0
    image[8:10, 25:27] = 0
    result = ocr.line_images(image)
    assert len(result) == 1
    assert result[0] is image


def test_tiny_isolated_mark_does_not_become_a_separate_line():
    image = canvas()
    text_row(image, 20)
    image[3:5, 9:11] = 0
    result = ocr.line_images(image)
    assert len(result) == 1
    assert result[0] is image


@pytest.mark.parametrize("value", [0, 128, 255])
def test_flat_crop_uses_the_original_path(value):
    image = np.full((30, 80, 3), value, dtype=np.uint8)
    assert ocr.line_images(image)[0] is image


def test_prediction_count_and_text_binding_follow_lines(monkeypatch):
    image = canvas(height=80)
    text_row(image, 4)
    text_row(image, 24)
    text_row(image, 55)
    table = {"cells": [
        {"bbox": [0, 0, 100, 40], "text": "old"},
        {"bbox": [0, 40, 100, 80], "text": "old"},
        {"bbox": [0, 0, 0, 10], "text": "old"},
    ]}
    calls = []

    class FakeModel:
        def predict(self, crops, *, batch_size):
            calls.append((len(crops), batch_size))
            return iter([{"rec_text": text} for text in ("first  line", " second ", "last")])

    monkeypatch.setattr(ocr, "ocr_model", lambda: FakeModel())
    ocr.recognize(image, table)
    assert calls == [(3, 8)]
    assert [cell["text"] for cell in table["cells"]] == ["first line second", "last", ""]


def test_wrong_line_result_count_fails_loudly(monkeypatch):
    image = canvas()
    text_row(image, 4)
    text_row(image, 24)

    class FakeModel:
        def predict(self, crops, *, batch_size):
            return iter([{"rec_text": "one"}])

    monkeypatch.setattr(ocr, "ocr_model", lambda: FakeModel())
    with pytest.raises(ValueError, match="1 instead of 2"):
        ocr.recognize(image, {"cells": [{"bbox": [0, 0, 100, 40]}]})


def test_empty_table_never_initializes_model(monkeypatch):
    def forbidden_model():
        raise AssertionError("No OCR model should be initialized")

    monkeypatch.setattr(ocr, "ocr_model", forbidden_model)
    ocr.recognize(canvas(), {"cells": []})
