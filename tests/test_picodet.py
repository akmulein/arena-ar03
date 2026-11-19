"""Regression for PicoDet's capitalized table label."""

from types import SimpleNamespace

import numpy as np
import pytest

from solution import detection


@pytest.mark.parametrize("label", ["Table", "table"])
def test_table_label_is_not_discarded(monkeypatch, label):
    result = {"boxes": [{"label": label, "coordinate": [10, 20, 90, 80]}]}
    model = SimpleNamespace(predict=lambda *args, **kwargs: iter([result]))
    monkeypatch.setattr(detection, "table_detector", lambda: model)

    binary = np.zeros((100, 100), dtype=np.uint8)
    boxes, _, _ = detection.detect(binary, gray=255 - binary)

    assert boxes == [(10, 20, 90, 80)]
