"""Synthetic checks only: no model imports or real examples."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "runtime"))

from ar03_solver.baseline import structure
from ar03_solver.common.schema import validate


class FakeModel:
    def __init__(self, html, boxes):
        self.html = html
        self.boxes = boxes

    def predict(self, crop, batch_size):
        assert crop.shape == (100, 100, 3)
        assert batch_size == 1
        yield {"structure": self.html, "bbox": self.boxes}


def recover(monkeypatch, html, boxes):
    monkeypatch.setattr(structure, "structure_model", lambda: FakeModel(html, boxes))
    return structure.recover((10, 20, 110, 120), np.zeros((140, 140, 3), np.uint8))


def test_complete_grid_preserves_geometry(monkeypatch):
    table = recover(monkeypatch, "<tr><td></td><td></td></tr>", [[2, 4, 22, 16], [60, 5, 90, 17]])
    assert table == {
        "bbox": [10, 20, 110, 120],
        "rows": 1,
        "cols": 2,
        "cells": [
            {"r0": 0, "r1": 0, "c0": 0, "c1": 0, "text": "", "bbox": [12, 24, 32, 36]},
            {"r0": 0, "r1": 0, "c0": 1, "c1": 1, "text": "", "bbox": [70, 25, 100, 37]},
        ],
    }


def test_incomplete_grid_adds_missing_position(monkeypatch):
    table = recover(
        monkeypatch,
        "<tr><td></td><td></td></tr><tr><td></td></tr>",
        [[5, 5, 35, 25], [60, 5, 90, 25], [5, 70, 35, 90]],
    )
    assert table is not None
    assert len(table["cells"]) == 4
    missing = table["cells"][-1]
    assert (missing["r0"], missing["c0"], missing["text"]) == (1, 1, "")
    assert missing["bbox"] == [57, 67, 110, 120]
    validate({"tables": [table]})


def test_invalid_box_uses_logical_spanning_geometry(monkeypatch):
    html = '<tr><td colspan="2"></td></tr><tr><td></td><td></td></tr>'
    table = recover(monkeypatch, html, [[0, 0, 0, 0], [5, 70, 35, 90], [60, 70, 90, 90]])
    assert table is not None
    x0, y0, x1, y1 = table["cells"][0]["bbox"]
    assert (x0, y0, x1) == (10, 20, 110)
    assert 20 < y1 < 90
    assert table["cells"][0]["c1"] == 1
    validate({"tables": [table]})


def test_all_invalid_boxes_remain_failure(monkeypatch):
    assert recover(monkeypatch, "<tr><td></td></tr>", [[0, 0, 0, 0]]) is None


def test_mismatched_result_count_remains_failure(monkeypatch):
    assert recover(monkeypatch, "<tr><td></td><td></td></tr>", [[1, 1, 5, 5]]) is None
