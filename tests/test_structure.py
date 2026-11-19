"""SLANet_plus HTML and cell geometry adapter."""

from unittest.mock import Mock

import numpy as np
import pytest

from baseline import structure
from common.schema import validate


def test_rowspan_and_colspan():
    tokens = [
        '<table><tr>', '<td', ' rowspan="2"', ' colspan="2"', '></td>',
        '<td></td></tr>', '<tr><td></td></tr>',
        '<tr><td></td><td></td><td></td></tr></table>',
    ]
    cells = structure.parse_structure(tokens)
    assert [(c['r0'], c['r1'], c['c0'], c['c1']) for c in cells] == [
        (0, 1, 0, 1), (0, 0, 2, 2), (1, 1, 2, 2),
        (2, 2, 0, 0), (2, 2, 1, 1), (2, 2, 2, 2),
    ]
    validate({'tables': [{'bbox': [0, 0, 30, 30], 'rows': 3,
                         'cols': 3, 'cells': cells}]})


@pytest.mark.parametrize('points', [
    [-1, 1, 9, 1, 9, 11, -1, 11], [-1, 1, 9, 11],
])
def test_normalize_bbox_clips_and_offsets(points):
    assert structure.normalize_bbox(points, (20, 30, 30, 40)) == [20, 31, 29, 40]


@pytest.mark.parametrize('points', [[], [1, 2, 3], [1, 2, 1, 2], [0, 0, np.nan, 4]])
def test_invalid_cell_geometry(points):
    assert structure.normalize_bbox(points, (0, 0, 10, 10)) is None


def test_recover_preserves_empty_cells(monkeypatch):
    model = Mock()
    model.predict.return_value = iter([{
        'structure': '<tr><td></td><td></td></tr>',
        'bbox': [[0, 0, 10, 10], [10, 0, 20, 10]],
    }])
    monkeypatch.setattr(structure, 'structure_model', lambda: model)
    image = np.full((50, 60, 3), 255, dtype=np.uint8)
    table = structure.recover((10, 20, 30, 30), image)
    assert (table['rows'], table['cols']) == (1, 2)
    assert [c['text'] for c in table['cells']] == ['', '']
    assert [c['bbox'] for c in table['cells']] == [[10, 20, 20, 30], [20, 20, 30, 30]]
    validate({'tables': [table]})


@pytest.mark.parametrize('bboxes', [[], [[0, 0, 0, 0]]])
def test_recover_rejects_inconsistent_geometry(monkeypatch, bboxes):
    model = Mock()
    model.predict.return_value = iter([{
        'structure': '<tr><td></td></tr>', 'bbox': bboxes,
    }])
    monkeypatch.setattr(structure, 'structure_model', lambda: model)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    assert structure.recover((0, 0, 10, 10), image) is None
