import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from ar03_solver.baseline.ocr import cell_image, trim_foreground


def test_blank_is_unchanged():
    image = np.full((30, 60, 3), 255, dtype=np.uint8)
    assert np.array_equal(trim_foreground(image), image)


def test_trim_preserves_glyph_and_punctuation():
    image = np.full((40, 80, 3), 255, dtype=np.uint8)
    image[10:20, 20:25] = 0
    image[22, 40] = 0
    result = trim_foreground(image)
    assert result.shape == (19, 27, 3)
    assert (result.min(axis=2) == 0).sum() == 51
    assert result.flags.c_contiguous


def test_rule_removed_short_minus_preserved():
    image = np.full((40, 80, 3), 255, dtype=np.uint8)
    image[0, :] = 0
    image[12:22, 20:25] = 0
    image[17:19, 32:40] = 0
    result = trim_foreground(image)
    assert (result.min(axis=2) == 0).sum() == 66
    assert result.shape[0] < 40


def test_outside_box_is_empty():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    assert cell_image(image, [21, 21, 25, 25]) is None
