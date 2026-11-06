import cv2
import numpy as np

from scripts.augment import distort
from solution.detection import detect
from solution.geometry import transform_box
from solution.pipeline import solve
from solution.structure import recover


def grid():
    image = np.full((300, 400, 3), 255, np.uint8)
    for y in [40, 140, 240]:
        cv2.line(image, (30, y), (330, y), (0, 0, 0), 2)
    for x in [30, 180, 330]:
        cv2.line(image, (x, 40), (x, 240), (0, 0, 0), 2)
    return image


def test_grid_and_empty_cells():
    result = solve(grid())
    assert len(result["tables"]) == 1
    table = result["tables"][0]
    assert (table["rows"], table["cols"], len(table["cells"])) == (2, 2, 4)
    assert all(not c["text"] for c in table["cells"])


def test_merge():
    image = grid()
    cv2.line(image, (180, 43), (180, 137), (255, 255, 255), 5)
    binary = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, h, v = detect(binary)
    table = recover(boxes[0], h, v)
    assert len(table["cells"]) == 3
    assert table["cells"][0]["c1"] == 1


def test_two_tables():
    image = np.concatenate([grid(), grid()], axis=1)
    assert len(solve(image)["tables"]) == 2


def test_blank():
    assert solve(np.full((300, 400, 3), 255, np.uint8)) == {"tables": []}


def test_geometry():
    matrix = np.array([[2, 0, 3], [0, 2, 5]])
    assert transform_box([1, 2, 5, 6], matrix, 100, 100) == [5, 9, 13, 17]


def test_reproducible_distortion():
    truth = {"tables": []}
    a, _, profile_a = distort(grid(), truth, "doc_1")
    b, _, profile_b = distort(grid(), truth, "doc_1")
    assert np.array_equal(a, b)
    assert profile_a == profile_b


def test_rowspan_and_colspan():
    image = np.full((350, 350), 0, np.uint8)
    for x in [20, 120, 220, 320]:
        cv2.line(image, (x, 20), (x, 320), 255, 2)
    for y in [20, 120, 220, 320]:
        cv2.line(image, (20, y), (320, y), 255, 2)
    cv2.line(image, (120, 23), (120, 217), 0, 5)
    cv2.line(image, (23, 120), (217, 120), 0, 5)
    boxes, h, v = detect(image)
    table = recover(boxes[0], h, v)
    assert len(table["cells"]) == 6
    merged = table["cells"][0]
    assert (merged["r0"], merged["r1"], merged["c0"], merged["c1"]) == (
        0,
        1,
        0,
        1,
    )


def test_deskew():
    from solution.preprocessing import preprocess

    image = grid()
    matrix = cv2.getRotationMatrix2D((200, 150), 2, 1)
    skewed = cv2.warpAffine(
        image, matrix, (400, 300), borderValue=(255, 255, 255)
    )
    _, _, inverse = preprocess(skewed)
    recovered = np.degrees(np.arctan2(inverse[0, 1], inverse[0, 0]))
    assert abs(recovered - 2) < 0.5
    assert len(solve(skewed)["tables"]) == 1
