import numpy as np
import pytest

from evaluator.detection import detection_summary, score_detection
from solution import pipeline


def table(box):
    return {"bbox": box}


def test_duplicate_is_a_false_positive():
    gt = table([0, 0, 10, 10])
    result = score_detection([gt, gt], [gt])
    assert result["detection_tp"] == 1
    assert result["detection_fp"] == 1
    assert result["detection_fn"] == 0
    assert result["detection_f1"] == pytest.approx(2 / 3)


def test_iou_threshold_is_inclusive():
    truth = [table([0, 0, 10, 10])]
    assert score_detection([table([0, 0, 5, 10])], truth)["detection_tp"] == 1
    assert (
        score_detection([table([0, 0, 4.9, 10])], truth)["detection_tp"] == 0
    )


@pytest.mark.parametrize(
    ("predicted", "truth", "expected"),
    [
        ([], [], 1),
        ([table([0, 0, 10, 10])], [], 0),
        ([], [table([0, 0, 10, 10])], 0),
    ],
)
def test_empty_cases(predicted, truth, expected):
    result = score_detection(predicted, truth)
    assert result["detection_precision"] == expected
    assert result["detection_recall"] == expected
    assert result["detection_f1"] == expected


def test_micro_aggregation():
    gt = table([0, 0, 10, 10])
    pages = [score_detection([gt], [gt]), score_detection([], [gt] * 3)]
    result = detection_summary(
        *(
            sum(p[f"detection_{key}"] for p in pages)
            for key in ("tp", "fp", "fn")
        )
    )
    assert result["detection_precision"] == 1
    assert result["detection_recall"] == 0.25
    assert result["detection_f1"] == 0.4


def test_detection_survives_rejected_structure(monkeypatch):
    gray = np.full((30, 30), 255, dtype=np.uint8)
    binary = 255 - gray
    inverse = np.array([[1, 0, 5], [0, 1, 7]], dtype=float)
    monkeypatch.setattr(
        pipeline, "preprocess", lambda image: (gray, binary, inverse)
    )
    monkeypatch.setattr(
        pipeline,
        "detect",
        lambda *args, **kwargs: ([(1, 2, 10, 12)], binary, binary),
    )
    monkeypatch.setattr(pipeline, "recover", lambda *args: None)
    detections = []

    result = pipeline.solve(np.dstack([gray] * 3), detections=detections)

    assert result == {"tables": []}
    assert detections == [{"bbox": [6, 9, 15, 19]}]
