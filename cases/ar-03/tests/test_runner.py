import base64
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

CASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ar03_solve", CASE / "ways/baseline/solve.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def task():
    image = np.full((32, 40, 3), 255, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    return {"input": json.dumps({
        "version": "ar03-page-v1", "image_base64": base64.b64encode(jpeg).decode(),
    })}


def test_image_and_raw_detections_survive_transport():
    def solve(image, *, detections):
        assert image.shape == (32, 40, 3)
        detections.append({"bbox": [1, 2, 30, 31]})
        return {"tables": []}

    result = json.loads(runner.solve_request(task(), solver=solve)["answer"])
    assert result == {"prediction": {"tables": []},
                      "detections": [{"bbox": [1, 2, 30, 31]}], "error": None}


def test_original_recognition_error_policy():
    def recognition_failure(image, *, detections):
        detections.append({"bbox": [1, 2, 30, 31]})
        raise RuntimeError("synthetic recognition failure")

    result = json.loads(runner.solve_request(task(), solver=recognition_failure)["answer"])
    assert result["prediction"] == {"tables": []}
    assert result["detections"]
    assert result["error"] == "synthetic recognition failure"

    def infrastructure_failure(image, *, detections):
        raise FileNotFoundError("synthetic missing weights")

    with pytest.raises(FileNotFoundError):
        runner.solve_request(task(), solver=infrastructure_failure)


@pytest.mark.parametrize("value", ["x" * 500001, '{"version":"wrong"}', "{}"])
def test_bad_input_is_not_a_successful_empty_prediction(value):
    with pytest.raises(ValueError):
        runner.decode_input(value)


def test_common_namespace_is_private_to_solver():
    source = CASE / "ways/baseline/runtime/ar03_solver"
    for path in source.rglob("*.py"):
        text = path.read_text()
        assert "from common." not in text
        assert "from baseline." not in text
