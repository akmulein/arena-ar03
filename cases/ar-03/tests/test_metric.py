"""Synthetic acceptance for the independent AR-03 command metric."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

CASE = Path(__file__).resolve().parents[1]
METRIC = CASE / "metric"


def page(text="12", *, bbox=None, cells=None, cols=1):
    return {"tables": [{
        "bbox": bbox or [10, 10, 110, 110], "rows": 1, "cols": cols,
        "cells": cells or [dict(r0=0, r1=0, c0=0, c1=cols - 1, text=text)],
    }]}


def invoke(script, payload, *, env=None, check=True):
    completed = subprocess.run(
        [sys.executable, "-B", str(METRIC / script)],
        input=json.dumps(payload), capture_output=True, text=True,
        check=check, cwd="/", env=env,
    )
    return json.loads(completed.stdout) if check else completed


def evaluate(prediction, truth, *, detections=None, error=None, example_id="synthetic"):
    if detections is None:
        detections = prediction.get("tables", []) if isinstance(prediction, dict) else []
    return invoke("score.py", {
        "example": {
            "id": example_id, "input": "synthetic-image",
            "reference": json.dumps(truth),
        },
        "answer": json.dumps({
            "prediction": prediction, "detections": detections, "error": error,
        }),
    })


def report(result, example_id="synthetic"):
    return {"example_id": example_id, **result}


def test_identity_and_text_distance():
    same = evaluate(page(), page())
    quality = same["statistics"]["quality"]
    assert same["score"] == 1
    assert quality == {
        "teds": 1.0, "teds_struct": 1.0, "cell_f1": 1.0,
        "numeric_accuracy": 1.0, "valid": True,
    }
    wrong = evaluate(page("13"), page())
    assert wrong["score"] == pytest.approx(1 - 0.5 / 3)
    assert wrong["statistics"]["quality"]["teds_struct"] == 1
    assert wrong["statistics"]["quality"]["cell_f1"] == 0
    assert wrong["statistics"]["quality"]["numeric_accuracy"] == 0


def test_missing_and_extra_tables():
    doubled = deepcopy(page())
    doubled["tables"].append(deepcopy(doubled["tables"][0]))
    doubled["tables"][1]["bbox"] = [200, 10, 300, 110]
    for prediction, truth in ((page(), doubled), (doubled, page())):
        result = evaluate(prediction, truth)
        assert result["score"] == 0.5
        assert result["statistics"]["quality"]["cell_f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(("prediction", "expected", "valid"), [
    ({"tables": []}, 1, True), (None, 0, False),
])
def test_empty_and_invalid_predictions(prediction, expected, valid):
    result = evaluate(prediction, {"tables": []})
    assert result["score"] == expected
    assert result["statistics"]["quality"]["valid"] is valid


@pytest.mark.parametrize("change", ["overlap", "hole", "span", "negative", "text", "bool"])
def test_invalid_prediction_keeps_original_zero_policy(change):
    prediction = page()
    table = prediction["tables"][0]
    if change == "overlap":
        table["cells"] *= 2
    elif change == "hole":
        table["cells"] = []
    elif change == "span":
        table["cells"][0]["r1"] = 2
    elif change == "negative":
        table["cells"][0]["c0"] = -1
    elif change == "text":
        table["cells"][0]["text"] = 42
    else:
        table["rows"] = True
    result = evaluate(prediction, page(), detections=[])
    assert result["score"] == 0
    assert result["statistics"]["quality"]["valid"] is False


def test_spans_whitespace_escaping_and_location():
    spanning = page("<&", cols=2)
    assert evaluate(spanning, spanning)["score"] == 1
    assert evaluate(page("<&"), spanning)["statistics"]["quality"]["teds_struct"] < 1
    assert evaluate(page(" 12\n"), page("12"))["score"] == 1
    assert evaluate(page(bbox=[200, 200, 300, 300]), page())["score"] == 0


def test_numeric_rule_and_duplicate_cells_preserve_counters():
    cells = [dict(r0=0, r1=0, c0=i, c1=i, text=text)
             for i, text in enumerate(("−1,23 %", ".25", "12a", "12"))]
    truth = page(cells=cells, cols=4)
    result = evaluate(truth, truth)
    assert result["statistics"]["counters"]["numeric_total"] == 3
    assert result["statistics"]["counters"]["numeric_correct"] == 3
    assert result["statistics"]["counters"]["cell_correct"] == 4


def test_matching_threshold_is_inclusive_and_detections_are_independent():
    truth = page(bbox=[0, 0, 10, 10])
    for right, expected_tp in ((5, 1), (4.9, 0)):
        result = evaluate({"tables": []}, truth, detections=[{"bbox": [0, 0, right, 10]}])
        assert result["score"] == 0
        assert result["statistics"]["counters"]["detection_tp"] == expected_tp
    result = evaluate(truth, truth, detections=truth["tables"] * 2)
    assert result["statistics"]["counters"]["detection_tp"] == 1
    assert result["statistics"]["counters"]["detection_fp"] == 1


def test_matching_maximizes_pair_count_before_iou():
    truth = page(bbox=[1, 0, 11, 10])
    truth["tables"].append(page(bbox=[0, 0, 6, 10])["tables"][0])
    predictions = [{"bbox": [0, 0, 10, 10]}, {"bbox": [4, 0, 14, 10]}]
    result = evaluate({"tables": []}, truth, detections=predictions)
    assert result["statistics"]["counters"]["detection_tp"] == 2


def test_macro_teds_and_micro_cell_numeric_detection_aggregation():
    first = evaluate(page(), page(), example_id="first")
    truth = page()
    truth["tables"] = [page(bbox=[i * 200 + 10, 10, i * 200 + 110, 110])["tables"][0]
                       for i in range(3)]
    second = evaluate({"tables": []}, truth, error="synthetic recognition failure",
                      example_id="second")
    summary = invoke("aggregate.py", {
        "results": [report(first, "first"), report(second, "second")],
    })
    assert summary["score"] == 0.5
    stats = summary["statistics"]
    assert stats["teds"] == 0.5
    assert stats["cell_f1"] == pytest.approx(0.4)
    assert stats["numeric_accuracy"] == 0.25
    assert stats["detection_precision"] == 1
    assert stats["detection_recall"] == 0.25
    assert stats["detection_f1"] == pytest.approx(0.4)
    assert stats["errors"] == 1
    assert stats["pages"] == 2
    assert "seconds_total" not in stats


def test_empty_corpus_diagnostics_preserve_original_aggregation():
    result = evaluate({"tables": []}, {"tables": []})
    summary = invoke("aggregate.py", {"results": [report(result)]})
    assert summary["score"] == 1
    assert summary["statistics"]["cell_f1"] == 0
    assert summary["statistics"]["numeric_accuracy"] == 0
    assert summary["statistics"]["detection_f1"] == 1


def test_aggregation_uses_page_mean_not_table_weighting():
    first = evaluate(page(), page(), example_id="first")
    truth = page()
    truth["tables"] *= 3
    second = evaluate({"tables": []}, truth, example_id="second")
    summary = invoke("aggregate.py", {
        "results": [report(first, "first"), report(second, "second")],
    })
    assert summary["score"] == 0.5


@pytest.mark.parametrize("results", [[], [{"example_id": "a"}], [{"example_id": "a", "score": 0}]])
def test_incomplete_aggregation_fails(results):
    result = invoke("aggregate.py", {"results": results}, check=False)
    assert result.returncode != 0
    assert result.stdout == ""


def test_duplicate_report_ids_fail():
    result = report(evaluate(page(), page()))
    completed = invoke("aggregate.py", {"results": [result, result]}, check=False)
    assert completed.returncode != 0


def test_malformed_answer_and_invalid_truth_fail_without_stdout():
    payload = {"example": {"id": "synthetic", "input": "safe", "reference": "null"},
               "answer": "not-json"}
    completed = invoke("score.py", payload, check=False)
    assert completed.returncode != 0
    assert completed.stdout == ""
    payload["answer"] = json.dumps({"prediction": page(), "detections": [], "error": None})
    completed = invoke("score.py", payload, check=False)
    assert completed.returncode != 0
    assert completed.stdout == ""


def test_metric_does_not_import_common_evaluation_or_solver(tmp_path):
    poison = tmp_path / "poison"
    poison.mkdir()
    for module in ("common", "evaluation", "baseline", "ar03_solver"):
        directory = poison / module
        directory.mkdir()
        (directory / "__init__.py").write_text("raise RuntimeError('unexpected import')\n")
    environment = {**os.environ, "PYTHONPATH": str(poison), "PYTHONDONTWRITEBYTECODE": "1"}
    payload = {"example": {"id": "synthetic", "input": "safe", "reference": json.dumps(page())},
               "answer": json.dumps({"prediction": page(), "detections": [], "error": None})}
    result = invoke("score.py", payload, env=environment)
    assert result["score"] == 1
    assert invoke("aggregate.py", {"results": [report(result)]}, env=environment)["score"] == 1


def test_metric_manifest_uses_report_aggregation():
    configuration = json.loads((METRIC / "metric.json").read_text())
    assert configuration["kind"] == "command"
    assert configuration["direction"] == "maximize"
    assert configuration["command"]["uses_model"] is False
    assert configuration["aggregation"]["uses_model"] is False
    assert configuration["aggregation"]["aggregation_input"] == "report"
