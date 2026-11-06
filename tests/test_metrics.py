from copy import deepcopy

import pytest

from evaluator.metrics import score
from solution.postprocess import to_html
from solution.schema import validate


def page(text="12"):
    return {
        "tables": [
            {
                "bbox": [10, 10, 110, 110],
                "rows": 1,
                "cols": 1,
                "cells": [dict(r0=0, r1=0, c0=0, c1=0, text=text)],
            }
        ]
    }


def test_identity():
    result = score(page(), page())
    assert result["teds"] == result["teds_struct"] == result["cell_f1"] == 1
    assert result["numeric_accuracy"] == 1


def test_character_edit():
    result = score(page("13"), page())
    assert result["teds"] == pytest.approx(1 - 0.5 / 3)
    assert result["teds_struct"] == 1
    assert result["cell_f1"] == result["numeric_accuracy"] == 0


def test_missing_and_extra():
    truth = page()
    doubled = deepcopy(truth)
    doubled["tables"].append(deepcopy(truth["tables"][0]))
    doubled["tables"][1]["bbox"] = [200, 10, 300, 110]
    for a, b in [(truth, doubled), (doubled, truth)]:
        result = score(a, b)
        assert result["teds"] == 0.5
        assert result["cell_f1"] == pytest.approx(2 / 3)
    assert score({"tables": []}, truth)["teds"] == 0


def test_empty():
    assert score({"tables": []}, {"tables": []})["teds"] == 1
    assert score(None, {"tables": []})["teds"] == 0


@pytest.mark.parametrize(
    "change", ["overlap", "hole", "span", "negative", "nan", "text", "bool"]
)
def test_invalid(change):
    invalid = page()
    table = invalid["tables"][0]
    if change == "overlap":
        table["cells"] *= 2
    elif change == "hole":
        table["cells"] = []
    elif change == "span":
        table["cells"][0]["r1"] = 2
    elif change == "negative":
        table["cells"][0]["c0"] = -1
    elif change == "nan":
        table["bbox"][0] = float("nan")
    elif change == "text":
        table["cells"][0]["text"] = 42
    else:
        table["rows"] = True
    with pytest.raises(ValueError):
        validate(invalid)
    assert not score(invalid, page())["valid"]
    assert score(invalid, page())["teds"] == 0


def test_spans_and_escaping():
    truth = page("<&")
    truth["tables"][0]["cols"] = 2
    truth["tables"][0]["cells"][0]["c1"] = 1
    validate(truth)
    assert 'colspan="2"' in to_html(truth)
    assert "&lt;&amp;" in to_html(truth)
    assert score(truth, truth)["teds"] == 1
    assert score(page("<&"), truth)["teds_struct"] < 1


def test_wrong_location():
    other = page()
    other["tables"][0]["bbox"] = [200, 200, 300, 300]
    assert score(other, page())["teds"] == 0


def test_malformed_json_cli(tmp_path):
    import json
    import subprocess
    import sys

    predicted = tmp_path / "predicted.json"
    truth = tmp_path / "truth.json"
    predicted.write_text("{broken")
    truth.write_text(json.dumps(page()))
    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "case.metric.score",
            str(predicted),
            str(truth),
        ],
        text=True,
    )
    result = json.loads(output)
    assert not result["valid"]
    assert result["teds"] == result["cell_f1"] == 0
