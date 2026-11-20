"""Synthetic operator-only checkpoint rendering; no inference or real data."""

import base64
import importlib.util
import json
import os
import sys
import threading
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
from evaluator.checkpoints import write_example, write_metadata

CASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ar03_operator_artifacts", CASE / "operator_artifacts.py"
)
assert SPEC is not None and SPEC.loader is not None
operator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = operator
SPEC.loader.exec_module(operator)


@pytest.fixture
def runtime():
    value = Path(os.environ.get("AR03_TEST_METRIC_RUNTIME", CASE / "metric" / "runtime"))
    assert value.is_dir(), "the frozen AR-03 metric runtime must be present"
    return value


def page(text="12<&"):
    return {
        "tables": [
            {
                "bbox": [5, 5, 95, 75],
                "rows": 2,
                "cols": 2,
                "cells": [
                    {
                        "r0": 0,
                        "r1": 0,
                        "c0": 0,
                        "c1": 1,
                        "text": text,
                        "bbox": [5, 5, 95, 35],
                    },
                    {
                        "r0": 1,
                        "r1": 1,
                        "c0": 0,
                        "c1": 0,
                        "text": "2",
                        "bbox": [5, 35, 50, 75],
                    },
                    {
                        "r0": 1,
                        "r1": 1,
                        "c0": 1,
                        "c1": 1,
                        "text": "",
                        "bbox": [50, 35, 95, 75],
                    },
                ],
            }
        ]
    }


@pytest.fixture
def setup(tmp_path, runtime):
    image = np.full((80, 100, 3), 255, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    example = {
        "id": "p1",
        "input": json.dumps(
            {
                "version": "ar03-page-v1",
                "image_base64": base64.b64encode(encoded.tobytes()).decode(),
            }
        ),
        "reference": json.dumps(page()),
    }
    dataset = tmp_path / "synthetic.jsonl"
    dataset.write_text(json.dumps(example) + "\n")
    state = tmp_path / "state"
    state.mkdir()
    source = operator.Source("train", state, dataset, ("evaluations/*/record.json",))
    config = operator.Configuration(tmp_path / "artifacts", runtime, (source,))
    return config, source, example


def row():
    return {
        "example_id": "p1",
        "answer": json.dumps(
            {
                "prediction": page(),
                "detections": [{"bbox": [5, 5, 95, 75]}],
                "error": None,
            }
        ),
        "score": 1.0,
        "statistics": {"quality": {"teds": 1.0}, "counters": {"cell_correct": 3}},
        "resources": {"solver": {"wall_seconds": 0.25}},
    }


def checkpoint(source, detail, *, final=False):
    path = source.state_dir / "evaluations" / "run1" / "record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"params": {"candidate_id": "S01"}, "status": "RUNNING"}
    if final:
        path.write_text(json.dumps({**metadata, "examples": [detail]}))
    else:
        write_metadata(path, metadata, example_ids=["p1"] if not path.exists() else None)
        write_example(path, 0, detail)
    return path


def test_all_outputs_and_exact_metric_trees(setup):
    config, source, _ = setup
    path = checkpoint(source, row())
    before = path.read_bytes()
    consumer = operator.ArtifactConsumer(config)
    consumer.scan()
    output = config.output_root / "train" / "S01" / "p1"
    expected = {
        "answer.txt",
        "answer.json",
        "prediction.json",
        "truth.json",
        "prediction.html",
        "truth.html",
        "prediction-teds-trees.json",
        "truth-teds-trees.json",
        "prediction-logical-grid.svg",
        "truth-logical-grid.svg",
        "tables-overlay.png",
        "prediction-cells-overlay.png",
        "truth-cells-overlay.png",
        "evaluation.json",
    }
    assert {p.name for p in output.iterdir()} == expected
    assert not list(output.glob("*.jpg"))
    assert path.read_bytes() == before
    metric = consumer.metric
    trees = json.loads((output / "prediction-teds-trees.json").read_text())
    assert trees == metric.trees(page())
    cell_tree = trees[0]["full"]["children"][0]["children"][0]["children"][0]
    assert cell_tree["tag"] == "td"
    assert cell_tree["colspan"] == 2
    assert cell_tree["content"] == list("12<&")
    assert trees[0]["structure_only"]["children"][0]["children"][0]["children"][0]["content"] == []
    assert "&lt;&amp;" in (output / "prediction.html").read_text()
    details = json.loads((output / "evaluation.json").read_text())
    assert details["score"] == 1
    assert details["statistics"] == row()["statistics"]
    assert details["artifact_errors"] == []
    assert details["missing_geometry"] == 0
    assert "image_base64" not in (output / "evaluation.json").read_text()
    overlay = cv2.imread(str(output / "tables-overlay.png"))
    assert overlay.shape == (80, 100, 3)
    assert np.any(overlay != 255)
    consumer.scan()
    assert consumer.processed == 1
    assert consumer.errors == 0
    assert os.stat(output).st_mode & 0o077 == 0
    assert all(os.stat(p).st_mode & 0o077 == 0 for p in output.iterdir())


def test_live_checkpoint_then_metric_failure_preserves_saved_answer(setup):
    config, source, _ = setup
    detail = {"example_id": "p1", "status": "solving"}
    checkpoint(source, detail)
    consumer = operator.ArtifactConsumer(config)
    consumer.scan()
    assert consumer.processed == 0
    detail.update(answer=row()["answer"], status="solved")
    checkpoint(source, detail)
    consumer.scan()
    output = config.output_root / "train" / "S01" / "p1"
    before = (output / "answer.txt").read_bytes()
    saved = json.loads((output / "evaluation.json").read_text())
    assert saved["metrics_available"] is False
    detail.update(failure="judge_failure", error_type="SyntheticMetricError")
    checkpoint(source, detail)
    consumer.scan()
    saved = json.loads((output / "evaluation.json").read_text())
    assert saved["failure"] == "judge_failure"
    assert (output / "answer.txt").read_bytes() == before
    assert consumer.processed == 2


@pytest.mark.parametrize("split", ["public", "private"])
def test_closed_candidate_details_join_only_configured_dataset(setup, split):
    config, source, _ = setup
    source = operator.Source(
        split,
        source.state_dir,
        source.dataset_path,
        tuple(operator.DEFAULT_GLOBS[split]),
    )
    config = operator.Configuration(config.output_root, config.metric_runtime, (source,))
    relative = (
        "dataset/public/candidate-results/digest/details.json"
        if split == "public"
        else "dataset/details/S02.json"
    )
    path = source.state_dir / relative
    path.parent.mkdir(parents=True)
    # Private solver failure records have no input, reference, or answer.
    detail = row() if split == "public" else {"example_id": "p1", "failure": "solver_failure"}
    path.write_text(json.dumps({"candidate_id": "S02", "examples": [detail]}))
    consumer = operator.ArtifactConsumer(config)
    consumer.scan()
    output = config.output_root / split / "S02" / "p1"
    assert (output / "truth.html").exists()
    assert (output / "tables-overlay.png").exists()
    assert consumer.processed == 1
    assert consumer.errors == 0
    assert (output / "prediction.json").exists() == (split == "public")


def test_invalid_answer_and_recognition_error_are_preserved(setup):
    config, source, _ = setup
    detail = row()
    detail["answer"] = "malformed answer\n"
    checkpoint(source, detail, final=True)
    consumer = operator.ArtifactConsumer(config)
    consumer.scan()
    output = config.output_root / "train" / "S01" / "p1"
    assert (output / "answer.txt").read_text() == detail["answer"]
    saved = json.loads((output / "evaluation.json").read_text())
    assert saved["artifact_errors"][0]["stage"] == "prediction"
    assert (output / "truth-cells-overlay.png").exists()
    assert saved["score"] == 1  # Saved metric evidence is never recomputed or repaired.


def test_unsafe_ids_and_symlink_destinations(setup, tmp_path):
    config, source, example = setup
    unsafe = "../outside/\u043f\u0443\u0442\u044c"
    example["id"] = unsafe
    source.dataset_path.write_text(json.dumps(example) + "\n")
    detail = row()
    detail["example_id"] = unsafe
    checkpoint(source, detail, final=True)
    consumer = operator.ArtifactConsumer(config)
    consumer.scan()
    assert (config.output_root / "train" / "S01" / operator.component(unsafe)).is_dir()
    assert not (tmp_path / "outside").exists()
    malicious = config.output_root / "train" / "S01" / "p1"
    malicious.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        consumer.render(source, "S01", row(), deepcopy(example), Path("record.json"))


def test_supervisor_final_scan_and_console_redaction(setup, capsys):
    config, source, _ = setup
    path = source.state_dir / "evaluations" / "run1" / "record.json"
    script = (
        "import json, pathlib, sys; "
        f"p=pathlib.Path({str(path)!r}); p.parent.mkdir(parents=True); "
        f"p.write_text({json.dumps({'params': {'candidate_id': 'S01'}, 'examples': [row()]})!r}); "
        "print('SENSITIVE SERVICE OUTPUT'); sys.exit(7)"
    )
    result = operator.run(
        config, "supervise", interval=0.01, command=[sys.executable, "-c", script]
    )
    assert result == 7
    assert (config.output_root / "train" / "S01" / "p1" / "evaluation.json").exists()
    captured = capsys.readouterr()
    assert "SENSITIVE" not in captured.out + captured.err
    assert "score" not in captured.out + captured.err
    assert "SENSITIVE SERVICE OUTPUT" in (config.output_root / "service.stdout.log").read_text()


def test_watch_stop_and_config_validation(setup, tmp_path):
    config, source, _ = setup
    checkpoint(source, row(), final=True)
    stop = threading.Event()
    stop.set()
    assert operator.run(config, "watch", interval=0.01, stop=stop) == 0
    file = tmp_path / "config.json"
    data = {
        "output_root": "output",
        "metric_runtime": str(config.metric_runtime),
        "sources": [{"split": "train", "state_dir": "state", "dataset_path": "synthetic.jsonl"}],
    }
    file.write_text(json.dumps(data))
    loaded = operator.Configuration.load(file)
    assert loaded.sources[0].state_dir == source.state_dir
    data["sources"][0]["record_globs"] = ["../anything.json"]
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="relative"):
        operator.Configuration.load(file)


def test_one_bad_row_does_not_hide_other_processed_pages(setup):
    config, source, _ = setup
    path = checkpoint(source, row(), final=True)
    document = json.loads(path.read_text())
    document["examples"].insert(0, {"example_id": "unknown", "failure": "solver_failure"})
    document["interrupted_attempts"] = [{"example_id": "p1", "failure": "solver_failure"}]
    path.write_text(json.dumps(document))
    consumer = operator.ArtifactConsumer(config)
    consumer.scan()
    assert (config.output_root / "train" / "S01" / "p1" / "evaluation.json").is_file()
    output = config.output_root / "train" / "S01" / "p1"
    assert json.loads((output / "evaluation.json").read_text())["score"] == 1
    assert (output / "interrupted-attempts" / "0" / "evaluation.json").is_file()
    assert consumer.processed == 2
    assert consumer.errors == 1
    consumer.scan()
    assert consumer.errors == 1  # A corrupt persistent row is not reported on every poll.


def test_parent_symlinks_and_log_symlinks_are_rejected(setup, tmp_path):
    config, _, _ = setup
    consumer = operator.ArtifactConsumer(config)
    target = tmp_path / "separate"
    target.mkdir()
    (config.output_root / "train").symlink_to(target)
    (target / "S01").mkdir()
    with pytest.raises(ValueError, match="symlinks"):
        operator.private_directory(config.output_root / "train" / "S01")
    original = target / "existing.log"
    original.write_text("preserved")
    link = config.output_root / "service.stdout.log"
    link.symlink_to(original)
    with pytest.raises(OSError):
        operator.private_log(link)
    assert original.read_text() == "preserved"
    assert consumer.processed == 0
