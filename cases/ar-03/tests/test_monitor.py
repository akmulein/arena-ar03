"""Synthetic journal tests; no real datasets, model calls or closed artifacts."""

import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest
from evaluator.search import Journal, Resources

SPEC = importlib.util.spec_from_file_location(
    "ar03_monitor", Path(__file__).resolve().parents[1] / "monitor.py",
)
assert SPEC is not None and SPEC.loader is not None
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)

COMPARISON = {
    "split": "public", "dataset_id": "synthetic-v1", "subset_id": "all",
    "conditions_id": "synthetic-conditions",
}


def propose(journal, directory, parents=()):
    directory.mkdir()
    (directory / "solve.py").write_text("print('synthetic')\n")
    return journal.propose(None, directory, parents=parents, hypothesis="synthetic", change="test")


def measure(journal, candidate, identifier, *, score=None, error=None, split="public"):
    journal.begin(identifier, candidate, **{**COMPARISON, "split": split})
    resources = {} if split == "public" else {
        "solver": Resources(0, 1), "judge": Resources(0, 0.1),
    }
    journal.finish(identifier, score=score, error=error, **resources)


def test_failed_candidate_keeps_place_and_auc_gap(tmp_path):
    journal = Journal.create(tmp_path / "search.sqlite3", max_solutions=3)
    first = propose(journal, tmp_path / "first")
    second = propose(journal, tmp_path / "second", [first["candidate_id"]])
    third = propose(journal, tmp_path / "third", [first["candidate_id"], second["candidate_id"]])
    measure(journal, first["candidate_id"], "one", score=0.7)
    measure(journal, second["candidate_id"], "two", error="evaluation_failed")
    measure(journal, third["candidate_id"], "three", score=0.9)

    result = monitor.snapshot(journal, COMPARISON, stage="structure")

    assert result["summary"]["proposed_candidates"] == 3
    assert result["summary"]["evaluated_candidates"] == 3
    assert result["summary"]["failed_candidates"] == 1
    assert result["summary"]["best_teds"] == 0.9
    assert result["curve"]["auc"] is None
    assert result["curve"]["observed_auc"] == 0.7
    assert result["curve"]["failed_proposals"] == [2]
    assert result["lineage"][1]["state"] == "failed"
    assert result["lineage"][2]["parents"] == ["S01", "S02"]


def test_complete_public_uses_arena_auc_exactly(tmp_path):
    journal = Journal.create(tmp_path / "search.sqlite3", max_solutions=2)
    first = propose(journal, tmp_path / "first")
    second = propose(journal, tmp_path / "second")
    measure(journal, first["candidate_id"], "one", score=0.5)
    measure(journal, second["candidate_id"], "two", score=0.8)
    result = monitor.snapshot(journal, COMPARISON, stage="end-to-end")
    assert result["curve"]["auc"] == 1.3
    assert result["curve"]["mean_quality"] == 0.65


def test_train_does_not_invent_auc(tmp_path):
    journal = Journal.create(tmp_path / "search.sqlite3", max_solutions=1)
    candidate = propose(journal, tmp_path / "first")
    measure(journal, candidate["candidate_id"], "one", score=0.7, split="train")
    result = monitor.snapshot(journal, {**COMPARISON, "split": "train"}, stage="ocr")
    assert result["curve"]["auc"] is None
    assert result["curve"]["observed_auc"] is None
    assert result["curve"]["auc_applicable"] is False


def test_empty_search_renders_valid_svg_and_no_fabricated_best(tmp_path):
    journal = Journal.create(tmp_path / "search.sqlite3", max_solutions=3)
    result = monitor.snapshot(journal, COMPARISON, stage="detection")
    assert result["summary"]["best_teds"] is None
    output = tmp_path / "artifacts"
    monitor.save_artifacts(result, output)
    for name in ("teds.svg", "auc.svg", "lineage.svg"):
        ElementTree.parse(output / name)
    assert json.loads((output / "curve.json").read_text())["auc"] is None


def test_train_progress_counts_recognition_and_execution_errors():
    record = {
        "split": "train", "status": "RUNNING", "start_time_ns": 0,
        "example_ids": ["a", "b", "c", "d"],
        "examples": [
            {"statistics": {}, "answer": json.dumps({"error": "recognition"})},
            {"statistics": {}, "answer": json.dumps({"error": None})},
            {"failure": "solver_failure"},
        ],
    }
    result = monitor.train_progress(record, now_ns=12_000_000_000)
    assert result["processed_pages"] == 3
    assert result["evaluated_pages"] == 2
    assert result["errors"] == 2
    assert result["percent"] == 75
    assert result["elapsed_seconds"] == 12
    assert result["remaining_seconds"] == 4


def test_private_record_and_private_curve_are_rejected(tmp_path):
    journal = Journal.create(tmp_path / "search.sqlite3", max_solutions=3)
    with pytest.raises(ValueError, match="train"):
        monitor.train_progress({"split": "private"})
    with pytest.raises(ValueError, match="train or public"):
        monitor.snapshot(journal, {**COMPARISON, "split": "private"}, stage="ocr")


def test_running_and_unmeasured_candidates_render(tmp_path):
    journal = Journal.create(tmp_path / "search.sqlite3", max_solutions=3)
    first = propose(journal, tmp_path / "first")
    propose(journal, tmp_path / "second", [first["candidate_id"]])
    journal.begin("one", first["candidate_id"], **COMPARISON)
    result = monitor.snapshot(journal, COMPARISON, stage="detection")
    assert [item["state"] for item in result["lineage"]] == ["running", "unmeasured"]
    assert result["summary"]["evaluated_candidates"] == 0
    monitor.save_artifacts(result, tmp_path / "artifacts")
    ElementTree.parse(tmp_path / "artifacts" / "lineage.svg")


def test_cli_preserves_json_stdout_and_progress_on_stderr(tmp_path, capsys):
    journal_path = tmp_path / "search.sqlite3"
    Journal.create(journal_path, max_solutions=3)
    assert monitor.main([
        "--journal", str(journal_path), "--output", str(tmp_path / "artifacts"),
        "--split", "public", "--dataset", "synthetic-v1", "--subset", "all",
        "--conditions", "synthetic-conditions", "--stage", "detection",
    ]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["proposed_candidates"] == 0
    assert "proposed=0/3" in captured.err
    assert "best_public_TEDS=unknown" in captured.err
