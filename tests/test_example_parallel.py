import json
import threading
from dataclasses import replace

import pytest

from tests.test_private_parallel import api_for, configuration, record_path, solved


@pytest.mark.parametrize("split", ["private", "public"])
def test_one_candidate_uses_available_slots_and_keeps_dataset_order(tmp_path, monkeypatch, split):
    settings, body = configuration(tmp_path, monkeypatch, finalists=1, examples=4)
    settings = replace(settings, split=split)
    barrier = threading.Barrier(4)
    calls = []

    def solve(candidate, task):
        calls.append(task)
        barrier.wait(timeout=2)
        return solved(candidate, task)

    _, api = api_for(settings, solve)
    assert api.post("/v1/evaluations", json=body).status_code == 202
    path = next(settings.state_dir.glob("**/record.json"))
    record = json.loads(path.read_text())
    assert record["status"] == "complete"
    assert sorted(calls) == [f"task-{index}" for index in range(4)]
    details = json.loads((path.parent / "details/a.json").read_text())
    assert [item["example_id"] for item in details["examples"]] == list("0123")
    assert record["scores"][0]["solver_tokens"] == 4


def test_private_pause_preserves_sparse_completed_examples(tmp_path, monkeypatch):
    settings, body = configuration(tmp_path, monkeypatch, finalists=1, examples=6)
    gate = threading.Barrier(4)
    release = threading.Event()
    first = threading.Event()
    calls = []
    app = None

    def solve(candidate, task):
        calls.append(task)
        gate.wait(timeout=2)
        if task == "task-0":
            first.set()
            assert release.wait(3)
        else:
            assert app is not None
            app.state.pause_private("operator pause")
        return solved(candidate, task)

    app, api = api_for(settings, solve)
    worker = threading.Thread(target=lambda: api.post("/v1/evaluations", json=body))
    worker.start()
    try:
        assert first.wait(2)
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    path = record_path(settings)
    assert json.loads(path.read_text())["status"] == "paused"
    assert len(calls) == 4
    details = json.loads((path.parent / "details/a.json").read_text())
    assert [item["example_id"] for item in details["examples"]] == list("0123")


def test_public_campaign_requires_registered_exact_running_measurement(tmp_path, monkeypatch):
    from evaluator.search import Journal

    settings, body = configuration(tmp_path, monkeypatch, finalists=1, examples=1)
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "prompt.txt").write_text("candidate-0")
    journal = Journal.create(tmp_path / "search.sqlite", max_solutions=10)
    settings = replace(settings, split="public", search_mode="campaign",
                       search_journal=journal.path, search_max_solutions=10)
    calls = []

    def solve(candidate, task):
        calls.append(task)
        return solved(candidate, task)

    _, api = api_for(settings, solve)
    assert api.post("/v1/evaluations", json=body).status_code == 409
    assert calls == []
    journal.propose("a", source, hypothesis="test", change="initial")
    assert api.post("/v1/evaluations", json=body).status_code == 409
    journal.begin("public-a", "a", split="public", dataset_id="data", subset_id="full",
                  conditions_id="fixed")
    altered = {"candidates": [{**body["candidates"][0], "files": {"prompt.txt": "eA=="}}]}
    assert api.post("/v1/evaluations", json=altered).status_code == 409
    assert calls == []
    assert api.post("/v1/evaluations", json=body).status_code == 202
    journal.finish("public-a", score=1)
    assert api.post("/v1/evaluations", json=body).status_code == 200
    assert calls == ["task-0"]


def test_aggregation_holds_global_capacity_slot(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from evaluator.capacity import CapacityError, acquire_slot

    settings, body = configuration(tmp_path, monkeypatch, finalists=1, examples=1)
    settings = replace(settings, execution_slots=1)
    started = threading.Event()
    release = threading.Event()

    def aggregate(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return 1

    monkeypatch.setattr("evaluator.service.aggregate_results", aggregate)
    _, api = api_for(settings, solved)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(api.post, "/v1/evaluations", json=body)
        try:
            assert started.wait(2)
            with pytest.raises(CapacityError):
                acquire_slot(settings.slot_dir, 1)
        finally:
            release.set()
        assert pending.result(timeout=3).status_code == 202
