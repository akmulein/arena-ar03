import base64
import json
import sys
import threading
from collections import Counter
from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from evaluator.checkpoints import read_document
from evaluator.metrics import CommandMetricError
from evaluator.reliability import add_usage, empty_usage
from evaluator.runner import SolveResult
from evaluator.service import Settings, create_app
from evaluator.trace_store import append_call
from evaluator.usage import add_usage as record_usage
from evaluator.usage import create_usage, seal_usage


def configuration(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluator.checkpoints.os.fsync", lambda descriptor: None)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(
        json.dumps({"id": str(index), "input": f"task-{index}"}) for index in range(4)
    ))
    script = tmp_path / "metric.py"
    script.write_text("print('{\"score\": 1}')\n")
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({"kind": "command", "version": "v1", "command": {
        "argv": [sys.executable, str(script)], "uses_model": True,
    }}))
    slots = tmp_path / "slots"
    slots.mkdir(mode=0o700)
    settings = Settings(
        dataset_path=dataset, metric_path=metric, state_dir=tmp_path / "state",
        token="test", runner_image="fixed", release_id="test", finalists=1, output_count=1,
        model="internal", model_base_url="http://internal/v1", uses_model=False,
        usage_dir=tmp_path / "usage", trace_dir=tmp_path / "traces",
        execution_slots=4, slot_dir=slots,
    )
    body = {"candidates": [{"id": "a", "files": {
        "prompt.txt": base64.b64encode(b"candidate").decode(),
    }}]}
    state = {"solves": [], "judges": [], "aggregates": [], "fail": False}

    def solve(candidate, task):
        answer = f"{candidate.name}:{task}"
        state["solves"].append(answer)
        return SolveResult(answer, empty_usage(), 0.25)

    def judge(metric, example, answer, complete, *, model_access, usage, **kwargs):
        state["judges"].append(answer)
        run_id = str(uuid4())
        create_usage(model_access.usage_dir, run_id)
        model_access.run_ids.append(run_id)
        failed = state["fail"] and example.id == "1"
        for _ in range(2 if failed else 1):
            measured = None if failed else {
                "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5,
            }
            add_usage(usage, measured)
            record_usage(model_access.usage_dir, run_id, measured)
            assert settings.trace_dir is not None
            append_call(settings.trace_dir, run_id, {
                "name": "model", "span_type": "LLM", "status": "ERROR" if failed else "OK",
                "attributes": {"usage": measured}, "inputs": {}, "outputs": {},
                "start_time_ns": 1, "end_time_ns": 2,
            })
        seal_usage(model_access.usage_dir, run_id)
        if failed:
            raise CommandMetricError("synthetic judge failure", {
                "model_run_id": run_id, "model_usage_complete": True,
            })
        return {"score": float(example.id), "criteria": {}}

    def aggregate(metric, details, **kwargs):
        state["aggregates"].append([item["example_id"] for item in details])
        return sum(item["score"] for item in details) / len(details)

    monkeypatch.setattr("evaluator.service.evaluate_answer", judge)
    monkeypatch.setattr("evaluator.service.aggregate_results", aggregate)
    return settings, body, state, solve, judge


def test_nonlast_failed_judge_resumes_without_repeating_parallel_answers(tmp_path, monkeypatch):
    settings, body, state, solve, judge = configuration(tmp_path, monkeypatch)
    barrier = threading.Barrier(4)
    state["fail"] = True

    def parallel_judge(*args, **kwargs):
        if state["fail"]:
            barrier.wait(timeout=3)
        return judge(*args, **kwargs)

    monkeypatch.setattr("evaluator.service.evaluate_answer", parallel_judge)
    app = create_app(settings, solve=solve)
    api = TestClient(app, headers={"Authorization": "Bearer test"})
    assert api.post("/v1/evaluations", json=body).status_code == 202
    record_path = next(settings.state_dir.glob("*/record.json"))
    assert json.loads(record_path.read_text())["status"] == "failed"
    details_path = record_path.parent / "details/a.json"
    before = read_document(details_path)
    assert [item["example_id"] for item in before["examples"]] == ["0", "1", "2", "3"]
    failed = deepcopy(before["examples"][1])
    assert failed["failure"] == "judge_failure"
    assert all("score" in before["examples"][index] for index in (0, 2, 3))
    assert state["aggregates"] == []
    solves_before = state["solves"].copy()
    state["fail"] = False
    assert app.state.resume_private("continue saved judge stage", check_only=True) == {
        "status": "resume_ready", "completed_examples": 3, "saved_answers": 4,
    }
    assert app.state.resume_private("continue saved judge stage") == {"status": "complete"}
    after = read_document(details_path)
    assert state["solves"] == solves_before
    assert Counter(state["solves"]) == Counter(f"a:task-{index}" for index in range(4))
    assert Counter(state["judges"]) == Counter([
        "a:task-0", "a:task-1", "a:task-1", "a:task-2", "a:task-3",
    ])
    assert state["aggregates"] == [["0", "1", "2", "3"]]
    assert all(after["examples"][index] == before["examples"][index] for index in (0, 2, 3))
    resumed = after["examples"][1]
    assert resumed["answer"] == failed["answer"]
    assert resumed["resources"]["solver"] == failed["resources"]["solver"]
    assert resumed["resources"]["judge"]["usage"]["attempts"] == 3
    assert resumed["resources"]["judge"]["usage"]["unreported_attempts"] == 2
    assert resumed["calls"][:len(failed["calls"])] == failed["calls"]


def interrupted_stage(tmp_path, monkeypatch, stage):
    import evaluator.service as service

    settings, body, state, solve, _ = configuration(tmp_path, monkeypatch)
    original_example = service.write_example
    original_json = service._write_json
    peers_ready = threading.Event()
    completed = set()
    lock = threading.Lock()
    crashed = False

    def ordered_solve(candidate, task):
        if task == "task-2" and stage != "solving":
            assert peers_ready.wait(3)
        return solve(candidate, task)

    def save_example(path, index, detail):
        nonlocal crashed
        original_example(path, index, detail)
        if "score" in detail and index != 2:
            with lock:
                completed.add(index)
                if completed == {0, 1, 3}:
                    peers_ready.set()
        if index == 2 and detail.get("status") == stage:
            crashed = True
            raise RuntimeError("synthetic process stopped at committed stage")

    def save_until_crash(path, value):
        if crashed and (path.name == "record.json" or path.parent.name == "details"):
            return
        original_json(path, value)

    monkeypatch.setattr(service, "write_example", save_example)
    monkeypatch.setattr(service, "_write_json", save_until_crash)
    app = create_app(settings, solve=ordered_solve)
    api = TestClient(app, headers={"Authorization": "Bearer test"})
    assert api.post("/v1/evaluations", json=body).status_code == 202
    assert crashed
    record_path = next(settings.state_dir.glob("*/record.json"))
    disk_record = json.loads(record_path.read_text())
    assert disk_record["status"] == "running"
    assert disk_record["example_workers"] == settings.execution_slots
    details_path = record_path.parent / "details/a.json"
    assert json.loads(details_path.read_text())["checkpoint_format"] == "arena.examples.v1"
    before = read_document(details_path)
    saved = next(item for item in before["examples"] if item["example_id"] == "2")
    assert saved["status"] == stage
    assert "a:task-2" not in state["judges"]
    monkeypatch.setattr(service, "write_example", original_example)
    monkeypatch.setattr(service, "_write_json", original_json)
    app = create_app(settings, solve=solve)
    record = json.loads(record_path.read_text())
    assert record["status"] == "failed" and record["failure"] == "interrupted"
    return settings, state, app, details_path, before


def test_crash_after_committed_solver_resumes_only_its_pending_judge(tmp_path, monkeypatch):
    settings, state, app, details_path, before = interrupted_stage(tmp_path, monkeypatch, "solved")
    assert [item["example_id"] for item in before["examples"]] == ["0", "1", "2", "3"]
    assert all("score" in before["examples"][index] for index in (0, 1, 3))
    files_before = {path: path.read_bytes() for path in settings.state_dir.rglob("*.json")}
    previous = deepcopy(state)
    assert app.state.resume_private("verify interrupted solver", check_only=True) == {
        "status": "resume_ready", "completed_examples": 3, "saved_answers": 4,
    }
    assert state == previous
    assert {path: path.read_bytes() for path in settings.state_dir.rglob("*.json")} == files_before
    assert app.state.resume_private("finish saved solver answer") == {"status": "complete"}
    after = read_document(details_path)
    assert state["solves"] == previous["solves"]
    assert Counter(state["solves"]) == Counter(f"a:task-{index}" for index in range(4))
    assert Counter(state["judges"]) == Counter(f"a:task-{index}" for index in range(4))
    assert state["aggregates"] == [["0", "1", "2", "3"]]
    assert all(after["examples"][index] == before["examples"][index] for index in (0, 1, 3))
    assert after["examples"][2]["answer"] == before["examples"][2]["answer"]
    assert (
        after["examples"][2]["resources"]["solver"]
        == before["examples"][2]["resources"]["solver"]
    )


@pytest.mark.parametrize("stage", ["solving", "judging"])
def test_unknown_active_stage_rejects_resume_without_repeating_work(tmp_path, monkeypatch, stage):
    settings, state, app, _, _ = interrupted_stage(tmp_path, monkeypatch, stage)
    previous = deepcopy(state)
    files_before = {path: path.read_bytes() for path in settings.state_dir.rglob("*.json")}
    with pytest.raises(ValueError):
        app.state.resume_private("verify unknown active stage")
    assert state == previous
    assert {path: path.read_bytes() for path in settings.state_dir.rglob("*.json")} == files_before
