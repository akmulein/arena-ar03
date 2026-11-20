import base64
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from evaluator.capacity import CapacityError, acquire_slot
from evaluator.reliability import empty_usage
from evaluator.runner import SolveResult
from evaluator.service import Settings, create_app


def configuration(tmp_path, monkeypatch, *, finalists=6, examples=2):
    monkeypatch.setattr("evaluator.checkpoints.os.fsync", lambda descriptor: None)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(
        json.dumps({"id": str(index), "input": f"task-{index}"})
        for index in range(examples)
    ))
    script = tmp_path / "metric.py"
    script.write_text("print('{\"score\": 1}')\n")
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({"kind": "command", "version": "v1", "command": {
        "argv": [sys.executable, str(script)], "uses_model": False,
    }}))
    slots = tmp_path / "slots"
    slots.mkdir(mode=0o700)
    settings = Settings(
        dataset_path=dataset, metric_path=metric, state_dir=tmp_path / "state",
        token="test", runner_image="fixed", release_id="test", finalists=finalists,
        model="internal", model_base_url="http://internal/v1",
        output_count=min(3, finalists), slot_dir=slots, execution_slots=4,
        trace_dir=tmp_path / "traces",
    )
    monkeypatch.setattr("evaluator.service.evaluate_answer", lambda *args, **kwargs: {"score": 1})
    monkeypatch.setattr("evaluator.service.aggregate_results", lambda *args, **kwargs: 1)
    body = {"candidates": [{"id": chr(97 + index), "files": {
        "prompt.txt": base64.b64encode(f"candidate-{index}".encode()).decode(),
    }} for index in range(finalists)]}
    return settings, body


def solved(candidate, task):
    tokens = ord(candidate.name) - 96
    return SolveResult(f"{candidate.name}:{task}", {
        **empty_usage(), "prompt_tokens": tokens, "total_tokens": tokens, "attempts": 1,
    }, tokens / 10)


def api_for(settings, solve):
    app = create_app(settings, solve=solve)
    return app, TestClient(app, headers={"Authorization": "Bearer test"})


def record_path(settings):
    return next(settings.state_dir.glob("*/record.json"))


def test_private_candidates_use_all_and_only_available_global_slots(tmp_path, monkeypatch):
    settings, body = configuration(tmp_path, monkeypatch)
    occupied = acquire_slot(settings.slot_dir, 4)
    assert occupied is not None
    release = threading.Event()
    condition = threading.Condition()
    started = set()
    calls = []
    active = 0
    peak = 0

    def solve(candidate, task):
        nonlocal active, peak
        with condition:
            calls.append((candidate.name, task))
            started.add((candidate.name, task))
            active += 1
            peak = max(peak, active)
            condition.notify_all()
        try:
            assert release.wait(5)
            return solved(candidate, task)
        finally:
            with condition:
                active -= 1

    _, api = api_for(settings, solve)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api.post, "/v1/evaluations", json=body)
        try:
            with condition:
                assert condition.wait_for(lambda: len(started) == 3, timeout=2)
                assert active == 3
            with pytest.raises(CapacityError):
                acquire_slot(settings.slot_dir, 4)
            occupied.close()
            with condition:
                assert condition.wait_for(lambda: len(started) == 4, timeout=2)
                assert active == 4
            with pytest.raises(CapacityError):
                acquire_slot(settings.slot_dir, 4)
        finally:
            occupied.close()
            release.set()
        assert pending.result(timeout=5).status_code == 202
    assert peak == 4
    assert Counter(calls) == Counter((chr(97 + index), f"task-{task}")
                                    for index in range(6) for task in range(2))
    record = json.loads(record_path(settings).read_text())
    assert record["status"] == "complete"
    assert record["winners"] == ["a", "b", "c"]
    assert [item["candidate_id"] for item in record["scores"]] == list("abcdef")
    for index, score in enumerate(record["scores"], start=1):
        assert score["solver_tokens"] == index * 2
        assert score["solver_seconds"] == pytest.approx(index / 5)
    leases = [acquire_slot(settings.slot_dir, 4) for _ in range(4)]
    for lease in leases:
        assert lease is not None
        lease.close()


@pytest.fixture
def paused_run(tmp_path, monkeypatch):
    import evaluator.service as service

    settings, body = configuration(tmp_path, monkeypatch, finalists=5)
    release = threading.Event()
    condition = threading.Condition()
    completed = set()
    calls = []
    write = service._write_json

    def write_checkpoint(path, value):
        write(path, value)
        if (path.parent.name == "details"
                and value.get("aggregation", {}).get("status") == "complete"):
            with condition:
                completed.add(value["candidate_id"])
                condition.notify_all()

    def solve(candidate, task):
        with condition:
            calls.append((candidate.name, task))
            condition.notify_all()
        if candidate.name == "a" and task == "task-0":
            assert release.wait(5)
        return solved(candidate, task)

    monkeypatch.setattr(service, "_write_json", write_checkpoint)
    app, api = api_for(settings, solve)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api.post, "/v1/evaluations", json=body)
        try:
            with condition:
                assert condition.wait_for(
                    lambda: completed == set("bcde") and ("a", "task-1") in calls, timeout=3,
                )
            app.state.pause_private("authorized operator migration")
            assert api.get("/v1/evaluations").json() == {"status": "running"}
            assert ("a", "task-1") in calls
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 202
    path = record_path(settings)
    assert json.loads(path.read_text())["status"] == "paused"
    assert ("a", "task-1") in calls
    assert Counter(calls) == Counter(
        (name, f"task-{index}") for name in "abcde" for index in range(2)
    )
    return settings, body, app, api, path, solve, calls


def test_pause_drains_current_example_and_resume_reuses_out_of_order_candidates(paused_run):
    settings, body, _, _, path, solve, calls = paused_run
    record_before = json.loads(path.read_text())
    details = path.parent / "details"
    completed = {name: (details / f"{name}.json").read_bytes() for name in "bcde"}
    first = json.loads((details / "a.json").read_text())["examples"][0]
    app, api = api_for(settings, solve)
    files = {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")}
    previous_calls = calls.copy()
    check = app.state.resume_private("validate migration", check_only=True)
    assert check["status"] == "resume_ready"
    assert check["completed_examples"] == 10
    assert check["saved_answers"] == 10
    assert calls == previous_calls
    assert {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")} == files
    assert app.state.resume_private("continue accepted batch") == {"status": "complete"}
    assert Counter(calls) == Counter((name, f"task-{index}")
                                    for name in "abcde" for index in range(2))
    assert json.loads((details / "a.json").read_text())["examples"][0] == first
    assert all((details / f"{name}.json").read_bytes() == content
               for name, content in completed.items())
    after = json.loads(path.read_text())
    for key in ("batch_digest", "candidates", "provenance", "release_id"):
        assert after[key] == record_before[key]
    assert [score["candidate_id"] for score in after["scores"]] == list("abcde")
    assert after["winners"] == ["a", "b", "c"]
    for index, score in enumerate(after["scores"], start=1):
        assert score["solver_tokens"] == index * 2
        assert score["solver_seconds"] == pytest.approx(index / 5)
    assert api.post("/v1/evaluations", json=body).json() == {"status": "complete"}
    assert Counter(calls) == Counter((name, f"task-{index}")
                                    for name in "abcde" for index in range(2))


def test_paused_private_replays_keep_batch_sealed_without_resuming(paused_run):
    _, body, _, api, path, _, calls = paused_run
    previous = calls.copy()
    assert api.get("/v1/evaluations").json() == {"status": "running"}
    assert api.post("/v1/evaluations", json=body).json() == {"status": "running"}
    changed = deepcopy(body)
    changed["candidates"][0]["files"]["prompt.txt"] = base64.b64encode(b"changed").decode()
    assert api.post("/v1/evaluations", json=changed).status_code == 409
    assert calls == previous
    assert json.loads(path.read_text())["status"] == "paused"


@pytest.mark.parametrize("damage", [
    "candidate", "wrong_example", "pending_judge", "provenance", "missing_details", "resources",
])
def test_paused_resume_rejects_unknown_or_changed_progress_without_solving(paused_run, damage):
    settings, _, app, _, path, _, calls = paused_run
    record = json.loads(path.read_text())
    details_path = path.parent / "details/a.json"
    document = json.loads(details_path.read_text())
    if damage == "candidate":
        (Path(record["candidates"][0]["path"]) / "prompt.txt").write_text("changed")
    elif damage == "provenance":
        record["provenance"]["parameters"] = {"changed": True}
        path.write_text(json.dumps(record))
    elif damage == "missing_details":
        details_path.unlink()
    else:
        detail = document["examples"][0]
        if damage == "wrong_example":
            detail["example_id"] = "different"
        elif damage == "pending_judge":
            detail["status"] = "judging"
        else:
            detail["resources"]["solver"]["usage"]["total_tokens"] += 1
        details_path.write_text(json.dumps(document))
    before = {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")}
    previous = calls.copy()
    with pytest.raises((ValueError, RuntimeError)):
        app.state.resume_private("technical migration")
    assert calls == previous
    assert {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")} == before


def test_private_batches_share_the_global_slot_limit(tmp_path, monkeypatch):
    first, body = configuration(tmp_path, monkeypatch, finalists=3, examples=1)
    second = replace(first, state_dir=tmp_path / "other-state")
    condition = threading.Condition()
    release = threading.Event()
    active = 0
    peak = 0
    started = []

    def solve(candidate, task):
        nonlocal active, peak
        with condition:
            active += 1
            peak = max(active, peak)
            started.append(candidate)
            condition.notify_all()
        try:
            assert release.wait(5)
            return solved(candidate, task)
        finally:
            with condition:
                active -= 1

    _, first_api = api_for(first, solve)
    _, second_api = api_for(second, solve)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_request = pool.submit(first_api.post, "/v1/evaluations", json=body)
        second_request = None
        try:
            with condition:
                assert condition.wait_for(lambda: len(started) == 3, timeout=2)
            second_request = pool.submit(second_api.post, "/v1/evaluations", json=body)
            with condition:
                assert condition.wait_for(lambda: len(started) == 4, timeout=2)
                assert active == 4
            with pytest.raises(CapacityError):
                acquire_slot(first.slot_dir, 4)
        finally:
            release.set()
        assert first_request.result(timeout=5).status_code == 202
        assert second_request is not None
        assert second_request.result(timeout=5).status_code == 202
    assert peak == 4
    assert len(started) == 6
    assert json.loads(record_path(first).read_text())["status"] == "complete"
    assert json.loads(record_path(second).read_text())["status"] == "complete"


def test_pause_prevents_a_waiting_worker_from_starting_after_global_slot_release(
    tmp_path, monkeypatch,
):
    settings, body = configuration(tmp_path, monkeypatch, finalists=5, examples=1)
    occupied = [acquire_slot(settings.slot_dir, 4) for _ in range(3)]
    release = threading.Event()
    started = threading.Event()
    calls = []

    def solve(candidate, task):
        calls.append((candidate.name, task))
        started.set()
        assert release.wait(5)
        return solved(candidate, task)

    app, api = api_for(settings, solve)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api.post, "/v1/evaluations", json=body)
        try:
            assert started.wait(2)
            app.state.pause_private("operator pause before expanding capacity")
            for lease in occupied:
                assert lease is not None
                lease.close()
        finally:
            for lease in occupied:
                if lease is not None:
                    lease.close()
            release.set()
        assert pending.result(timeout=5).status_code == 202
    assert len(calls) == 1
    assert json.loads(record_path(settings).read_text())["status"] == "paused"


def test_failure_waits_for_active_examples_and_does_not_launch_remaining_candidates(
    tmp_path, monkeypatch,
):
    import evaluator.service as service
    from evaluator.runner import RunnerError

    settings, body = configuration(tmp_path, monkeypatch, examples=1)
    condition = threading.Condition()
    fail = threading.Event()
    release = threading.Event()
    failure_written = threading.Event()
    terminal_written = threading.Event()
    started = set()
    calls = []
    write = service._write_json

    def write_checkpoint(path, value):
        write(path, value)
        if path.name == "a.json" and value.get("examples", [{}])[-1].get("failure"):
            failure_written.set()
        if path.name == "record.json" and value.get("status") == "failed":
            terminal_written.set()

    def solve(candidate, task):
        with condition:
            started.add(candidate.name)
            calls.append((candidate.name, task))
            condition.notify_all()
        if candidate.name == "a":
            assert fail.wait(5)
            raise RunnerError("synthetic solver failure")
        assert release.wait(5)
        return solved(candidate, task)

    monkeypatch.setattr(service, "_write_json", write_checkpoint)
    _, api = api_for(settings, solve)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api.post, "/v1/evaluations", json=body)
        try:
            with condition:
                assert condition.wait_for(lambda: started == set("abcd"), timeout=2)
            fail.set()
            assert failure_written.wait(2)
            assert not terminal_written.wait(0.1)
            assert not pending.done()
        finally:
            fail.set()
            release.set()
        assert pending.result(timeout=5).status_code == 202
    assert Counter(calls) == Counter((name, "task-0") for name in "abcd")
    path = record_path(settings)
    assert json.loads(path.read_text())["status"] == "failed"
    for name in "bcd":
        document = json.loads((path.parent / "details" / f"{name}.json").read_text())
        assert len(document["examples"]) == 1
        assert "failure" not in document["examples"][0]
        assert "judge" in document["examples"][0]["resources"]
    leases = [acquire_slot(settings.slot_dir, 4) for _ in range(4)]
    for lease in leases:
        assert lease is not None
        lease.close()


def test_parallel_native_judges_keep_candidate_usage_isolated(tmp_path, monkeypatch):
    from evaluator.judge import ModelClient
    from evaluator.metrics import evaluate_answer
    from evaluator.reliability import add_usage

    settings, body = configuration(tmp_path, monkeypatch, finalists=4, examples=1)
    settings.dataset_path.write_text(json.dumps({
        "id": "0", "input": "task-0", "reference": "synthetic reference",
    }))
    settings.metric_path.write_text(json.dumps({
        "kind": "llm", "version": "v1", "instruction": "Synthetic judging", "votes": 2,
        "criteria": [{"id": "correct", "question": "Correct?", "weight": 1}],
    }))
    barrier = threading.Barrier(4)

    def attempt(self, messages, validator, attributes):
        answer = json.loads(messages[-1]["content"])["answer"]
        index = ord(answer[0]) - 96
        if attributes["vote"] == 1:
            barrier.wait(timeout=2)
        add_usage(self.usage, {
            "prompt_tokens": 10 * index, "completion_tokens": 1, "total_tokens": 10 * index + 1,
        })
        result = '{"score": 4, "reason": "synthetic"}'
        if validator is not None:
            validator(result)
        return result

    monkeypatch.setattr("evaluator.service.evaluate_answer", evaluate_answer)
    monkeypatch.setattr(ModelClient, "_attempt", attempt)
    _, api = api_for(settings, solved)
    assert api.post("/v1/evaluations", json=body).status_code == 202
    path = record_path(settings)
    record = json.loads(path.read_text())
    assert record["status"] == "complete"
    for index, score in enumerate(record["scores"], start=1):
        assert score["judge_tokens"] == 2 * (10 * index + 1)
        assert score["judge_unreported_attempts"] == 0
        details_path = path.parent / "details" / f"{score['candidate_id']}.json"
        document = json.loads(details_path.read_text())
        assert document["examples"][0]["resources"]["judge"]["usage"] == {
            "prompt_tokens": 20 * index, "completion_tokens": 2, "total_tokens": 20 * index + 2,
            "attempts": 2, "unreported_attempts": 0,
        }


def interrupted_tail(settings, path, *, closed=True, active=False):
    from evaluator.usage import begin_request, create_usage, seal_usage

    run_id = str(uuid4())
    usage_dir = settings.usage_dir or settings.state_dir / "usage"
    capability = create_usage(usage_dir, run_id)
    if active:
        begin_request(usage_dir, run_id, capability)
    if closed:
        if active:
            with pytest.raises(ValueError, match="active requests"):
                seal_usage(usage_dir, run_id)
        else:
            seal_usage(usage_dir, run_id)
    attempt = {
        "example_id": "1", "failure": "solver_failure", "error_type": "RunnerError",
        "run_id": run_id, "resources_complete": True,
        "resources": {
            "solver": {"usage": empty_usage(), "wall_seconds": 5.0, "run_id": run_id},
            "judge": {"usage": empty_usage(), "wall_seconds": 0.0},
        },
        "calls": [{
            "span_type": "CHAIN", "name": "solver", "status": "ERROR", "inputs": {},
            "outputs": {"failure": "solver_failure"}, "start_time_ns": 1, "end_time_ns": 2,
            "attributes": {"run_id": run_id},
        }],
    }
    details_path = path.parent / "details/a.json"
    document = json.loads(details_path.read_text())
    document["interrupted_attempts"] = [attempt]
    details_path.write_text(json.dumps(document))
    record = json.loads(path.read_text())
    record["pause"]["checkpoints"]["a"]["interrupted_attempts"] = 1
    path.write_text(json.dumps(record))
    return attempt, details_path, usage_dir


def test_zero_call_interrupted_tail_preserves_cost_without_quality_or_answer_repetition(paused_run):
    settings, _, app, _, path, _, calls = paused_run
    attempt, details_path, _ = interrupted_tail(settings, path)
    before = deepcopy(json.loads(details_path.read_text())["examples"][0])
    assert app.state.resume_private("operator migration") == {"status": "complete"}
    document = json.loads(details_path.read_text())
    assert document["interrupted_attempts"] == [attempt]
    assert document["examples"][0] == before
    assert len(document["examples"]) == 2
    assert Counter(calls) == Counter((name, f"task-{index}")
                                    for name in "abcde" for index in range(2))
    score = next(item for item in json.loads(path.read_text())["scores"]
                 if item["candidate_id"] == "a")
    assert score["score"] == 1
    assert score["solver_seconds"] == pytest.approx(5.2)
    assert score["solver_tokens"] == 2
    assert score["solver_unreported_attempts"] == 0


@pytest.mark.parametrize("damage", [
    "model_usage", "model_trace", "answer", "open_usage", "active_request", "wrong_example",
])
def test_interrupted_tail_with_unknown_or_model_progress_is_rejected_before_work(
    paused_run, damage,
):
    from evaluator.trace_store import append_call
    from evaluator.usage import add_usage

    settings, _, app, _, path, _, calls = paused_run
    attempt, details_path, usage_dir = interrupted_tail(
        settings, path, closed=damage not in {"open_usage", "model_usage"},
        active=damage == "active_request",
    )
    if damage == "model_usage":
        from evaluator.usage import seal_usage

        add_usage(usage_dir, attempt["run_id"], {
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
        })
        seal_usage(usage_dir, attempt["run_id"])
    elif damage == "model_trace":
        assert settings.trace_dir is not None
        append_call(settings.trace_dir, attempt["run_id"], {
            "span_type": "LLM", "status": "ERROR", "inputs": {}, "outputs": {},
            "start_time_ns": 1, "end_time_ns": 2,
        })
    elif damage in {"answer", "wrong_example"}:
        if damage == "answer":
            attempt["answer"] = "already produced answer"
        else:
            attempt["example_id"] = "unknown"
        document = json.loads(details_path.read_text())
        document["interrupted_attempts"] = [attempt]
        details_path.write_text(json.dumps(document))
    previous = calls.copy()
    before = {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")}
    with pytest.raises((ValueError, RuntimeError)):
        app.state.resume_private("operator migration")
    assert calls == previous
    assert {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")} == before


@pytest.mark.parametrize("damage", [
    "duplicate_within_candidate", "duplicate_across_candidates", "completed_solver_collision",
    "solver_run_ids", "judge_run_id", "judge_run_ids",
])
def test_interrupted_tail_rejects_duplicate_or_contradictory_run_ownership(paused_run, damage):
    settings, _, app, _, path, _, calls = paused_run
    attempt, details_path, _ = interrupted_tail(settings, path)
    document = json.loads(details_path.read_text())
    record = json.loads(path.read_text())
    interrupted_resources = document["interrupted_attempts"][0]["resources"]
    if damage == "duplicate_within_candidate":
        document["interrupted_attempts"].append(deepcopy(attempt))
        record["pause"]["checkpoints"]["a"]["interrupted_attempts"] = 2
    elif damage == "duplicate_across_candidates":
        other_path = path.parent / "details/b.json"
        other = json.loads(other_path.read_text())
        other["interrupted_attempts"] = [deepcopy(attempt)]
        other_path.write_text(json.dumps(other))
        record["pause"]["checkpoints"]["b"]["interrupted_attempts"] = 1
    elif damage == "completed_solver_collision":
        solver = document["examples"][0]["resources"]["solver"]
        solver["run_id"] = attempt["run_id"]
        solver["usage"] = empty_usage()
    elif damage == "solver_run_ids":
        interrupted_resources["solver"]["run_ids"] = [attempt["run_id"]]
    elif damage == "judge_run_id":
        interrupted_resources["judge"]["run_id"] = attempt["run_id"]
    else:
        interrupted_resources["judge"]["run_ids"] = [attempt["run_id"]]
    details_path.write_text(json.dumps(document))
    path.write_text(json.dumps(record))
    previous = calls.copy()
    before = {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")}
    with pytest.raises((ValueError, RuntimeError)):
        app.state.resume_private("operator migration", check_only=True)
    assert calls == previous
    assert {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")} == before


def native_paused_run(tmp_path, monkeypatch, *, default_uses_model, selected_uses_model):
    import evaluator.service as service
    from evaluator.runner import DockerRunner

    settings, body = configuration(tmp_path, monkeypatch, finalists=2)
    registry = {} if selected_uses_model is None else {
        "selected": {"image": "selected-image", "uses_model": selected_uses_model},
    }
    settings = replace(settings, uses_model=default_uses_model, harnesses=registry)
    if registry:
        for candidate in body["candidates"]:
            candidate["files"]["solution.json"] = base64.b64encode(
                json.dumps({"harness": "selected"}).encode(),
            ).decode()
    release = threading.Event()
    other_completed = threading.Event()
    other_example_started = threading.Event()
    calls = []
    write = service._write_json

    def write_checkpoint(path, value):
        write(path, value)
        if (path.name == "b.json"
                and value.get("aggregation", {}).get("status") == "complete"):
            other_completed.set()

    def run(self, candidate, task, *, parameters=None):
        uses_model = default_uses_model if selected_uses_model is None else selected_uses_model
        assert self.uses_model == uses_model
        calls.append((candidate.name, task))
        if candidate.name == "a" and task == "task-1":
            other_example_started.set()
        if candidate.name == "a" and task == "task-0":
            assert release.wait(5)
        return SolveResult(f"{candidate.name}:{task}", empty_usage(), 0.5, str(uuid4()))

    monkeypatch.setattr(service, "_write_json", write_checkpoint)
    monkeypatch.setattr(DockerRunner, "__call__", run)
    app = create_app(settings)
    api = TestClient(app, headers={"Authorization": "Bearer test"})
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api.post, "/v1/evaluations", json=body)
        try:
            assert other_completed.wait(3)
            assert other_example_started.wait(3)
            app.state.pause_private("operator native runner migration")
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 202
    path = record_path(settings)
    assert json.loads(path.read_text())["status"] == "paused"
    assert ("a", "task-1") in calls
    return settings, path, calls


@pytest.mark.parametrize("default_uses_model,selected_uses_model", [
    (True, False), (False, True), (False, None),
])
def test_paused_resume_uses_selected_harness_model_contract(
    tmp_path, monkeypatch, default_uses_model, selected_uses_model,
):
    settings, path, calls = native_paused_run(
        tmp_path, monkeypatch, default_uses_model=default_uses_model,
        selected_uses_model=selected_uses_model,
    )
    assert not list((settings.state_dir / "usage").glob("*.json"))
    app = create_app(settings)
    previous = calls.copy()
    before = {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")}
    uses_model = default_uses_model if selected_uses_model is None else selected_uses_model
    if uses_model:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            app.state.resume_private("validate selected model accounting", check_only=True)
        assert calls == previous
        after = {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")}
        assert after == before
        return
    check = app.state.resume_private("validate model-free checkpoint", check_only=True)
    assert check["status"] == "resume_ready"
    assert calls == previous
    assert {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")} == before
    completed = (path.parent / "details/b.json").read_bytes()
    first = json.loads((path.parent / "details/a.json").read_text())["examples"][0]
    assert app.state.resume_private("continue model-free candidate") == {"status": "complete"}
    assert Counter(calls) == Counter((name, f"task-{index}")
                                    for name in "ab" for index in range(2))
    assert (path.parent / "details/b.json").read_bytes() == completed
    assert json.loads((path.parent / "details/a.json").read_text())["examples"][0] == first


def test_paused_model_free_harness_rejects_nonzero_model_usage(tmp_path, monkeypatch):
    settings, path, calls = native_paused_run(
        tmp_path, monkeypatch, default_uses_model=True, selected_uses_model=False,
    )
    details_path = path.parent / "details/a.json"
    document = json.loads(details_path.read_text())
    document["examples"][0]["resources"]["solver"]["usage"] = {
        **empty_usage(), "prompt_tokens": 1, "total_tokens": 1, "attempts": 1,
    }
    details_path.write_text(json.dumps(document))
    app = create_app(settings)
    previous = calls.copy()
    before = {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")}
    with pytest.raises((ValueError, RuntimeError, OSError)):
        app.state.resume_private("validate model-free checkpoint", check_only=True)
    assert calls == previous
    assert {str(item): item.read_bytes() for item in settings.state_dir.rglob("*.json")} == before
