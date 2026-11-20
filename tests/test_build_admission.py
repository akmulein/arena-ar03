import base64
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evaluator.build import BuildStore, install_build_routes
from evaluator.contracts import Candidate
from evaluator.reliability import empty_usage
from evaluator.runner import SolveResult


def candidate(candidate_id="S1", source="print('{}')"):
    return Candidate(id=candidate_id, files={
        "build.py": base64.b64encode(source.encode()).decode(),
    })


def client_with_admission(tmp_path, admission):
    runs = []

    def build(candidate_dir, output):
        runs.append(candidate_dir)
        (output / "prepared.txt").write_text("prepared")
        return SolveResult("build complete", empty_usage(), 1)

    store = BuildStore(
        tmp_path / "builds", tmp_path / "assets", "a" * 64, "b" * 64,
        build, admission=admission,
    )
    app = FastAPI()
    install_build_routes(app, store)
    return TestClient(app), store, runs


@pytest.mark.parametrize("problem", ["unregistered", "modified", "stopped"])
def test_raw_build_rejects_candidate_before_launch_or_reservation(tmp_path, problem):
    registered = candidate()
    submitted = candidate("S2") if problem == "unregistered" else (
        candidate(source="print('{\"changed\":true}')") if problem == "modified" else registered
    )

    @contextmanager
    def admission(item):
        if problem == "stopped" or (item.id, item.digest()) != (
            registered.id, registered.digest(),
        ):
            raise ValueError("candidate is not admitted")
        yield

    client, store, runs = client_with_admission(tmp_path, admission)
    response = client.post("/v1/builds", json={"candidate": submitted.model_dump()})
    assert response.status_code == 422
    assert runs == []
    assert list((store.root / "receipts").iterdir()) == []
    assert list((store.root / "jobs").iterdir()) == []


def test_build_admission_covers_acceptance_and_cached_receipt_survives_stop(tmp_path):
    item = candidate()
    admissions = []
    stopped = False

    @contextmanager
    def admission(selected):
        if stopped:
            raise ValueError("campaign stopped")
        admissions.append((selected.id, selected.digest()))
        yield
        assert store.read(selected.digest())["status"] in {"RUNNING", "FINISHED"}

    client, store, runs = client_with_admission(tmp_path, admission)
    response = client.post("/v1/builds", json={"candidate": item.model_dump()})
    assert response.status_code == 202
    stopped = True
    repeat = client.post("/v1/builds", json={"candidate": item.model_dump()})
    assert repeat.status_code == 200
    assert repeat.json()["status"] == "FINISHED"
    assert client.get(f"/v1/builds/{item.digest()}").json() == repeat.json()
    alias = client.post("/v1/builds", json={"candidate": candidate("S2").model_dump()})
    assert alias.status_code == 200
    assert alias.json() == repeat.json()
    assert alias.json()["candidate_id"] == item.id
    assert admissions == [(item.id, item.digest())]
    assert len(runs) == 1


def test_standalone_build_does_not_require_a_journal(tmp_path):
    client, _, runs = client_with_admission(tmp_path, None)
    response = client.post("/v1/builds", json={"candidate": candidate().model_dump()})
    assert response.status_code == 202
    assert len(runs) == 1
