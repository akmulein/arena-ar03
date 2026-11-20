"""New operator/research API connections only; synthetic data and no models."""

import base64
import importlib.util
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from evaluator.client import pack_candidate
from evaluator.reliability import empty_usage
from evaluator.runner import SolveResult
from evaluator.search import Journal
from evaluator.service import Settings
from evaluator.service import create_app as closed_app
from evaluator.train import TrainSettings
from evaluator.train import create_app as train_app
from fastapi.testclient import TestClient

CASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def control():
    spec = importlib.util.spec_from_file_location("ar03_control", CASE / "deploy/control.py")
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    return control


@pytest.fixture
def private(tmp_path):
    path = tmp_path / "private"
    path.mkdir()
    return path


@pytest.fixture
def journal(tmp_path):
    (tmp_path / "search").mkdir()
    return Journal.create(tmp_path / "search/search.sqlite3", max_solutions=1)


@pytest.fixture
def metric(private):
    scorer = private / "synthetic-score.py"
    scorer.write_text('import json; print(json.dumps({"score":0.625}))')
    metric = private / "synthetic-metric.json"
    metric.write_text(
        json.dumps(
            {
                "kind": "command",
                "direction": "maximize",
                "version": "wire",
                "command": {"argv": [sys.executable, str(scorer)], "uses_model": False},
            }
        )
    )
    return metric


@pytest.fixture
def datasets(private):
    ok, image = cv2.imencode(".png", np.full((8, 8, 3), 255, dtype=np.uint8))
    assert ok
    truth = {
        "tables": [
            {
                "bbox": [0, 0, 8, 8],
                "rows": 1,
                "cols": 1,
                "cells": [
                    {
                        "bbox": [0, 0, 8, 8],
                        "r0": 0,
                        "r1": 0,
                        "c0": 0,
                        "c1": 0,
                        "text": "SYNTHETIC-CLOSED-CANARY",
                    }
                ],
            }
        ]
    }
    datasets = {}
    for split in ("train", "public", "private"):
        path = private / f"{split}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "synthetic-" + split,
                    "input": json.dumps(
                        {
                            "version": "ar03-page-v1",
                            "image_base64": base64.b64encode(image).decode(),
                        }
                    ),
                    "reference": json.dumps(truth),
                }
            )
            + "\n"
        )
        datasets[split] = path
    return datasets


@pytest.fixture
def operator_root(tmp_path, private, datasets):
    (tmp_path / "research-token").write_text("research-token")
    for split in ("admin", "train", "public", "private"):
        (private / f"{split}-token").write_text(split + "-token")
    config = {
        "monitor": str(CASE / "monitor.py"),
        "dataset_id": "synthetic",
        "conditions_id": "wire-v1",
        "finalists": 1,
        "train_ids": ["synthetic-train"],
        "evaluation_ids": ["synthetic-train"],
        "services": {split: "http://" + split + ".test" for split in datasets},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def proposal(tmp_path):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "solve.py").write_text("raise RuntimeError('must never execute on host')")
    return {
        "candidate": pack_candidate("proposal", candidate_dir).model_dump(),
        "hypothesis": "synthetic connection",
        "change": "wiring only",
    }


@pytest.fixture
def evaluator_clients(operator_root, private, datasets, metric, journal):
    answer = json.dumps({"prediction": {"tables": []}, "detections": [], "error": None})

    def fake_solve(path, text):
        return SolveResult(answer, empty_usage(), 0.001)

    with ExitStack() as stack:
        clients = {}
        for split in datasets:
            settings = Settings(
                dataset_path=datasets[split],
                metric_path=metric,
                state_dir=private / "state" / split,
                token=split + "-token",
                runner_image="unused-synthetic",
                uses_model=False,
                release_id="wire-v1",
                finalists=1,
                output_count=1,
                split="public" if split == "public" else "private",
                search_mode="campaign",
                search_journal=journal.path,
                search_max_solutions=1,
            )
            app = (
                train_app(
                    TrainSettings(
                        settings, "wire", "ar-03", "synthetic", "wire-v1", tracking=False
                    ),
                    solve=fake_solve,
                )
                if split == "train"
                else closed_app(settings, solve=fake_solve)
            )
            clients[split] = stack.enter_context(TestClient(app))
        yield clients


@pytest.fixture
def api(
    control, evaluator_clients, operator_root, private, datasets, journal, proposal, monkeypatch
):
    constructor = httpx.Client

    def forward(request):
        split = request.url.host.split(".")[0]
        return evaluator_clients[split].request(
            request.method,
            request.url.raw_path.decode(),
            content=request.content,
            headers=request.headers,
        )

    def connection(**kwargs):
        return constructor(transport=httpx.MockTransport(forward), **kwargs)

    monkeypatch.setattr(control.httpx, "Client", connection)
    with TestClient(control.create_app(operator_root)) as broker:
        headers = {"Authorization": "Bearer research-token"}
        yield {
            "broker": broker,
            "headers": headers,
            "proposal": proposal,
            "root": operator_root,
            "private": private,
            "datasets": datasets,
            "journal": journal,
        }


@pytest.fixture
def registered(api):
    (api["private"] / "authorized.json").write_text('{"synthetic_check":true}')
    response = api["broker"].post("/propose", headers=api["headers"], json=api["proposal"])
    assert response.status_code == 200, response.text
    api["candidate_id"] = response.json()["candidate_id"]
    return api


def evaluate(api, split):
    response = api["broker"].post(
        "/evaluate",
        headers=api["headers"],
        json={"candidate_id": api["candidate_id"], "split": split},
    )
    assert response.status_code == 200, response.text
    measurement = response.json()["measurement"]
    deadline = time.monotonic() + 10
    while True:
        result = api["broker"].get("/result/" + measurement, headers=api["headers"])
        assert result.status_code == 200, result.text
        if result.json()["status"] not in ("RUNNING", "running"):
            return result
        assert time.monotonic() < deadline
        time.sleep(0.02)


def finalize(api):
    response = api["broker"].post(
        "/finalize", headers=api["headers"], json={"candidate_ids": [api["candidate_id"]]}
    )
    assert response.status_code == 200, response.text
    return api["broker"].get("/result/private", headers=api["headers"])


def test_submission_requires_operator_authorization(api):
    response = api["broker"].post("/propose", headers=api["headers"], json=api["proposal"])
    assert response.status_code == 423
    assert api["journal"].status()["proposed"] == 0


def test_researcher_cannot_supply_score(registered):
    api = registered
    assert (
        api["broker"]
        .post(
            "/evaluate",
            headers=api["headers"],
            json={"candidate_id": api["candidate_id"], "split": "train", "score": 1},
        )
        .status_code
        == 422
    )


def test_researcher_cannot_finish_measurement(registered):
    api = registered
    assert (
        api["broker"].post("/finish", headers=api["headers"], json={"score": 1}).status_code == 404
    )


def test_researcher_cannot_issue_operator_stop(registered):
    api = registered
    assert (
        api["broker"]
        .post("/admin/stop", headers=api["headers"], json={"reason": "forged"})
        .status_code
        == 401
    )


def test_train_public_results_and_progress_use_trusted_scores(registered):
    api = registered
    evaluate(api, "train")
    result = evaluate(api, "public")
    assert result.json() == {"status": "complete", "score": 0.625}
    assert "SYNTHETIC-CLOSED-CANARY" not in result.text
    assert [m["score"] for m in api["journal"].export()["measurements"]] == [0.625, 0.625]
    assert (api["root"] / "progress/teds.svg").is_file()
    assert (api["root"] / "progress/lineage.svg").is_file()


@pytest.fixture
def finalized(registered):
    evaluate(registered, "public")
    finalize(registered)
    return registered


def test_private_feedback_contains_only_status(finalized):
    api = finalized
    result = api["broker"].get("/result/private", headers=api["headers"])
    assert result.status_code == 200 and set(result.json()) == {"status"}
    assert "SYNTHETIC-CLOSED-CANARY" not in result.text


def test_operator_visualizations_use_saved_predictions(finalized):
    save_visualizations(finalized)


def test_human_stop_preserves_finalization(registered):
    api = registered
    response = api["broker"].post(
        "/admin/stop",
        headers={"Authorization": "Bearer admin-token"},
        json={"reason": "synthetic stop", "source_ref": "synthetic human instruction"},
    )
    assert response.status_code == 200, response.text
    assert not (api["private"] / "authorized.json").exists()
    result = finalize(api)
    assert result.status_code == 200 and set(result.json()) == {"status"}


def save_visualizations(api):
    private, datasets, identifier = api["private"], api["datasets"], api["candidate_id"]
    spec = importlib.util.spec_from_file_location(
        "ar03_operator_artifacts", CASE / "operator_artifacts.py"
    )
    artifacts = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = artifacts
    spec.loader.exec_module(artifacts)
    source_config = {
        "output_root": str(private / "visualizations"),
        "metric_runtime": str(CASE / "campaigns/campaign-01/metric/runtime"),
        "sources": [
            {
                "split": split,
                "state_dir": str(private / "state" / split),
                "dataset_path": str(datasets[split]),
            }
            for split in datasets
        ],
    }
    path = private / "artifacts.json"
    path.write_text(json.dumps(source_config))
    consumer = artifacts.ArtifactConsumer(artifacts.Configuration.load(path))
    consumer.scan()
    assert consumer.errors == 0
    for split in ("public", "private"):
        assert (
            private
            / "visualizations"
            / split
            / identifier
            / f"synthetic-{split}"
            / "evaluation.json"
        ).is_file()
