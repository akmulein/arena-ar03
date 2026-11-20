"""Research CLI contract without network or credentials."""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

CASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    spec = importlib.util.spec_from_file_location("ar03_research_client", CASE / "client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def transport(client, monkeypatch):
    requests, replies = [], [{"status": "complete", "score": 0.625}]
    constructor = httpx.Client
    original_read = Path.read_text

    def read_token(path, *args, **kwargs):
        if str(path) == "/run/secrets/submit-token":
            return "synthetic-token"
        return original_read(path, *args, **kwargs)

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=replies.pop(0))

    def connection(**kwargs):
        assert kwargs["trust_env"] is False
        return constructor(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(Path, "read_text", read_token)
    monkeypatch.setattr(client.httpx, "Client", connection)
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("AR03_API", "http://operator.test")
    return requests, replies


@pytest.mark.parametrize(
    ("arguments", "method", "route", "body"),
    [
        (["status"], "GET", "/status", None),
        (["progress"], "GET", "/progress", None),
        (["result", "--measurement", "check-id"], "GET", "/result/check-id", None),
        (
            ["evaluate", "--candidate", "S01", "--example-id", "open-page"],
            "POST",
            "/evaluate",
            {"candidate_id": "S01", "split": "train", "example_ids": ["open-page"]},
        ),
        (
            ["evaluate", "--candidate", "S01", "--split", "public"],
            "POST",
            "/evaluate",
            {"candidate_id": "S01", "split": "public"},
        ),
        (["finalize", "--candidate", "S01"], "POST", "/finalize", {"candidate_ids": ["S01"]}),
    ],
)
def test_commands_keep_http_contract(
    client, transport, monkeypatch, capsys, arguments, method, route, body
):
    monkeypatch.setattr(sys, "argv", ["client.py", *arguments])
    client.main()
    request = transport[0][0]
    assert request.method == method and request.url.path == route
    assert request.headers["Authorization"] == "Bearer synthetic-token"
    assert (json.loads(request.content) if request.content else None) == body
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "score": 0.625}


def test_proposal_sends_snapshot_and_lineage(client, transport, tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "solve.py").write_text("raise RuntimeError('not executed')")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client.py",
            "propose",
            "--directory",
            str(candidate),
            "--parent",
            "S01",
            "--hypothesis",
            "geometry",
            "--change",
            "synthetic",
            "--stage",
            "ocr",
        ],
    )
    client.main()
    request = transport[0][0]
    body = json.loads(request.content)
    assert request.method == "POST" and request.url.path == "/propose"
    assert body["candidate"]["id"] == "proposal"
    assert body["parents"] == ["S01"] and body["stage"] == "ocr"
    assert body["hypothesis"] == "geometry" and body["change"] == "synthetic"


def test_wait_saves_full_result_and_prints_json_summary(
    client, transport, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "results/train.json"
    result = {"status": "FINISHED", "metrics": {"score": 0.625}, "records": ["open-page"]}
    transport[1][:] = [{"status": "RUNNING"}, result]
    monkeypatch.setattr(
        sys,
        "argv",
        ["client.py", "result", "--measurement", "check-id", "--wait", "--output", str(output)],
    )
    client.main()
    assert len(transport[0]) == 2
    assert json.loads(output.read_text()) == result
    assert json.loads(capsys.readouterr().out) == {
        "status": "FINISHED",
        "saved": str(output),
        "score": 0.625,
    }
