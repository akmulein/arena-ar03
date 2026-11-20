import json
import sys
from types import SimpleNamespace

import pytest

from integrations.hermes import researcher


@pytest.fixture
def campaign():
    return {
        "max_solutions": 100, "proposed": 11, "remaining_proposals": 89,
        "running": [], "completed_measurements": 21, "stopped": False,
        "finalization": None,
    }


@pytest.mark.parametrize("mismatch", [None, "finalists", "output_count", "release_id", "empty"])
def test_launcher_verifies_private_conditions_before_starting(
    monkeypatch, tmp_path, campaign, capsys, mismatch,
):
    prompt = tmp_path / "start.md"
    prompt.write_text("Synthetic campaign")
    monkeypatch.setattr(sys, "argv", [
        "researcher", "--model", "synthetic-model", "--prompt-file", str(prompt),
        "--journal", str(tmp_path / "search.sqlite"),
        "--private-url", "http://private", "--private-token-file", str(tmp_path / "token"),
        "--private-finalists", "10", "--private-output-count", "4",
        "--private-release-id", "synthetic-release",
    ])
    monkeypatch.setattr(researcher.runtime, "credentials", lambda path: {
        "model_base_url": "http://gateway/runs/synthetic/secret/v1",
    })
    monkeypatch.setattr(researcher.runtime, "prepare_home", lambda path, **kwargs: None)
    monkeypatch.setattr(
        researcher.runtime, "install", lambda *args, **kwargs: SimpleNamespace(check=lambda: None),
    )
    monkeypatch.setattr(
        researcher, "Journal", lambda path: SimpleNamespace(status=lambda: campaign),
    )
    requests = []
    private_requests = []

    def readiness(*args, **kwargs):
        requests.append((args, kwargs))
        return {
            "mode": "campaign", "gate_enabled": True, "journal_ready": True, "max_solutions": 100,
            "finalists": 9 if mismatch == "finalists" else 10,
            "output_count": 3 if mismatch == "output_count" else 4,
            "evaluation_enabled": mismatch != "empty",
            "release_id": "other-release" if mismatch == "release_id" else "synthetic-release",
        }

    def private(*args):
        private_requests.append(args)
        return "complete"

    monkeypatch.setattr("evaluator.client.request", readiness)
    monkeypatch.setattr(researcher, "private_status", private)

    def forbidden(**kwargs):
        raise AssertionError("must not create Hermes")

    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=forbidden))
    assert researcher.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert requests == [(
        ("GET", "http://private", tmp_path / "token"),
        {"timeout": 60, "resource": "search-readiness"},
    )]
    assert bool(private_requests) is (mismatch is None)
    assert json.loads(captured.err) == {
        "status": "incomplete",
        "reason": (
            "private search admission is unavailable or mismatched; search was not continued"
            if mismatch else "private attempt already exists before valid search finalization"
        ),
    }


@pytest.mark.parametrize("missing", [
    "--private-finalists", "--private-output-count", "--private-release-id",
])
def test_launcher_requires_expected_private_conditions(monkeypatch, tmp_path, capsys, missing):
    arguments = [
        "researcher", "--model", "synthetic-model", "--journal", str(tmp_path / "search.sqlite"),
        "--private-url", "http://private", "--private-token-file", str(tmp_path / "token"),
    ]
    for flag, value in (
        ("--private-finalists", "10"), ("--private-output-count", "4"),
        ("--private-release-id", "synthetic"),
    ):
        if flag != missing:
            arguments.extend([flag, value])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        researcher.main()
    assert error.value.code == 2
    assert missing in capsys.readouterr().err




def test_launcher_requires_explicit_private_mode(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [
        "researcher", "--model", "arbitrary-internal-model", "--journal", str(tmp_path / "journal"),
    ])
    with pytest.raises(SystemExit) as error:
        researcher.main()
    assert error.value.code == 2
    assert "--private-url" in capsys.readouterr().err


def test_empty_private_rejects_leftover_private_conditions(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [
        "researcher", "--model", "arbitrary-internal-model", "--journal", str(tmp_path / "journal"),
        "--no-private", "--private-finalists", "2",
    ])
    with pytest.raises(SystemExit) as error:
        researcher.main()
    assert error.value.code == 2
    assert "--no-private" in capsys.readouterr().err


def test_empty_private_generic_launcher_uses_another_model(monkeypatch, tmp_path, capsys):
    prompt = tmp_path / "start.md"
    prompt.write_text("An unrelated synthetic campaign")
    config = tmp_path / "profile.yaml"
    credentials = []
    models = []
    setup = []
    closes = []
    campaign = {
        "max_solutions": 1, "proposed": 1, "remaining_proposals": 0,
        "running": [], "stopped": False, "finalization": None,
    }

    def access(path):
        credentials.append(path)
        return {"model_base_url": "http://gateway/runs/synthetic/secret/v1",
                "model": "another-internal-model", "parameters": {"temperature": 0}}

    class Agent:
        def __init__(self, **kwargs):
            models.append(kwargs)

        def run_conversation(self, prompt, **kwargs):
            assert prompt == "An unrelated synthetic campaign"
            assert "DeepSeek" not in kwargs["system_message"]
            campaign["finalization"] = {"batch_digest": "a" * 64}
            return {"completed": True, "final_response": "selection saved"}

        def close(self):
            closes.append(True)

    monkeypatch.setattr(researcher.runtime, "credentials", access)
    monkeypatch.setattr(
        researcher.runtime, "prepare_home", lambda home, **kwargs: setup.append(kwargs),
    )
    monkeypatch.setattr(researcher.runtime, "install", lambda *args, **kwargs: (
        setup.append(kwargs) or SimpleNamespace(check=lambda: None)
    ))
    monkeypatch.setattr(
        researcher, "Journal", lambda path: SimpleNamespace(status=lambda: campaign),
    )
    monkeypatch.setattr(researcher, "private_status", lambda *args: pytest.fail("private disabled"))
    monkeypatch.setattr(
        researcher, "search_readiness", lambda *args: pytest.fail("private disabled"),
    )
    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=Agent))
    result = researcher.main([
        "--model", "another-internal-model", "--prompt-file", str(prompt),
        "--config-file", str(config), "--journal", str(tmp_path / "journal"), "--no-private",
    ])
    assert result == 0
    assert models[0]["model"] == "another-internal-model"
    assert "reasoning_config" not in models[0]
    assert setup[0]["config_path"] == config
    assert all(item["parameters"] == {"temperature": 0} for item in setup)
    assert closes == [True]
    assert json.loads(capsys.readouterr().out) == {"answer": "selection saved"}


def test_generic_launcher_imports_without_case_packages():
    import subprocess

    program = '''
import builtins
original = builtins.__import__
def import_module(name, *args, **kwargs):
    if name == "cases" or name.startswith("cases."):
        raise AssertionError("generic researcher must not import a case")
    return original(name, *args, **kwargs)
builtins.__import__ = import_module
from evaluator.research import continuation
from integrations.hermes.researcher import main
assert callable(main)
assert "fresh context" in continuation({
    "remaining_proposals": 1, "proposed": 0, "max_solutions": 1,
    "stopped": False, "running": [],
}, private_required=False)
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_launcher_rejects_parameters_changed_from_credentials(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(researcher.runtime, "credentials", lambda path: {
        "model": "another-internal-model", "model_base_url": "http://internal/v1",
        "parameters": {"temperature": 0},
    })
    monkeypatch.setattr(researcher.runtime, "prepare_home", lambda *args, **kwargs: pytest.fail(
        "a changed fixed parameter must be rejected before preparing Hermes"
    ))
    prompt = tmp_path / "start.md"
    prompt.write_text("synthetic")
    assert researcher.main([
        "--model", "another-internal-model", "--parameters-json", '{"temperature": 1}',
        "--prompt-file", str(prompt), "--journal", str(tmp_path / "journal"), "--no-private",
    ]) == 1
    assert json.loads(capsys.readouterr().err) == {
        "status": "incomplete", "reason": "Hermes researcher failed",
    }
