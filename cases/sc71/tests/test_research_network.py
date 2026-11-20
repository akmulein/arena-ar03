import json
import subprocess

import pytest

from cases.sc71.deploy.gpu6 import network_check


@pytest.mark.parametrize("web_status", [0, 1])
def test_network_acceptance_checks_hermes_search(monkeypatch, capsys, web_status):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "hermes-image" in command:
            return subprocess.CompletedProcess(command, web_status, "{}", "")
        return subprocess.CompletedProcess(command, 0, "[]", "")

    monkeypatch.setenv("ARENA_EVALUATOR_IMAGE", "evaluator-image")
    monkeypatch.setenv("ARENA_HERMES_IMAGE", "hermes-image")
    monkeypatch.setattr(network_check.subprocess, "run", run)

    assert network_check.main() == web_status
    web_command = next(command for command in commands if "hermes-image" in command)
    assert "DDGS_PROXY=http://internet-proxy:3128" in web_command
    assert "integrations.hermes.network" in web_command
    assert "--entrypoint" in web_command
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["hermes_web_search"] == (
        web_status == 0
    )
