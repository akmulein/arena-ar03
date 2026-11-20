import json
import subprocess
import sys
from types import ModuleType

import pytest

from integrations.hermes import network


@pytest.mark.parametrize("proxy", [None, "http://research-egress:3128"])
def test_search_container_uses_selected_environment_without_case_profile(proxy):
    command = network.container_command("custom-hermes", "custom-network", "internal-custom", proxy)
    assert command[command.index("--network") + 1] == "custom-network"
    assert command[-5:] == [
        "-m", "integrations.hermes.network", "--model", "internal-custom", "--probe",
    ]
    assert "custom-hermes" in command
    assert not any("cases." in value or "sc71" in value for value in command)
    if proxy:
        assert f"DDGS_PROXY={proxy}" in command
    else:
        assert not any("PROXY=" in value for value in command)


@pytest.mark.parametrize("hits,success", [([], False), ([{"title": "Reference"}], True)])
def test_web_probe_uses_supplied_model_and_requires_search_results(
    tmp_path, monkeypatch, capsys, hits, success,
):
    prepared = []
    runtime = ModuleType("integrations.hermes.runtime")
    monkeypatch.setattr(runtime, "prepare_home", lambda home, **kwargs: prepared.append(
        (home, kwargs)
    ), raising=False)
    web_tools = ModuleType("tools.web_tools")
    monkeypatch.setattr(web_tools, "web_search_tool", lambda query: json.dumps({
        "success": True, "data": {"web": hits},
    }), raising=False)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setitem(sys.modules, web_tools.__name__, web_tools)

    assert network.main([
        "--model", "internal-custom", "--home", str(tmp_path), "--probe",
    ]) == (0 if success else 1)
    assert prepared == [(tmp_path, {"model": "internal-custom"})]
    assert json.loads(capsys.readouterr().out)["success"] is success


def test_network_command_runs_selected_image_and_reports_failure(monkeypatch, capsys):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, '{"success":false}', "")

    monkeypatch.setattr(network.subprocess, "run", run)
    assert network.main([
        "--model", "custom-model", "--image", "custom-image", "--network", "research",
    ]) == 1
    assert commands[0] == network.container_command("custom-image", "research", "custom-model")
    assert json.loads(capsys.readouterr().out) == {"success": False}
