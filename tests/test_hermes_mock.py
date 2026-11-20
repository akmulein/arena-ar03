import ast
from pathlib import Path

from integrations.hermes import mock_check


def test_mock_uses_selected_model(monkeypatch, tmp_path):
    monkeypatch.setattr(mock_check, "effective_settings", lambda: {})
    model = "alternate-synthetic-model"
    gateway = mock_check.MockGateway("failure", tmp_path / "fact.md", model=model)
    request = {
        "model": model,
        "reasoning_effort": "max",
        "stream": False,
        "messages": [{"role": "user", "content": "Synthetic request"}],
    }
    status, _ = gateway.respond(request)
    assert status == 500
    result = mock_check.verify(gateway, None, True, tmp_path, {})
    assert result["passed"]
    request["model"] = "unexpected-model"
    result = mock_check.verify(gateway, None, True, tmp_path, {})
    assert not result["checks"]["fixed_model_parameters"]
    assert not result["passed"]


def test_mock_has_no_case_imports():
    tree = ast.parse(Path(mock_check.__file__).read_text())
    imports = [
        name
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names] if isinstance(node, ast.Import)
            else [node.module or ""] if isinstance(node, ast.ImportFrom)
            else []
        )
    ]
    assert not any(name == "cases" or name.startswith("cases.") for name in imports)
