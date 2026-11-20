import json

import pytest

from integrations.hermes import runtime


def test_campaign_model_and_parameters_are_used_without_case_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.os, "environ", {})
    runtime.prepare_home(tmp_path, model="other/internal-model", parameters={"temperature": 0.25})
    config = (tmp_path / "config.yaml").read_text()
    assert 'default: "other/internal-model"' in config
    assert "deepseek" not in config.lower()
    assert runtime.os.environ["OPENAI_MODEL"] == "other/internal-model"
    request = runtime.gateway_request(
        {"model": "caller-cannot-override", "messages": [{"role": "user", "content": "x"}],
         "temperature": 0.9, "reasoning_effort": "max"},
        model="other/internal-model", parameters={"temperature": 0.25},
    )
    assert request == {
        "model": "other/internal-model", "messages": [{"role": "user", "content": "x"}],
        "stream": False, "temperature": 0.25,
    }


def test_profile_model_is_safely_encoded_in_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.os, "environ", {})
    model = 'synthetic"\nprofile'
    runtime.prepare_home(tmp_path, model=model, parameters={})
    assert 'default: ' + json.dumps(model) in (tmp_path / "config.yaml").read_text()
    assert "@MODEL@" not in (tmp_path / "config.yaml").read_text()


@pytest.mark.parametrize("model", ["", " "])
def test_empty_campaign_model_is_rejected(tmp_path, model):
    with pytest.raises(ValueError, match="model"):
        runtime.prepare_home(tmp_path, model=model)


def test_generic_request_preserves_completed_tool_history_and_profile():
    history = [
        {"role": "assistant", "content": None, "reasoning_content": "reason",
         "tool_calls": [{"id": "tool-1", "type": "function", "index": 0,
                         "function": {"name": "lookup", "arguments": "{}", "parsed": {}}}]},
        {"role": "tool", "tool_call_id": "tool-1", "content": "evidence", "timestamp": 1},
    ]
    request = runtime.gateway_request(
        {"messages": [], "extra_body": {"messages": history, "model": "ignored"}},
        model="second/internal-model", parameters={"reasoning_effort": "low"},
    )
    assert request["model"] == "second/internal-model"
    assert request["reasoning_effort"] == "low"
    assert request["messages"][0]["reasoning_content"] == "reason"
    assert request["messages"][0]["tool_calls"] == [{
        "id": "tool-1", "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }]
    assert request["messages"][1] == {
        "role": "tool", "tool_call_id": "tool-1", "content": "evidence",
    }
    assert history[1]["timestamp"] == 1


@pytest.mark.parametrize("key", ["model", "messages", "stream", "tools", "tool_choice"])
def test_profile_parameters_cannot_override_request_envelope(tmp_path, key):
    with pytest.raises(ValueError, match="model parameters"):
        runtime.prepare_home(tmp_path, model="synthetic", parameters={key: "override"})
    assert not (tmp_path / "config.yaml").exists()
