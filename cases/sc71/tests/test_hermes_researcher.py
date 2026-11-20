from pathlib import Path

from cases.sc71.harness.hermes import researcher
from evaluator import research


def test_case_launcher_only_selects_existing_profile(monkeypatch):
    arguments = []
    monkeypatch.setattr(researcher, "launch", lambda **kwargs: arguments.append(kwargs) or 0)
    assert researcher.main() == 0
    assert arguments == [{
        "model": researcher.runtime.MODEL,
        "parameters": {"reasoning_effort": "max"},
        "default_prompt": Path("/workspace/cases/sc71/campaigns/sc71-first-20261019/start.md"),
        "require_private": True,
    }]


def test_legacy_imports_share_the_common_guard():
    assert researcher.run_search is research.run_search
    assert researcher.continuation is research.continuation
    assert researcher.private_status is research.private_status
    assert researcher.ResearchError is research.ResearchError
