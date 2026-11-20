"""Operator preparation uses synthetic files only; no Docker or closed datasets."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import uvicorn
from evaluator.search import Journal

CASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def operatorctl():
    spec = importlib.util.spec_from_file_location(
        "ar03_operatorctl", CASE / "deploy/operatorctl.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_tree(tmp_path, operatorctl, monkeypatch, args):
    arena = tmp_path / "source"
    case = arena / "cases/ar-03"
    for name in ("evaluator", "common", "prompts"):
        (arena / name).mkdir(parents=True)
    for name in ("AGENTS.md", "README.md", "pyproject.toml", "uv.lock", ".gitignore"):
        (arena / name).write_text("synthetic open source")
    for name in (
        "client.py",
        "monitor.py",
        "provenance.json",
        ".gitignore",
        "ways/baseline/solve.py",
        "campaigns/pilot-01/start.md",
        "campaigns/campaign-01/metric/score.py",
    ):
        path = case / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic open code")
    exported = args.resources / "data/arena"
    exported.mkdir(parents=True)
    (exported / "train-500.jsonl").write_text('{"id":"open-page"}\n')
    (exported / "evaluation-ids.json").write_text('["open-page"]')
    (case / "_local").mkdir()
    (case / "_local/operator-secret").write_text("must not reach researcher")
    monkeypatch.setattr(operatorctl, "CASE", case)
    monkeypatch.setattr(operatorctl, "ARENA", arena)
    return case


@pytest.fixture
def args(tmp_path):
    return argparse.Namespace(
        root=tmp_path / "operator",
        mode="pilot",
        max_solutions=None,
        finalists=None,
        output_count=None,
        closed_source=None,
        port=8799,
        resources=tmp_path / "resources",
    )


def test_pilot_keeps_operator_state_outside_research_workspace(operatorctl, source_tree, args):
    operatorctl.prepare(args)
    config = json.loads((args.root / "config.json").read_text())
    workspace = args.root / "research-workspace"
    research_case = workspace / "cases/ar-03"
    assert config["mode"] == "pilot" and config["max_solutions"] == 4
    assert config["services"] == {"train": "http://127.0.0.1:8800"}
    assert config["evaluation_ids"] == ["open-page"]
    assert Path(config["assets"]) == args.resources / "runner-assets"
    assert Journal(args.root / "search/search.sqlite3").status()["proposed"] == 0
    assert not (args.root / "private/authorized.json").exists()
    assert (args.root / "private").stat().st_mode & 0o777 == 0o700
    assert not (workspace / "private").exists()
    assert not (research_case / "_local/operator-secret").exists()
    assert (research_case / "_local/train-500.jsonl").read_text() == ""
    assert (args.root / "private/data/train.jsonl").read_text() == (
        args.resources / "data/arena/train-500.jsonl"
    ).read_text()
    artifacts = json.loads((args.root / "private/artifacts.json").read_text())
    assert [source["split"] for source in artifacts["sources"]] == ["train"]
    assert Path(artifacts["output_root"]).is_relative_to(args.root / "private")


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_solutions": None},
        {"finalists": None},
        {"output_count": None},
        {"output_count": 3},
        {"closed_source": None},
    ],
)
def test_main_requires_agreed_conditions_before_creating_files(operatorctl, args, overrides):
    args.mode, args.max_solutions, args.finalists, args.output_count = "main", 5, 2, 1
    args.closed_source = Path("unused-closed-source")
    vars(args).update(overrides)
    with pytest.raises(ValueError, match="main"):
        operatorctl.prepare(args)
    assert not args.root.exists()


def test_prepare_preserves_existing_campaign(operatorctl, args):
    args.root.mkdir()
    config = args.root / "config.json"
    config.write_text('{"existing":"campaign"}')
    with pytest.raises(ValueError, match="already prepared"):
        operatorctl.prepare(args)
    assert config.read_text() == '{"existing":"campaign"}'


def test_moving_operator_directory_preserves_campaign_id(
    operatorctl, source_tree, args, monkeypatch
):
    operatorctl.prepare(args)
    original_id = args.root.name
    destination = args.root.with_name("relocated")
    args.root.rename(destination)
    args.root, args.role = destination, "train"
    configurations = []
    monkeypatch.setattr(operatorctl, "settings", lambda root, split: object())
    monkeypatch.setattr(operatorctl, "train_app", configurations.append)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    operatorctl.serve(args)
    assert configurations[0].campaign_id == original_id
