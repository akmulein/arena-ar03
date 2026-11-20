"""Publish permitted campaign sources/history using Arena export and restore."""

import base64
import importlib.util
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from evaluator.search import Journal

CASE = Path(__file__).resolve().parents[1]
TEXT_FILES = {".py", ".json", ".md", ".toml", ".txt", ".yaml", ".yml", ".sh"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def permitted_history(history):
    """Allowlist journal fields; do not publish evidence, jobs, endpoints or raw errors."""
    fields = (
        "measurement_id",
        "candidate_id",
        "digest",
        "comparison",
        "state",
        "score",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "completion_order",
        "proposal_count_at_finish",
    )
    result = {key: history[key] for key in ("schema_version", "campaign", "candidates")}
    result["research_usage"] = None
    result["measurements"] = []
    for measurement in history["measurements"]:
        split = measurement["comparison"]["split"]
        if split not in {"train", "public"}:
            raise ValueError("only train/public journal measurements can be published")
        row = {key: measurement[key] for key in fields if key in measurement}
        if split == "train":
            row.update({key: measurement.get(key) for key in ("solver", "judge")})
        row["error"] = "evaluation_failed" if measurement.get("error") else None
        result["measurements"].append(row)
    return result


def copy_sources(source, destination, *, markdown_only=False):
    """Only explicit researcher source trees; never follow links or copy hidden state."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError("publication source must be a regular source directory")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(
            part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise ValueError("publication source contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix not in TEXT_FILES:
            raise ValueError("publication source contains a non-source file")
        if any(
            part.startswith(".") or "token" in part.lower() or "secret" in part.lower()
            for part in relative.parts
        ):
            raise ValueError("publication source contains a restricted path")
        if markdown_only and path.suffix != ".md":
            raise ValueError("only Markdown research notes are published")
        path.read_text(encoding="utf-8")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)


def publication_secrets(root):
    values = []
    for path in [root / "research-token", *(root / "private").glob("*-token")]:
        if path.is_file():
            value = path.read_bytes().strip()
            if value:
                values.extend((value, base64.b64encode(value)))
    return values


def check_publication(directory, secrets):
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("publication must not contain symlinks")
        if path.is_file():
            relative = path.relative_to(directory)
            if path.suffix not in TEXT_FILES | {".svg"} or any(
                part.startswith(".") for part in relative.parts
            ):
                raise ValueError("publication contains a restricted file")
            contents = path.read_bytes()
            contents.decode("utf-8")
            if any(value in contents for value in secrets):
                raise ValueError("publication contains an operator or submit token")


def publish_graphs(journal, history, destination):
    spec = importlib.util.spec_from_file_location("ar03_publication_monitor", CASE / "monitor.py")
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    comparisons = []
    for row in history["measurements"]:
        if row["comparison"] not in comparisons:
            comparisons.append(row["comparison"])
    for index, comparison in enumerate(comparisons, start=1):
        result = monitor.snapshot(journal, comparison, stage="end-to-end")
        monitor.save_artifacts(result, destination / f"{index:02d}-{comparison['split']}")


def export_campaign(args):
    root = args.root.resolve()
    output = (args.output or CASE / "campaigns" / root.name / "publication").resolve()
    if not output.is_relative_to(CASE) or output.exists():
        raise ValueError("publication requires a new destination inside cases/ar-03")
    config = json.loads((root / "config.json").read_text())
    journal = Journal(root / "search/search.sqlite3")
    history = journal.export()
    state = history["campaign"]
    if state["running"] or (root / "private/security/permission.json").exists():
        raise ValueError("pause the campaign and finish accepted evaluations before export")
    if not state["stopped"] and not state["finalization"] and state["remaining_proposals"]:
        raise ValueError("only completed or explicitly stopped campaigns can be published")
    workspace = root / "research-workspace/cases/ar-03"
    with tempfile.TemporaryDirectory(dir=root / "private") as temporary:
        staged = Path(temporary) / "publication"
        staged.mkdir()
        write_json(staged / "history.json", permitted_history(history))
        write_json(
            staged / "conditions.json",
            {
                key: config[key]
                for key in (
                    "campaign_id",
                    "mode",
                    "max_solutions",
                    "finalists",
                    "output_count",
                    "dataset_id",
                    "conditions_id",
                )
            },
        )
        for candidate in history["candidates"]:
            identifier = candidate["solution_id"]
            if not re.fullmatch(r"S\d{2,}", identifier):
                raise ValueError("invalid canonical solution ID")
            if not candidate.get("snapshot_error"):
                journal.restore(candidate["candidate_id"], staged / "ways" / identifier)
        drafts = workspace / "ways"
        if drafts.exists():
            copy_sources(drafts, staged / "drafts")
        for board in sorted((workspace / "campaigns").glob("*/board")):
            copy_sources(
                board, staged / "research" / board.parent.name / "board", markdown_only=True
            )
        if args.skill is not None:
            skill = args.skill.resolve()
            if not skill.is_relative_to(workspace) or not (skill / "SKILL.md").is_file():
                raise ValueError("Skill must already exist in the permitted researcher workspace")
            copy_sources(skill, staged / "skill")
        metrics = {}
        for row in history["measurements"]:
            identifier = row["measurement_id"]
            if row["comparison"]["split"] != "train" or not re.fullmatch(
                r"check-[a-f0-9]{32}", identifier
            ):
                continue
            result = root / "private/results" / f"{identifier}.json"
            if result.is_file():
                values = json.loads(result.read_text()).get("metrics", {})
                metrics[identifier] = {
                    key: value
                    for key, value in values.items()
                    if type(value) in (int, float) and math.isfinite(value)
                }
        write_json(staged / "train-metrics.json", metrics)
        publish_graphs(journal, history, staged / "graphs")
        check_publication(staged, publication_secrets(root))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), output)
    print(json.dumps({"exported": True, "candidates": len(history["candidates"])}))
