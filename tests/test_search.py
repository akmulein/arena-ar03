import json
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from evaluator.client import pack_candidate
from evaluator.contracts import Submission
from evaluator.search import Journal, Resources, SearchError, main

ZERO = Resources(total_tokens=0, seconds=0)


def propose(journal: Journal, directory: Path, identifier: str, parents: tuple[str, ...] = ()):
    return journal.propose(
        identifier,
        directory,
        parents=parents,
        hypothesis="A general mechanism improves the task metric",
        change="A recorded change to the solution",
        resources=ZERO,
    )


def begin(journal: Journal, identifier: str, candidate: str, **overrides):
    arguments = {
        "split": "train",
        "dataset_id": "dataset-v1",
        "subset_id": "all",
        "conditions_id": "metric-v1-fixed-conditions",
    }
    arguments.update(overrides)
    return journal.begin(identifier, candidate, **arguments)


def finish(journal: Journal, identifier: str, score: float):
    return journal.finish(identifier, score=score, solver=ZERO, judge=ZERO)


def curve(journal: Journal, **overrides):
    arguments: dict[str, Any] = {
        "split": "train",
        "dataset_id": "dataset-v1",
        "subset_id": "all",
        "conditions_id": "metric-v1-fixed-conditions",
        "direction": "minimize",
    }
    arguments.update(overrides)
    return journal.curve(**arguments)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "solve.py").write_text("print(42)\n", encoding="utf-8")
    (directory / "solve.py").chmod(0o755)
    return directory


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal.create(tmp_path / "search.sqlite", max_solutions=10)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit(tmp_path: Path, limit):
    with pytest.raises(SearchError, match="positive integer"):
        Journal.create(tmp_path / "invalid.sqlite", max_solutions=limit)


def test_limit_retains_failed_parents_and_allows_last_check(tmp_path: Path, source: Path):
    journal = Journal.create(tmp_path / "search.sqlite", max_solutions=2)
    propose(journal, source, "failed")
    begin(journal, "check-1", "failed")
    journal.finish("check-1", error="execution failed", solver=ZERO, judge=ZERO)
    child = propose(journal, source, "child", ("failed",))
    assert child["proposal"] == 2
    assert child["parents"] == ["failed"]
    begin(journal, "check-2", "child")
    finish(journal, "check-2", 3)
    with pytest.raises(SearchError, match="limit"):
        propose(journal, source, "excess")
    assert journal.status()["proposed"] == 2
    assert journal.status()["stopped"] is False
    assert len(journal.export()["candidates"]) == 2


def _race_proposal(database: str, source: str, identifier: str) -> bool:
    try:
        propose(Journal(Path(database)), Path(source), identifier)
        return True
    except SearchError:
        return False


def test_processes_cannot_overfill_limit(tmp_path: Path, source: Path):
    path = tmp_path / "race.sqlite"
    Journal.create(path, max_solutions=1)
    with ProcessPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(
            _race_proposal, [str(path)] * 4, [str(source)] * 4, ["a", "b", "c", "d"]
        ))
    assert outcomes.count(True) == 1
    assert Journal(path).status()["proposed"] == 1


def test_stop_drains_running_only_and_persists(journal: Journal, source: Path):
    propose(journal, source, "running")
    propose(journal, source, "queued")
    begin(journal, "started", "running")
    stopped = journal.stop("human requested selection")
    assert stopped["running"] == ["started"]
    reopened = Journal(journal.path)
    with pytest.raises(SearchError, match="stopped"):
        begin(reopened, "new", "queued")
    with pytest.raises(SearchError, match="stopped"):
        propose(reopened, source, "new")
    finish(reopened, "started", 1)
    assert reopened.status()["running"] == []
    assert reopened.status()["stop_reason"] == "human requested selection"
    assert reopened.stop("another reason")["stop_reason"] == "human requested selection"


def test_finalization_requires_complete_search_and_drained_checks(tmp_path: Path, source: Path):
    journal = Journal.create(tmp_path / "search.sqlite", max_solutions=2)
    propose(journal, source, "one")
    batch = Submission(candidates=[pack_candidate("one", source)])
    with pytest.raises(SearchError, match="proposal limit"):
        journal.finalize(batch)
    propose(journal, source, "two")
    begin(journal, "running", "two")
    with pytest.raises(SearchError, match="running"):
        journal.finalize(batch)
    finish(journal, "running", 3)
    sealed = journal.finalize(batch)
    assert sealed["batch_digest"] == batch.digest()
    assert Journal(journal.path).finalize(batch) == sealed
    assert journal.status()["finalization"] == sealed
    with pytest.raises(SearchError, match="finalized"):
        begin(journal, "late", "one")
    with pytest.raises(SearchError, match="finalized"):
        propose(journal, source, "late")
    with pytest.raises(SearchError, match="different"):
        journal.finalize(Submission(candidates=[pack_candidate("two", source)]))


def test_finalist_reselection_preserves_history_and_does_not_reopen_search(
    tmp_path: Path, source: Path,
):
    journal = Journal.create(tmp_path / "search.sqlite", max_solutions=2)
    propose(journal, source, "one")
    propose(journal, source, "two")
    first = Submission(candidates=[pack_candidate("one", source)])
    second = Submission(candidates=[pack_candidate("two", source)])
    initial = journal.finalize(first)
    with pytest.raises(SearchError, match="different"):
        journal.finalize(second)
    selected = journal.finalize(second, replace=True)
    assert selected["batch_digest"] == second.digest()
    assert selected["previous"] == [initial]
    assert journal.finalize(second, replace=True) == selected
    with pytest.raises(SearchError, match="finalized"):
        begin(journal, "late", "one")
    with pytest.raises(SearchError, match="finalized"):
        propose(journal, source, "late")
    reverted = journal.finalize(first, replace=True)
    assert reverted["previous"] == [
        initial, {key: value for key, value in selected.items() if key != "previous"},
    ]
    assert journal.status()["finalization"] == reverted


def test_finalization_binds_registered_candidate_contents(journal: Journal, source: Path):
    propose(journal, source, "one")
    journal.stop("human requested selection")
    with pytest.raises(SearchError, match="registered"):
        journal.finalize(Submission(candidates=[pack_candidate("unknown", source)]))
    (source / "solve.py").write_text("changed")
    with pytest.raises(SearchError, match="digest"):
        journal.finalize(Submission(candidates=[pack_candidate("one", source)]))
    assert journal.status()["finalization"] is None


def test_finalization_cli_and_legacy_journal(journal: Journal, source: Path, capsys):
    import sqlite3

    propose(journal, source, "one")
    journal.stop("human requested selection")
    with sqlite3.connect(journal.path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(campaign)")}
        if "finalization" in columns:
            connection.execute("ALTER TABLE campaign DROP COLUMN finalization")
    assert journal.status()["finalization"] is None
    batch = Submission(candidates=[pack_candidate("one", source)])
    bundle = source.parent / "bundle.json"
    bundle.write_text(batch.model_dump_json())
    assert main(["--journal", str(journal.path), "finalize", "--bundle", str(bundle)]) == 0
    assert json.loads(capsys.readouterr().out)["batch_digest"] == batch.digest()


def test_private_admission_reads_a_readonly_journal(tmp_path: Path, source: Path):
    directory = tmp_path / "readonly"
    journal = Journal.create(directory / "search.sqlite", max_solutions=1)
    propose(journal, source, "one")
    batch = Submission(candidates=[pack_candidate("one", source)])
    journal.finalize(batch)
    journal.path.chmod(0o444)
    directory.chmod(0o555)
    try:
        with journal.private_admission(batch, max_solutions=1) as admission:
            assert admission["remaining_proposals"] == 0
            assert admission["finalization"]["batch_digest"] == batch.digest()
    finally:
        directory.chmod(0o755)
        journal.path.chmod(0o644)


def test_private_admission_keeps_the_selected_snapshot_until_acceptance(
    tmp_path: Path, source: Path,
):
    journal = Journal.create(tmp_path / "admission.sqlite", max_solutions=2)
    propose(journal, source, "one")
    propose(journal, source, "two")
    first = Submission(candidates=[pack_candidate("one", source)])
    second = Submission(candidates=[pack_candidate("two", source)])
    journal.finalize(first)
    started = threading.Event()

    def reselect():
        started.set()
        return journal.finalize(second, replace=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with journal.private_admission(first, max_solutions=2) as admission:
            future = executor.submit(reselect)
            assert started.wait(2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
            assert admission["finalization"]["batch_digest"] == first.digest()
        assert future.result(timeout=5)["batch_digest"] == second.digest()


def test_snapshot_and_lineage_are_immutable(journal: Journal, source: Path, tmp_path: Path):
    original = propose(journal, source, "one")
    assert original["digest"] == pack_candidate("one", source).digest()
    (source / "solve.py").write_text("changed\n", encoding="utf-8")
    second = propose(journal, source, "two", ("one",))
    assert second["digest"] != original["digest"]
    propose(journal, source, "combined", ("one", "two"))
    target = tmp_path / "restored"
    journal.restore("one", target)
    assert (target / "solve.py").read_text(encoding="utf-8") == "print(42)\n"
    assert (target / "solve.py").stat().st_mode & 0o111
    with pytest.raises(SearchError):
        journal.restore("one", target)
    with pytest.raises(SearchError, match="already exists"):
        propose(journal, source, "one")
    with pytest.raises(SearchError, match="parent"):
        propose(journal, source, "invalid", ("missing",))
    with pytest.raises(SearchError, match="parent"):
        propose(journal, source, "invalid", ("one", "one"))
    assert journal.status()["proposed"] == 3


@pytest.mark.parametrize("path", [".env", ".git/config", "nested/.env.local"])
def test_unsafe_snapshot_counts_and_preserves_failed_parent(
    journal: Journal, source: Path, path: str, tmp_path: Path,
):
    secret = source / path
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("secret", encoding="utf-8")
    record = journal.propose(
        "unsafe", source, hypothesis="Prepared local input", change="An unsafe snapshot",
        resources=Resources(total_tokens=19, seconds=2.5),
    )
    assert journal.status()["proposed"] == 1
    assert "restricted" in record["snapshot_error"]
    assert record["digest"] is None
    assert "snapshot_digest" not in record
    assert record["resources"] == {"total_tokens": 19, "seconds": 2.5, "tokens_complete": True}
    assert "secret" not in json.dumps(journal.export())
    with pytest.raises(SearchError, match="snapshot is unavailable"):
        journal.restore("unsafe", tmp_path / "restored")
    with pytest.raises(SearchError, match="snapshot is unavailable"):
        begin(journal, "unsafe-check", "unsafe")
    secret.unlink()
    if path.startswith(".git/"):
        secret.parent.rmdir()
    repaired = propose(journal, source, "repaired", ("unsafe",))
    assert repaired["parents"] == ["unsafe"]
    assert repaired["snapshot_error"] is None
    assert journal.status()["proposed"] == 2


def test_snapshot_rejects_symlinks(journal: Journal, source: Path):
    (source / "linked").symlink_to(source / "solve.py")
    record = propose(journal, source, "unsafe")
    assert "symlink" in record["snapshot_error"]
    assert journal.status()["proposed"] == 1


@pytest.mark.parametrize("failure", [PermissionError, FileNotFoundError])
def test_unreadable_directory_counts_without_claiming_restorability(
    journal: Journal, source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    failure: type[OSError],
):
    blocked = source / "config"
    blocked.mkdir()
    (blocked / "options.json").write_text("{}", encoding="utf-8")
    original_scandir = os.scandir

    def scandir(path):
        if Path(path) == blocked:
            raise failure("directory cannot be read")
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    record = propose(journal, source, "unreadable")
    assert journal.status()["proposed"] == 1
    assert record["snapshot_error"] == "cannot read snapshot files"
    assert "snapshot_digest" not in record
    assert record["digest"] is None
    with pytest.raises(SearchError, match="snapshot is unavailable"):
        journal.restore("unreadable", tmp_path / "restored")
    with pytest.raises(SearchError, match="snapshot is unavailable"):
        begin(journal, "unreadable-check", "unreadable")
    monkeypatch.setattr(os, "scandir", original_scandir)
    repaired = propose(journal, source, "repaired", ("unreadable",))
    assert repaired["parents"] == ["unreadable"]
    assert repaired["snapshot_error"] is None
    assert journal.status()["proposed"] == 2


def test_records_resources_failures_and_train_evidence(journal: Journal, source: Path):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    solver = Resources(total_tokens=17, seconds=2.5, tokens_complete=False)
    record = journal.finish(
        "check", error="infrastructure failure", solver=solver, judge=ZERO,
        evidence={"trace_path": "traces/check.json", "failed_examples": ["synthetic"]},
    )
    assert record["score"] is None
    assert record["solver"] == {"total_tokens": 17, "seconds": 2.5, "tokens_complete": False}
    assert record["judge"] == {"total_tokens": 0, "seconds": 0, "tokens_complete": True}
    assert record["state"] == "failed"
    assert record["completion_order"] == 1
    assert record["proposal_count_at_finish"] == 1
    assert record["elapsed_seconds"] >= 0
    with pytest.raises(SearchError, match="finished"):
        finish(journal, "check", 5)


def test_public_accepts_only_aggregate_and_private_is_rejected(journal: Journal, source: Path):
    propose(journal, source, "one")
    with pytest.raises(SearchError, match="train or public"):
        begin(journal, "private", "one", split="private")
    begin(journal, "public", "one", split="public")
    with pytest.raises(SearchError, match="public"):
        journal.finish("public", score=1, evidence={"trace": "x"})
    result = journal.finish("public", score=1)
    assert result["solver"] is None
    assert result["judge"] is None
    assert result["elapsed_seconds"] >= 0
    assert curve(journal)["final_quality"] is None


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_nonfinite_or_non_numeric_scores_rejected(journal: Journal, source: Path, score):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    with pytest.raises(SearchError, match="finite"):
        finish(journal, "check", score)
    assert journal.status()["running"] == ["check"]


@pytest.mark.parametrize("score", [2**53, 2**53 + 1, -(2**53)])
def test_scores_outside_exact_integer_contract_are_rejected(
    journal: Journal, source: Path, score: int,
):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    with pytest.raises(SearchError, match="exact binary64"):
        finish(journal, "check", score)
    assert journal.status()["running"] == ["check"]


def test_cli_does_not_round_out_of_range_integer_scores(
    journal: Journal, source: Path, capsys: pytest.CaptureFixture[str],
):
    propose(journal, source, "one")
    begin(journal, "check", "one", split="public")
    with pytest.raises(SystemExit) as error:
        main([
            "--journal", str(journal.path), "finish", "--id", "check",
            "--score", "9007199254740993",
        ])
    assert error.value.code == 2
    assert "--score" in capsys.readouterr().err
    assert journal.status()["running"] == ["check"]


def test_cannot_repeat_valid_same_conditions(journal: Journal, source: Path):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    with pytest.raises(SearchError, match="already"):
        begin(journal, "overlap", "one")
    finish(journal, "check", 1)
    with pytest.raises(SearchError, match="already"):
        begin(journal, "repeat", "one")
    begin(journal, "different-subset", "one", subset_id="other")


def test_train_curve_preserves_proposals_and_marks_gaps(journal: Journal, source: Path):
    propose(journal, source, "slow")
    begin(journal, "slow-check", "slow")
    propose(journal, source, "second")
    begin(journal, "second-check", "second")
    finish(journal, "second-check", 8)
    propose(journal, source, "third")
    begin(journal, "third-check", "third")
    finish(journal, "third-check", 9)
    propose(journal, source, "fourth")
    finish(journal, "slow-check", 5)
    propose(journal, source, "failed")
    begin(journal, "failed-check", "failed")
    journal.finish("failed-check", error="failure", solver=ZERO, judge=ZERO)
    result = curve(journal)
    assert result["horizon"] == 10
    assert result["auc"] is None
    assert result["mean_quality"] is None
    assert [(point["proposals"], point["quality"]) for point in result["points"]] == [
        (1, 5), (2, 5), (3, 5), (4, None), (5, None),
    ]
    assert result["points"][0]["candidate_id"] == "slow"
    assert result["points"][0]["completion_order"] == 3


@pytest.mark.parametrize("scores", [(1, 2, 3), (0.1, 0.2, 0.3)])
def test_batched_curve_keeps_final_quality_separate_from_auc(
    journal: Journal, source: Path, scores: tuple[float, ...],
):
    for index in range(3):
        propose(journal, source, f"solution-{index}")
        begin(journal, f"check-{index}", f"solution-{index}")
    for index, score in enumerate(scores):
        finish(journal, f"check-{index}", score)
    result = curve(journal, direction="maximize")
    assert result["auc"] is None
    assert result["final_quality"] == max(scores)


def test_curve_requires_comparable_evidence(
    journal: Journal, source: Path,
):
    propose(journal, source, "one")
    begin(journal, "wrong-subset", "one", subset_id="tiny")
    finish(journal, "wrong-subset", 1000)
    begin(journal, "train", "one")
    finish(journal, "train", -5)
    propose(journal, source, "two")
    begin(journal, "train-two", "two")
    finish(journal, "train-two", -2)
    propose(journal, source, "three")
    result = curve(journal, direction="maximize")
    assert result["auc"] is None
    assert result["observed_auc"] is None
    assert result["observed_from"] == 1
    assert result["covered_proposals"] == 2
    assert result["points"][-1]["quality"] is None
    assert result["final_quality"] == -2
    assert curve(journal, dataset_id="missing")["observed_auc"] is None


def test_cli_round_trip(tmp_path: Path, source: Path, capsys: pytest.CaptureFixture[str]):
    prefix = ["--journal", str(tmp_path / "search.sqlite")]
    assert main([*prefix, "init", "--max-solutions", "1"]) == 0
    assert main([
        *prefix, "propose", "--id", "one", "--directory", str(source),
        "--hypothesis", "synthetic", "--change", "initial", "--tokens", "0", "--seconds", "1",
    ]) == 0
    assert main([
        *prefix, "begin", "--id", "check", "--candidate", "one", "--split", "train",
        "--dataset", "v1", "--subset", "all", "--conditions", "metric-v1",
    ]) == 0
    assert main([*prefix, "stop", "--reason", "human"]) == 0
    assert main([
        *prefix, "finish", "--id", "check", "--score", "1", "--solver-tokens", "0",
        "--solver-seconds", "1", "--judge-tokens", "0", "--judge-seconds", "0",
    ]) == 0
    assert main([*prefix, "export"]) == 0
    assert "one" in capsys.readouterr().out
    assert main([*prefix, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["running"] == []


def test_large_positive_limit_is_not_a_resource_budget(tmp_path: Path):
    journal = Journal.create(tmp_path / "huge.sqlite", max_solutions=10**30)
    assert journal.status()["max_solutions"] == 10**30


def test_early_stop_keeps_campaign_horizon_incomplete(journal: Journal, source: Path):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    finish(journal, "check", 2)
    propose(journal, source, "two")
    journal.stop("human")
    result = curve(journal, horizon=10)
    assert result["actual_proposals"] == 2
    assert result["horizon"] == 10
    assert result["auc"] is None
    assert result["mean_quality"] is None
    assert result["unproposed_range"] == [3, 10]
    assert result["missing_count"] == 9
    with pytest.raises(SearchError, match="horizon"):
        curve(journal, horizon=2)


def test_snapshot_preserves_executable_bits(journal: Journal, source: Path, tmp_path: Path):
    original = propose(journal, source, "one")
    (source / "solve.py").chmod(0o644)
    changed = propose(journal, source, "two")
    assert original["digest"] != changed["digest"]
    assert "snapshot_digest" not in original
    restored = tmp_path / "restored"
    journal.restore("one", restored)
    assert (restored / "solve.py").stat().st_mode & 0o777 == 0o755
    journal.restore("two", tmp_path / "changed")
    assert (tmp_path / "changed" / "solve.py").stat().st_mode & 0o777 == 0o644


def test_legacy_snapshot_remains_restorable(journal: Journal, source: Path, tmp_path: Path):
    import sqlite3

    from evaluator.contracts import Candidate

    record = propose(journal, source, "legacy")
    record.pop("digest_version", None)
    record["digest"] = Candidate(id="legacy", files=pack_candidate("legacy", source).files).digest()
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE candidates SET record = ? WHERE id = 'legacy'", (json.dumps(record),)
        )
    restored = tmp_path / "legacy"
    journal.restore("legacy", restored)
    assert (restored / "solve.py").read_text(encoding="utf-8") == "print(42)\n"
    assert (restored / "solve.py").stat().st_mode & 0o777 == 0o755


def test_legacy_restore_preserves_empty_directories_and_access_modes(
    journal: Journal, source: Path, tmp_path: Path,
):
    import io
    import sqlite3
    import subprocess
    import sys
    import zipfile

    from evaluator.contracts import Candidate

    (source / "required").mkdir()
    (source / "solve.py").write_text(
        "from pathlib import Path\nprint(Path('required').is_dir())\n", encoding="utf-8",
    )
    (source / "solve.py").chmod(0o600)
    (source / "helper.sh").write_text("#!/bin/sh\nprintf answer", encoding="utf-8")
    (source / "helper.sh").chmod(0o700)
    record = propose(journal, source, "legacy")
    record.pop("digest_version")
    record["digest"] = Candidate(id="legacy", files=pack_candidate("legacy", source).files).digest()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path in sorted(source.rglob("*")):
            archive.write(path, path.relative_to(source).as_posix())
    snapshot = output.getvalue()
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE candidates SET record = ?, snapshot = ? WHERE id = 'legacy'",
            (json.dumps(record), snapshot),
        )
    restored = tmp_path / "legacy"
    journal.restore("legacy", restored)
    result = subprocess.run(
        [sys.executable, "solve.py"], cwd=restored, check=True, capture_output=True, text=True,
    )
    assert result.stdout == "True\n"
    assert (restored / "required").stat().st_mode & 0o777 == 0o755
    assert (restored / "solve.py").stat().st_mode & 0o777 == 0o600
    assert (restored / "helper.sh").stat().st_mode & 0o777 == 0o700
    assert restored.stat().st_mode & 0o777 == 0o755


def test_large_snapshot_is_counted_restorable_and_finalizable(
    journal: Journal, source: Path, tmp_path: Path,
):
    with (source / "large.bin").open("wb") as stream:
        stream.truncate(11 * 1024 * 1024)
    record = journal.propose(
        "too-large", source, hypothesis="Prepared index accelerates lookup",
        change="Precomputed index", resources=Resources(total_tokens=123, seconds=45),
    )
    assert journal.status()["proposed"] == 1
    assert record["snapshot_error"] is None
    assert record["bundle_error"] is None
    assert record["resources"] == {"total_tokens": 123, "seconds": 45, "tokens_complete": True}
    assert record["digest"]
    (source / "large.bin").unlink()
    restored = tmp_path / "restored-large"
    journal.restore("too-large", restored)
    assert (restored / "large.bin").stat().st_size == 11 * 1024 * 1024
    begin(journal, "train-large", "too-large")
    finish(journal, "train-large", 1)
    child = propose(journal, source, "smaller", ("too-large",))
    assert child["parents"] == ["too-large"]
    assert child["proposal"] == 2
    assert child["bundle_error"] is None
    journal.stop("human requested selection")
    batch = Submission(candidates=[pack_candidate("too-large", restored)])
    assert journal.finalize(batch)["batch_digest"] == batch.digest()


def test_file_count_does_not_cap_research_snapshot_or_finalization(
    journal: Journal, source: Path, tmp_path: Path,
):
    for index in range(1000):
        (source / f"input-{index}.txt").write_text(str(index), encoding="utf-8")
    record = propose(journal, source, "many-files")
    assert record["snapshot_error"] is None
    assert record["bundle_error"] is None
    assert journal.status()["proposed"] == 1
    restored = tmp_path / "many-files"
    journal.restore("many-files", restored)
    assert len(list(restored.iterdir())) == 1001
    journal.stop("human requested selection")
    batch = Submission(candidates=[pack_candidate("many-files", restored)])
    assert journal.finalize(batch)["batch_digest"] == batch.digest()


@pytest.mark.parametrize("kind", ["large", "many-files"])
def test_finalization_revalidates_legacy_bundle_error_without_rewriting_history(
    tmp_path: Path, source: Path, kind: str,
):
    import sqlite3

    if kind == "large":
        with (source / "large.bin").open("wb") as stream:
            stream.truncate(11 * 1024 * 1024)
    else:
        for index in range(1000):
            (source / f"input-{index}.txt").write_text(str(index), encoding="utf-8")
    journal = Journal.create(tmp_path / "search.sqlite", max_solutions=1)
    record = propose(journal, source, "legacy")
    record["bundle_error"] = "snapshot does not satisfy the candidate bundle contract"
    original_record = json.dumps(record)
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE candidates SET record = ? WHERE id = 'legacy'", (original_record,),
        )
        original_snapshot = connection.execute(
            "SELECT snapshot FROM candidates WHERE id = 'legacy'",
        ).fetchone()[0]
    restored = tmp_path / "restored"
    journal.restore("legacy", restored)
    batch = Submission(candidates=[pack_candidate("legacy", restored)])
    assert journal.finalize(batch)["batch_digest"] == batch.digest()
    with journal.private_admission(batch, max_solutions=1) as admission:
        assert admission["remaining_proposals"] == 0
    with sqlite3.connect(journal.path) as connection:
        stored = connection.execute(
            "SELECT record, snapshot FROM candidates WHERE id = 'legacy'",
        ).fetchone()
    assert stored == (original_record, original_snapshot)


def test_missing_snapshot_counts_without_claiming_restorability(
    journal: Journal, tmp_path: Path,
):
    record = propose(journal, tmp_path / "missing", "missing-source")
    assert record["snapshot_error"]
    assert "snapshot_digest" not in record
    assert record["digest"] is None
    assert journal.status()["proposed"] == 1


def test_unsafe_filename_counts_without_claiming_restorability(journal: Journal, source: Path):
    (source / "bad\\path").write_text("payload", encoding="utf-8")
    record = propose(journal, source, "unsafe-name")
    assert record["snapshot_error"] == "snapshot contains an unsafe path"
    assert "snapshot_digest" not in record
    assert journal.status()["proposed"] == 1


def test_unencodable_filename_retains_resources_and_ancestry(
    journal: Journal, source: Path, tmp_path: Path,
):
    propose(journal, source, "parent")
    filename = os.fsencode(source) + b"/bad-\xff.py"
    with open(filename, "wb") as stream:
        stream.write(b"print(1)\n")
    record = journal.propose(
        "bad-filename", source, parents=("parent",), hypothesis="A revised program",
        change="An unsupported filename", resources=Resources(total_tokens=41, seconds=3),
    )
    assert journal.status()["proposed"] == 2
    assert record["snapshot_error"] == "snapshot filenames must be UTF-8"
    assert record["resources"] == {"total_tokens": 41, "seconds": 3, "tokens_complete": True}
    assert record["parents"] == ["parent"]
    assert record["digest"] is None
    assert "snapshot_digest" not in record
    with pytest.raises(SearchError, match="snapshot is unavailable"):
        journal.restore("bad-filename", tmp_path / "restored")
    os.unlink(filename)
    repaired = propose(journal, source, "repaired", ("bad-filename",))
    assert repaired["parents"] == ["bad-filename"]
    assert repaired["snapshot_error"] is None
    assert journal.status()["proposed"] == 3


def test_restore_omits_empty_directories(journal: Journal, source: Path, tmp_path: Path):
    (source / "empty").mkdir()
    record = propose(journal, source, "empty-dir")
    assert record["digest"] == pack_candidate("empty-dir", source).digest()
    restored = tmp_path / "restored"
    journal.restore("empty-dir", restored)
    assert not (restored / "empty").exists()


@pytest.mark.parametrize("score", [1e308, -1e308])
def test_curve_reports_integral_overflow(journal: Journal, source: Path, score: float):
    for identifier in ("first", "second"):
        propose(journal, source, identifier)
        begin(journal, identifier, identifier, split="public")
        journal.finish(identifier, score=score)
    with pytest.raises(SearchError, match="curve integral exceeds finite numeric range"):
        curve(journal, split="public")


def test_curve_cli_reports_integral_overflow(
    journal: Journal, source: Path, capsys: pytest.CaptureFixture[str],
):
    for identifier in ("first", "second"):
        propose(journal, source, identifier)
        begin(journal, identifier, identifier, split="public")
        journal.finish(identifier, score=1e308)
    assert main([
        "--journal", str(journal.path), "curve", "--split", "public", "--dataset", "dataset-v1",
        "--subset", "all", "--conditions", "metric-v1-fixed-conditions", "--direction", "maximize",
    ]) == 1
    assert "curve integral exceeds finite numeric range" in capsys.readouterr().err


def test_restore_can_be_read_and_executed_by_another_uid(
    journal: Journal, source: Path, tmp_path: Path,
):
    (source / "nested").mkdir(mode=0o700)
    (source / "nested" / "data.txt").write_text("payload", encoding="utf-8")
    (source / "nested" / "data.txt").chmod(0o600)
    (source / "solve.py").chmod(0o700)
    propose(journal, source, "local-uid")
    restored = tmp_path / "restored"
    mask = os.umask(0o077)
    try:
        journal.restore("local-uid", restored)
    finally:
        os.umask(mask)
    assert restored.stat().st_mode & 0o777 == 0o755
    assert (restored / "nested").stat().st_mode & 0o777 == 0o755
    assert (restored / "nested" / "data.txt").stat().st_mode & 0o777 == 0o644
    assert (restored / "solve.py").stat().st_mode & 0o777 == 0o755


def test_public_failure_does_not_accept_detailed_error(journal: Journal, source: Path):
    propose(journal, source, "one")
    begin(journal, "check", "one", split="public")
    with pytest.raises(SearchError, match="public"):
        journal.finish("check", error="hidden example details")
    result = journal.finish("check", error="evaluation_failed")
    assert result["error"] == "evaluation_failed"


@pytest.mark.parametrize("resources", [{}, {"solver": ZERO}, {"judge": ZERO}])
def test_train_requires_both_resource_measurements(journal: Journal, source: Path, resources):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    with pytest.raises(SearchError, match="train requires"):
        journal.finish("check", score=1, **resources)
    assert journal.status()["running"] == ["check"]


@pytest.mark.parametrize("resources", [
    {"solver": ZERO}, {"judge": ZERO}, {"solver": ZERO, "judge": ZERO},
])
def test_public_rejects_resource_accounting(journal: Journal, source: Path, resources):
    propose(journal, source, "one")
    begin(journal, "check", "one", split="public")
    with pytest.raises(SearchError, match="public"):
        journal.finish("check", score=1, **resources)
    assert journal.status()["running"] == ["check"]


def test_public_cli_finishes_with_score_only(
    journal: Journal, source: Path, capsys: pytest.CaptureFixture[str],
):
    propose(journal, source, "one")
    begin(journal, "check", "one", split="public")
    assert main([
        "--journal", str(journal.path), "finish", "--id", "check", "--score", "-3",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == -3
    assert result["solver"] is None
    assert result["judge"] is None


@pytest.mark.parametrize("arguments", [
    [], ["--solver-tokens", "0"], ["--solver-seconds", "1"],
    ["--judge-tokens", "0"], ["--judge-seconds", "1"],
    ["--solver-tokens-incomplete"], ["--judge-tokens-incomplete"],
    ["--solver-tokens", "0", "--solver-seconds", "1"],
])
def test_train_cli_rejects_missing_or_partial_resources(
    journal: Journal, source: Path, arguments: list[str], capsys: pytest.CaptureFixture[str],
):
    propose(journal, source, "one")
    begin(journal, "check", "one")
    assert main([
        "--journal", str(journal.path), "finish", "--id", "check", "--score", "1", *arguments,
    ]) == 1
    assert "arena search:" in capsys.readouterr().err
    assert journal.status()["running"] == ["check"]


def test_proposal_assigns_one_canonical_solution_number(journal: Journal, source: Path):
    first = journal.propose(
        None, source, hypothesis="A new mechanism", change="Initial version",
    )
    assert (first["candidate_id"], first["solution_id"], first["proposal"]) == ("S01", "S01", 1)
    second = propose(journal, source, "semantic-retrieval", ("S01",))
    assert second["solution_id"] == "S02"
    assert first["resources"]["tokens_complete"] is False
    assert first["research_resources"]["attribution"] == "unknown"


@pytest.mark.parametrize("candidate_id", ["S02", "S01a", "sc71-s06-retrieval", "sc71-s01b"])
def test_proposal_rejects_conflicting_or_suffixed_solution_numbers(
    journal: Journal, source: Path, candidate_id: str,
):
    with pytest.raises(SearchError, match="solution number"):
        propose(journal, source, candidate_id)
    assert journal.status()["proposed"] == 0


def test_registered_candidate_checks_exact_snapshot(journal: Journal, source: Path):
    original = propose(journal, source, "first")
    assert journal.registered_candidate("first", original["digest"])["solution_id"] == "S01"
    with pytest.raises(SearchError, match="registered"):
        journal.registered_candidate("unknown")
    with pytest.raises(SearchError, match="digest"):
        journal.registered_candidate("first", "different")


def test_public_auc_follows_proposal_order_without_baseline(tmp_path: Path, source: Path):
    journal = Journal.create(tmp_path / "public.sqlite", max_solutions=3)
    for name in ("slow", "fast", "last"):
        propose(journal, source, name)
        begin(journal, name, name, split="public")
    journal.finish("fast", score=5)
    journal.finish("last", score=3)
    journal.finish("slow", score=2)
    result = curve(journal, split="public", direction="maximize")
    assert result["integration"] == "right_step"
    assert result["auc"] == 12
    assert result["mean_quality"] == 4
    assert [(point["proposal"], point["quality"]) for point in result["points"]] == [
        (1, 2), (2, 5), (3, 5),
    ]
    assert result["complete"] is True
    assert "baseline" not in result


def test_public_auc_is_incomplete_until_every_proposal_has_score(tmp_path: Path, source: Path):
    journal = Journal.create(tmp_path / "public.sqlite", max_solutions=4)
    for name in ("first", "failed", "last"):
        propose(journal, source, name)
        begin(journal, name, name, split="public")
    journal.finish("last", score=-1)
    journal.finish("first", score=-3)
    journal.finish("failed", error="evaluation_failed")
    result = curve(journal, split="public", direction="maximize")
    assert result["auc"] is None
    assert result["mean_quality"] is None
    assert result["observed_auc"] == -3
    assert result["covered_proposals"] == 1
    assert result["missing_proposals"] == [2]
    assert result["unproposed_range"] == [4, 4]
    assert result["failed_proposals"] == [2]
    assert result["final_quality"] == -1
    assert result["points"][1]["score"] is None
    assert result["points"][1]["quality"] is None


def test_train_curve_never_reports_auc(tmp_path: Path, source: Path):
    journal = Journal.create(tmp_path / "train.sqlite", max_solutions=1)
    propose(journal, source, "first")
    begin(journal, "first", "first")
    finish(journal, "first", 7)
    result = curve(journal)
    assert result["auc"] is None
    assert result["observed_auc"] is None
    assert result["auc_applicable"] is False


def test_measurement_admission_requires_matching_running_registration(
    journal: Journal, source: Path,
):
    candidate = propose(journal, source, "registered")
    comparison = {"split": "train", "dataset_id": "dataset-v1", "subset_id": "all",
                  "conditions_id": "metric-v1-fixed-conditions"}
    with pytest.raises(SearchError, match="running measurement"):
        journal.admit_measurement("registered", candidate["digest"], **comparison)
    begin(journal, "check", "registered")
    admitted = journal.admit_measurement("registered", candidate["digest"], **comparison)
    assert admitted["measurement_id"] == "check"
    with pytest.raises(SearchError, match="digest"):
        journal.admit_measurement("registered", "different", **comparison)
    with pytest.raises(SearchError, match="running measurement"):
        journal.admit_measurement("registered", candidate["digest"], measurement_id="other",
                                  **comparison)
    finish(journal, "check", 1)
    with pytest.raises(SearchError, match="running measurement"):
        journal.admit_measurement("registered", candidate["digest"], **comparison)


def test_research_usage_attaches_real_calls_without_overwriting_history(
    journal: Journal, source: Path, tmp_path: Path,
):
    from evaluator.usage import seal_usage
    from tests.test_research_usage import record

    original = propose(journal, source, "first")
    run_id, request_id = record(tmp_path, 41)
    seal_usage(tmp_path / "usage", run_id)
    summary = journal.record_research_usage(
        usage_dir=tmp_path / "usage", trace_dir=tmp_path / "traces", run_ids=[run_id],
        assignments={"first": [{"run_id": run_id, "request_id": request_id}]},
    )
    assert summary["total"]["usage"]["total_tokens"] == 41
    exported = journal.export()
    assert exported["research_usage"]["source"] == "gateway_ledger"
    assert exported["candidates"][0]["research_resources"]["usage"]["total_tokens"] == 41
    assert (journal.registered_candidate("first")["research_resources"]
            == original["research_resources"])
    assert exported["candidates"][0]["resources_source"] == "self_reported"
    with pytest.raises(SearchError, match="registered"):
        journal.record_research_usage(
            usage_dir=tmp_path / "usage", trace_dir=tmp_path / "traces", run_ids=[run_id],
            assignments={"unknown": [{"run_id": run_id}]},
        )


def test_research_usage_cli_keeps_unattributed_total(
    journal: Journal, tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    from evaluator.usage import seal_usage
    from tests.test_research_usage import record

    run_id, _ = record(tmp_path, 19)
    seal_usage(tmp_path / "usage", run_id)
    assert main([
        "--journal", str(journal.path), "research-usage", "--usage-dir", str(tmp_path / "usage"),
        "--trace-dir", str(tmp_path / "traces"), "--run-id", run_id,
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total"]["usage"]["total_tokens"] == 19
    assert result["unattributed"]["usage"]["total_tokens"] == 19
    assert result["attribution_complete"] is False


def test_candidate_admission_binds_snapshot_and_defers_stop(journal: Journal, source: Path):
    candidate = propose(journal, source, "registered")
    started = threading.Event()

    def stop():
        started.set()
        return journal.stop("human requested stop")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with journal.candidate_admission("registered", candidate["digest"], max_solutions=10):
            future = executor.submit(stop)
            assert started.wait(2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
        assert future.result(timeout=5)["stopped"]
    assert journal.registered_candidate("registered")["digest"] == candidate["digest"]
    with pytest.raises(SearchError, match="stopped"):
        with journal.candidate_admission("registered", candidate["digest"]):
            pytest.fail("stopped search admitted a new build")


def test_cli_assigns_id_and_does_not_require_self_reported_resources(
    journal: Journal, source: Path, capsys: pytest.CaptureFixture[str],
):
    assert main([
        "--journal", str(journal.path), "propose", "--directory", str(source),
        "--hypothesis", "A mechanism", "--change", "Initial version",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidate_id"] == result["solution_id"] == "S01"
    assert result["resources_source"] == "unknown"


def test_measurement_admission_rejects_limit_mismatch_and_ambiguous_subset(
    journal: Journal, source: Path,
):
    candidate = propose(journal, source, "registered")
    begin(journal, "first", "registered")
    with pytest.raises(SearchError, match="proposal limit"):
        journal.admit_measurement("registered", candidate["digest"], split="train", max_solutions=9)
    begin(journal, "second", "registered", subset_id="tiny")
    with pytest.raises(SearchError, match="running measurement"):
        journal.admit_measurement("registered", candidate["digest"], split="train")
    assert journal.admit_measurement(
        "registered", candidate["digest"], split="train", measurement_id="second",
    )["comparison"]["subset_id"] == "tiny"
