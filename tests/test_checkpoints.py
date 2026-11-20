import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from evaluator.checkpoints import (
    materialize_document,
    read_document,
    write_example,
    write_metadata,
)


def solver(example_id: str, answer: str = "answer") -> dict:
    return {
        "example_id": example_id, "answer": answer, "status": "judging",
        "resources": {"solver": {"tokens": 10}},
        "calls": [{"name": "solver", "payload": "large solver history"}],
    }


def judged(detail: dict, *, failed: bool = False) -> dict:
    result = deepcopy(detail)
    result.pop("status", None)
    result["resources"]["judge"] = {"tokens": 5}
    result["calls"].append({"name": "judge", "payload": "judge history"})
    result["failure" if failed else "score"] = "judge_failure" if failed else 1.0
    return result


def test_checkpoint_restores_out_of_order_stages_in_dataset_order(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {"candidate_id": "one"}, example_ids=["x", "y", "z"])
    write_example(path, 2, judged(solver("z")))
    write_example(path, 0, solver("x"))
    assert read_document(path) == {
        "candidate_id": "one", "examples": [solver("x"), judged(solver("z"))],
    }
    assert "answer" not in path.read_text()


def test_stage_update_does_not_rewrite_or_duplicate_solver_payload(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {"candidate_id": "one"}, example_ids=["x"])
    first = solver("x", "unique answer " * 1000)
    write_example(path, 0, first)
    before = {item: item.read_bytes() for item in tmp_path.rglob("*.json")}
    final = judged(first)
    write_example(path, 0, final)
    assert read_document(path)["examples"] == [final]
    for item, value in before.items():
        assert item.read_bytes() == value
    new = [item for item in tmp_path.rglob("*.json") if item not in before]
    assert len(new) == 1
    assert "unique answer" not in new[0].read_text()
    assert "large solver history" not in new[0].read_text()
    assert "judge history" in new[0].read_text()
    write_example(path, 0, final)
    assert len(list(tmp_path.rglob("*.json"))) == len(before) + 1


def test_failed_judge_can_continue_without_repeating_saved_solver(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {"candidate_id": "one"}, example_ids=["x"])
    first = solver("x")
    failed = judged(first, failed=True)
    write_example(path, 0, first)
    write_example(path, 0, failed)
    final = deepcopy(failed)
    final.pop("failure")
    final["score"] = 0.5
    final["resources"]["judge"]["tokens"] += 7
    final["calls"].append({"name": "judge", "payload": "second judge history"})
    write_example(path, 0, final)
    assert read_document(path)["examples"] == [final]


def test_materialization_is_legacy_readable_and_can_be_resumed(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {"candidate_id": "one"}, example_ids=["x", "y"])
    write_example(path, 0, judged(solver("x")))
    document = read_document(path)
    materialize_document(path)
    assert json.loads(path.read_text()) == document
    assert read_document(path) == document
    write_metadata(path, {"candidate_id": "one"}, example_ids=["x", "y"])
    write_example(path, 1, solver("y"))
    assert read_document(path)["examples"] == [judged(solver("x")), solver("y")]


def test_legacy_document_is_read_without_side_effects(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    document = {"candidate_id": "old", "examples": [solver("x")]}
    path.write_text(json.dumps(document))
    before = path.read_bytes()
    assert read_document(path) == document
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_independent_example_writers_do_not_lose_completions(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    ids = [f"example-{index}" for index in range(24)]
    write_metadata(path, {"candidate_id": "one"}, example_ids=ids)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write_example, path, index, judged(solver(example_id)))
                   for index, example_id in reversed(list(enumerate(ids)))]
        for future in futures:
            future.result()
    assert [detail["example_id"] for detail in read_document(path)["examples"]] == ids


@pytest.mark.parametrize("change", ["answer", "solver", "calls", "example_id"])
def test_saved_solver_stage_is_immutable(tmp_path: Path, change: str) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {}, example_ids=["x"])
    detail = solver("x")
    write_example(path, 0, detail)
    replacement = deepcopy(detail)
    if change == "solver":
        replacement["resources"]["solver"]["tokens"] = 20
    elif change == "calls":
        replacement["calls"] = []
    else:
        replacement[change] = "different"
    before = {item: item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    with pytest.raises(ValueError):
        write_example(path, 0, replacement)
    assert {item: item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()} == before


@pytest.mark.parametrize("ids", [["a", "a"], [""], [1]])
def test_order_rejects_duplicate_or_invalid_ids(tmp_path: Path, ids: list) -> None:
    with pytest.raises(ValueError):
        write_metadata(tmp_path / "candidate.json", {}, example_ids=ids)


def test_order_cannot_change_during_resume(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {}, example_ids=["x", "y"])
    with pytest.raises(ValueError):
        write_metadata(path, {}, example_ids=["y", "x"])


@pytest.mark.parametrize("index,example_id", [(-1, "x"), (2, "x"), (0, "y")])
def test_example_must_match_frozen_order(tmp_path: Path, index: int, example_id: str) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {}, example_ids=["x"])
    with pytest.raises(ValueError):
        write_example(path, index, solver(example_id))


@pytest.mark.parametrize("damage", ["unknown", "outside", "gap", "symlink"])
def test_reader_rejects_invalid_stage_inventory(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {}, example_ids=["x"])
    write_example(path, 0, solver("x"))
    store = tmp_path / "candidate.examples"
    if damage == "unknown":
        (store / "unexpected.json").write_text("{}")
    elif damage == "outside":
        (store / "1").mkdir()
    elif damage == "gap":
        stage = next((store / "0").glob("*.json"))
        stage.rename(stage.with_name("stage-000002.json"))
    else:
        target = tmp_path / "outside"
        target.mkdir()
        (store / "0").rename(target / "original")
        (store / "0").symlink_to(target / "original", target_is_directory=True)
    with pytest.raises(ValueError):
        read_document(path)


def test_crash_before_first_stage_leaves_no_completed_example(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {}, example_ids=["x"])
    pending = tmp_path / "candidate.examples/0"
    pending.mkdir()
    (pending / ".pending-incomplete").write_text('{"detail":')
    assert read_document(path)["examples"] == []
    write_example(path, 0, solver("x"))
    assert read_document(path)["examples"] == [solver("x")]


def test_stage_transition_preserves_solved_answer(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    write_metadata(path, {}, example_ids=["x"])
    detail = solver("x")
    detail["status"] = "solved"
    write_example(path, 0, detail)
    assert read_document(path)["examples"][0]["status"] == "solved"
    detail["status"] = "judging"
    write_example(path, 0, detail)
    assert read_document(path)["examples"][0]["status"] == "judging"
    write_example(path, 0, judged(detail))
    assert read_document(path)["examples"][0] == judged(detail)


def test_first_example_flushes_parent_directory_before_stage_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    path = tmp_path / "candidate.json"
    write_metadata(path, {}, example_ids=["x"])
    store_inode = (tmp_path / "candidate.examples").stat().st_ino
    flushed = set()
    fsync = os.fsync
    link = os.link

    def sync(descriptor: int) -> None:
        flushed.add(os.fstat(descriptor).st_ino)
        fsync(descriptor)

    def commit(source: str | Path, destination: str | Path) -> None:
        assert store_inode in flushed
        link(source, destination)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "link", commit)
    write_example(path, 0, solver("x"))
