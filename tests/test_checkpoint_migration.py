import json

import pytest

from evaluator import checkpoints


def test_legacy_migration_keeps_complete_snapshot_until_all_stages_are_durable(
    tmp_path, monkeypatch,
):
    path = tmp_path / "legacy.json"
    document = {"candidate_id": "a", "examples": [
        {"example_id": str(index), "answer": f"answer-{index}", "resources": {"solver": {}}}
        for index in range(3)
    ]}
    path.write_text(json.dumps(document))
    original = path.read_bytes()
    write = checkpoints.write_example

    def interrupted(path, index, detail):
        if index == 1:
            raise OSError("synthetic interruption")
        write(path, index, detail)

    monkeypatch.setattr(checkpoints, "write_example", interrupted)
    with pytest.raises(OSError, match="synthetic interruption"):
        checkpoints.write_metadata(path, {"candidate_id": "a"}, example_ids=list("012"))
    assert path.read_bytes() == original
    assert checkpoints.read_document(path) == document
    monkeypatch.setattr(checkpoints, "write_example", write)
    checkpoints.write_metadata(path, {"candidate_id": "a"}, example_ids=list("012"))
    assert checkpoints.read_document(path) == document
