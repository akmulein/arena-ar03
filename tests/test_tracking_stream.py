import json
import tracemalloc
from pathlib import Path

import pytest

from evaluator.tracking_stream import read_fields, split_object


def test_split_preserves_all_fields_and_safe_paths(tmp_path: Path) -> None:
    value = {
        "split": "train", "params": {"candidate_id": "S01", "unicode": "тест"},
        "odd/../../field": [True, None, 4, 1.25, {"nested": []}], "empty": {},
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    files = split_object(source, tmp_path / "fields")
    assert {key: json.loads(path.read_text()) for key, path in files.items()} == value
    assert {path.name for path in files.values()} == {
        "00000000.json", "00000001.json", "00000002.json", "00000003.json",
    }
    assert all(path.parent == tmp_path / "fields" for path in files.values())
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in files.values())


def test_split_nested_legacy_record_and_redacts_values(tmp_path: Path) -> None:
    value = {
        "status": "tracking_pending",
        "record": {"calls": [{"api_key": {"nested": ["credential"]},
                               "text": "Bearer secret", "secret-key-name": "value"}]},
        "after": "unrelated",
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    files = split_object(
        source, tmp_path / "fields", prefix="record",
        transform=lambda item: item.replace("secret", "redacted")
        if isinstance(item, str) else item,
        secret_key=lambda key: key == "api_key",
    )
    assert json.loads(files["calls"].read_text()) == [
        {"api_key": "[REDACTED]", "text": "Bearer redacted", "redacted-key-name": "value"},
    ]
    assert read_fields(source, {"status", "after", "missing"}) == {
        "status": "tracking_pending", "after": "unrelated",
    }
    assert read_fields(source, {"calls"}, prefix="record") == value["record"]


def test_streams_large_arrays_without_materializing_them(tmp_path: Path) -> None:
    source = tmp_path / "large.json"
    with source.open("w", encoding="utf-8") as stream:
        stream.write('{"before": 1, "examples": [')
        item = json.dumps({"input": "x" * 1024, "score": 0.5})
        for index in range(20000):
            stream.write(("," if index else "") + item)
        stream.write('], "after": "finished"}')
    tracemalloc.start()
    try:
        assert read_fields(source, {"before", "after"}) == {"before": 1, "after": "finished"}
        files = split_object(source, tmp_path / "fields")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024
    assert files["examples"].stat().st_size > 20 * 1024 * 1024


@pytest.mark.parametrize("content,prefix", [
    ('[]', ''), ('{"record": []}', 'record'), ('{"other": {}}', 'record'),
    ('{"a": 1, "a": 2}', ''), ('{"a": [1, 2}', ''),
])
def test_rejects_invalid_or_ambiguous_records(tmp_path: Path, content: str, prefix: str) -> None:
    source = tmp_path / "source.json"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        split_object(source, tmp_path / "fields", prefix=prefix)
    with pytest.raises(ValueError):
        read_fields(source, {"a"}, prefix=prefix)


def test_rewrite_streams_arbitrary_json_and_preserves_source(tmp_path: Path) -> None:
    from evaluator.tracking_stream import rewrite_json

    source = tmp_path / "source.json"
    value = [{"token": ["hidden"], "text": "old credential"}, True, None, 3.75]
    source.write_text(json.dumps(value), encoding="utf-8")
    target = tmp_path / "result.json"
    rewrite_json(
        source, target,
        transform=lambda item: item.replace("credential", "[REDACTED]")
        if isinstance(item, str) else item,
        secret_key=lambda key: key == "token",
    )
    assert json.loads(target.read_text()) == [
        {"token": "[REDACTED]", "text": "old [REDACTED]"}, True, None, 3.75,
    ]
    assert json.loads(source.read_text()) == value
    assert target.stat().st_mode & 0o777 == 0o600


def test_rewrite_failure_keeps_previous_output(tmp_path: Path) -> None:
    from evaluator.tracking_stream import rewrite_json

    source = tmp_path / "source.json"
    source.write_text('[{"incomplete": 1}', encoding="utf-8")
    target = tmp_path / "result.json"
    target.write_text('"previous"', encoding="utf-8")
    with pytest.raises(ValueError):
        rewrite_json(source, target)
    assert target.read_text() == '"previous"'
    assert not list(tmp_path.glob("*.tmp"))


def test_split_array_preserves_order_and_redacts_nested_fields(tmp_path: Path) -> None:
    from evaluator.tracking_stream import split_array

    source = tmp_path / "calls.json"
    source.write_text(json.dumps([
        {"name": "first", "api_key": {"nested": ["hidden"]}, "text": "old secret"},
        None, [], {"name": "last"},
    ]), encoding="utf-8")
    paths = split_array(
        source, tmp_path / "calls",
        transform=lambda item: item.replace("secret", "redacted")
        if isinstance(item, str) else item,
        secret_key=lambda key: key == "api_key",
    )
    assert [json.loads(path.read_text()) for path in paths] == [
        {"name": "first", "api_key": "[REDACTED]", "text": "old redacted"},
        None, [], {"name": "last"},
    ]
    assert [path.name for path in paths] == [f"{index:08d}.json" for index in range(4)]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)


def test_split_array_streams_single_call_with_large_history(tmp_path: Path) -> None:
    from evaluator.tracking_stream import split_array

    source = tmp_path / "calls.json"
    with source.open("w", encoding="utf-8") as stream:
        stream.write('[{"name":"model","inputs":{"messages":[')
        item = json.dumps({"role": "user", "content": "x" * 1024})
        for index in range(20000):
            stream.write(("," if index else "") + item)
        stream.write(']},"outputs":"done"}]')
    tracemalloc.start()
    try:
        paths = split_array(source, tmp_path / "calls")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(paths) == 1
    assert paths[0].stat().st_size > 20 * 1024 * 1024
    assert peak < 2 * 1024 * 1024
    assert read_fields(paths[0], {"name", "outputs"}) == {"name": "model", "outputs": "done"}


@pytest.mark.parametrize("content", ['{}', 'null', '[1, {"incomplete": []}', '[] []'])
def test_split_array_rejects_invalid_input(tmp_path: Path, content: str) -> None:
    from evaluator.tracking_stream import split_array

    source = tmp_path / "calls.json"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        split_array(source, tmp_path / "calls")
    assert not list((tmp_path / "calls").glob("*.tmp"))


def test_split_array_accepts_empty_input_array(tmp_path: Path) -> None:
    from evaluator.tracking_stream import split_array

    source = tmp_path / "calls.json"
    source.write_text("[]", encoding="utf-8")
    assert split_array(source, tmp_path / "calls") == []
