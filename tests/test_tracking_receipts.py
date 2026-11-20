import json
import tracemalloc
from pathlib import Path
from unittest.mock import Mock

from evaluator import tracking


def write_record(path: Path, count: int = 1000) -> None:
    with path.open("w") as stream:
        stream.write('{"split":"train","status":"FINISHED",')
        stream.write('"params":{"case_id":"synthetic","campaign_id":"test",')
        stream.write('"candidate_id":"S01"},"metrics":{"score":1},')
        stream.write('"start_time_ns":1,"end_time_ns":2,"examples":[')
        for index in range(count):
            if index:
                stream.write(",")
            json.dump({"answer": "x" * 32768}, stream)
        stream.write("]}")


def test_large_record_has_small_receipt_and_streamed_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    source = tmp_path / "record.json"
    write_record(source)
    queue = tmp_path / "queue"
    tracemalloc.start()
    try:
        assert tracking.export_record(source, queue, defer=True) == {"status": "tracking_pending"}
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024
    receipt = next(queue.glob("*.json"))
    entry = json.loads(receipt.read_text())
    assert receipt.stat().st_size < 4096
    assert "record" not in entry
    payload = json.loads(Path(entry["record_file"]).read_text())
    assert "examples" not in payload
    assert Path(payload["artifact_files"]["examples"]).stat().st_size > 30 * 1024 * 1024


def test_exported_receipt_does_not_open_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    queue = tmp_path / "queue"
    queue.mkdir()
    entry = {"uri": "http://shared", "operator": False, "export_id": "old",
             "status": "exported", "run_id": "run-one",
             "record_file": str(tmp_path / "missing.json")}
    (queue / "receipt.json").write_text(json.dumps(entry))
    monkeypatch.setattr(tracking, "_publish", Mock(side_effect=AssertionError("published")))
    assert tracking.flush_queue(queue) == []


def test_legacy_exported_queue_migrates_without_materializing_record(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    queue = tmp_path / "queue"
    queue.mkdir()
    source = queue / "legacy.json"
    with source.open("w") as stream:
        stream.write('{"uri":"http://shared","operator":false,"export_id":"legacy",')
        stream.write('"record":{"examples":[')
        for index in range(1000):
            stream.write(("," if index else "") + json.dumps("x" * 32768))
        stream.write(']},"status":"exported","run_id":"run-one"}')
    tracemalloc.start()
    try:
        assert tracking.flush_queue(queue) == []
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024
    entry = json.loads(source.read_text())
    assert source.stat().st_size < 4096
    assert Path(entry["legacy_file"]).exists()


def test_deferred_payload_survives_removed_source_and_keeps_call_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    source = tmp_path / "record.json"
    write_record(source, 0)
    queue = tmp_path / "queue"
    tracking.export_record(source, queue, defer=True)
    entry_path = next(queue.glob("*.json"))
    entry = json.loads(entry_path.read_text())
    source.unlink()
    publish = Mock(return_value="run-one")
    monkeypatch.setattr(tracking, "_publish", publish)
    assert tracking.flush_queue(queue) == [{"status": "exported", "run_id": "run-one"}]
    assert publish.call_args.args[2] == entry["export_id"]
    assert tracking.flush_queue(queue) == []


def test_call_snapshot_survives_in_place_source_mutation(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    source = tmp_path / "record.json"
    write_record(source, 0)
    call = tmp_path / "call.json"
    call.write_text(json.dumps({"name": "model", "inputs": "original"}, ensure_ascii=False))
    value = json.loads(source.read_text())
    value["call_files"] = [str(call)]
    source.write_text(json.dumps(value))
    queue = tmp_path / "queue"
    tracking.export_record(source, queue, defer=True)
    entry = json.loads(next(queue.glob("*.json")).read_text())
    record = json.loads(Path(entry["record_file"]).read_text())
    snapshot = Path(record["call_files"][0])
    call.write_text(json.dumps({"name": "model", "inputs": "mutated"}))
    assert json.loads(snapshot.read_text())["inputs"] == "original"


def test_single_call_history_is_split_without_materializing_call(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared")
    source = tmp_path / "record.json"
    write_record(source, 0)
    prefix = source.read_text()[:-1]
    with source.open("w") as stream:
        stream.write(prefix + ',"calls":[{"name":"model","inputs":{"messages":[')
        for index in range(1000):
            stream.write(("," if index else "") + json.dumps("x" * 32768))
        stream.write("]}}]}")
    queue = tmp_path / "queue"
    tracemalloc.start()
    try:
        tracking.export_record(source, queue, defer=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024
    entry = json.loads(next(queue.glob("*.json")).read_text())
    record = json.loads(Path(entry["record_file"]).read_text())
    assert "calls" not in record
    assert len(record["call_files"]) == 1
    assert Path(record["call_files"][0]).stat().st_size > 30 * 1024 * 1024
