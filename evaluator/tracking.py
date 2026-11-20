import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, cast
from urllib.parse import urlsplit
from uuid import uuid4

from evaluator.tracking_artifacts import (
    _write_json,
    download_json_artifact,
    log_json_artifact,
    log_json_file_artifact,
    read_json_artifact,
)
from evaluator.tracking_stream import read_fields, rewrite_json, split_array, split_object

_PUBLIC_PARAMS = {
    "campaign_id", "case_id", "candidate_id", "candidate_digest", "revision", "parent_ids",
    "proposal_index", "hypothesis", "change", "model", "harness", "harness_version", "parameters",
    "metric_version", "direction", "dataset_id", "dataset_version",
    "conditions_id",
}
_SECRET_KEY = re.compile(
    r"(^|_)(authorization|api_key|password|secret|access_token|refresh_token|capability|token)$",
    re.IGNORECASE,
)
_EXPORT_LOCK = RLock()
MAX_TRACE_BYTES = 1024 * 1024
MAX_TRACE_CALLS = 64


class _SizeLimit(Exception):
    pass


class _SizeCounter:
    def __init__(self, limit: int):
        self.limit = limit
        self.size = 0

    def write(self, value: bytes) -> int:
        self.size += len(value)
        if self.size > self.limit:
            raise _SizeLimit
        return len(value)


def _large_json(value: Any, limit: int = MAX_TRACE_BYTES) -> bool:
    try:
        _write_json(cast(BinaryIO, _SizeCounter(limit)), value)
        return False
    except _SizeLimit:
        return True


def _parameters(params: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in params.items():
        if value is None:
            continue
        pointer = str(key).replace("~", "~0").replace("/", "~1")
        if _large_json(value, 6000):
            result[key] = f"artifact://record.json#/params/{pointer}"
        else:
            result[key] = json.dumps(value, ensure_ascii=False) \
                if isinstance(value, (dict, list)) else value
    return result


def redact(value: Any, *, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            redact(str(key), secrets=secrets): (
                "[REDACTED]" if _SECRET_KEY.search(str(key).replace("-", "_"))
                else redact(item, secrets=secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, secrets=secrets) for item in value]
    if not isinstance(value, str):
        return value
    for secret in secrets:
        if secret:
            pattern = "".join(
                f"(?:{re.escape(char)}|\\\\u{ord(char):04x}|%{ord(char):02x})" for char in secret
            )
            value = re.sub(pattern, "[REDACTED]", value, flags=re.IGNORECASE)
    value = value.replace("\\/", "/")
    value = re.sub(r"(/runs/[^/\s]+/)[^/\s]+(/v1)", r"\1[REDACTED]\2", value)
    value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[^\s\"',;]+", "[REDACTED]", value)
    value = re.sub(r"(https?://)[^/\s@]+@", r"\1[REDACTED]@", value)
    value = re.sub(
        r"(?i)(\b(?:api[_-]?key|access_token|password|authorization|secret)[\"']?"
        r"\s*[=:]\s*[\"']?)[^\s&,;\"']+", r"\1[REDACTED]", value,
    )
    return value


def sanitize_record(record: dict[str, Any], *, operator: bool = False) -> dict[str, Any]:
    split = record.get("split")
    if split not in {"train", "public", "private"}:
        raise ValueError("Unknown tracking split")
    if record.get("status") not in {"FINISHED", "FAILED"}:
        raise ValueError("Only completed records can be exported")
    if split == "private" and not operator:
        raise ValueError("Private records require operator export")
    if not isinstance(record.get("params"), dict) or not isinstance(record.get("metrics"), dict):
        raise ValueError("Tracking record requires params and metrics")
    if split == "public" and not operator:
        score = record["metrics"].get("score")
        if score is not None and (type(score) not in {int, float} or not math.isfinite(score)):
            raise ValueError("Public score must be a finite numeric scalar")
        record = {
            "split": "public", "status": record["status"],
            "params": {key: value for key, value in record["params"].items()
                       if key in _PUBLIC_PARAMS},
            "metrics": {"score": record["metrics"]["score"]}
            if record["metrics"].get("score") is not None else {},
        }
    if record.get("calls") and record.get("call_files"):
        raise ValueError("Tracking record must use either calls or call_files")
    return redact(record, secrets=_environment_secrets())


def _environment_secrets() -> tuple[str, ...]:
    with _EXPORT_LOCK:
        return tuple(
            value for key, value in os.environ.items()
            if _SECRET_KEY.search(key) and len(value) >= 8
        )


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
    try:
        temporary.touch(mode=0o600, exist_ok=False)
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _destination(operator: bool) -> str:
    with _EXPORT_LOCK:
        shared = os.getenv("MLFLOW_TRACKING_URI", "").strip().rstrip("/")
        if not operator:
            if not shared:
                raise ValueError("Configure the shared tracking destination explicitly")
            _validate_uri(shared)
            return shared
        target = os.getenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "").strip().rstrip("/")
        if not target:
            raise ValueError("Configure the operator tracking destination explicitly")
        if target == shared:
            raise ValueError("The operator tracking destination must be distinct")
        _validate_uri(target)
        return target


def _validate_uri(uri: str) -> None:
    parts = urlsplit(uri)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Tracking requires an HTTP(S) service")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Tracking URL must not contain credentials or query parameters")


@contextmanager
def _credentials(operator: bool):
    names = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN")
    previous = {key: os.environ.get(key) for key in names}
    try:
        if operator:
            for key in names:
                value = os.getenv(f"ARENA_OPERATOR_{key}")
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _tracking_uri(mlflow: Any, uri: str):
    previous_uri = mlflow.get_tracking_uri()
    previous_environment = {
        key: os.environ.get(key)
        for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACE_EXTRACT_ATTACHMENTS")
    }
    try:
        os.environ["MLFLOW_TRACE_EXTRACT_ATTACHMENTS"] = "false"
        mlflow.set_tracking_uri(uri)
        yield
    finally:
        mlflow.set_tracking_uri(previous_uri)
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _publish(record: dict[str, Any], uri: str, export_id: str, operator: bool) -> str:
    if record["split"] != "train" and not operator:
        raise ValueError("Public/private records require operator export")
    mlflow = importlib.import_module("mlflow")
    secrets = _environment_secrets()
    with _EXPORT_LOCK, _credentials(operator), _tracking_uri(mlflow, uri):
        client = mlflow.MlflowClient(tracking_uri=uri)
        name = f"arena-{'operator' if operator else record['split']}/{record['params']['case_id']}"
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        existing = client.search_runs(
            [experiment_id], filter_string=f"tags.arena_export_id = '{export_id}'", max_results=1,
        )
        if existing and existing[0].data.tags.get("arena_export_complete") == "true":
            return existing[0].info.run_id
        run_id = existing[0].info.run_id if existing else None
        with mlflow.start_run(
            run_id=run_id, experiment_id=experiment_id,
            run_name=str(record["params"]["candidate_id"]),
            tags={"arena_export_id": export_id} if run_id is None else None,
        ) as run:
            run_id = run.info.run_id
            mlflow.set_tags({
                "split": record["split"], "campaign_id": record["params"]["campaign_id"],
                "arena_export_id": export_id,
                "arena_visibility": "operator" if operator else "shared",
                "arena_execution_status": record["status"], "arena_export_complete": "false",
            }, synchronous=True)
            params = _parameters(record["params"])
            existing_params = getattr(existing[0].data, "params", {}) if existing else {}
            mlflow.log_params({
                key: value for key, value in params.items()
                if not (key in existing_params and isinstance(value, str)
                        and value.startswith("artifact://record.json#/params/"))
            }, synchronous=True)
            mlflow.log_metrics({
                key: value for key, value in record["metrics"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }, synchronous=True)
            artifact = {key: value for key, value in record.items()
                        if key not in {"call_files", "artifact_files"}}
            if record.get("artifact_files"):
                artifact["artifacts"] = {
                    key: f"fields/{Path(path).name}"
                    for key, path in record["artifact_files"].items()
                }
                _publish_artifact_files(mlflow, client, run_id, record, secrets)
            if record.get("call_files"):
                artifact["call_artifacts"] = [
                    f"calls/{Path(path).name}" for path in record["call_files"]
                ]
            log_json_artifact(mlflow, client, run_id, artifact, "record.json")
            _publish_call_files(mlflow, client, record, experiment_id, run_id, export_id, secrets)
            if record.get("calls"):
                if len(record["calls"]) > MAX_TRACE_CALLS or _large_json({
                    key: record[key] for key in ("calls", "params", "metrics")
                }):
                    for index, call in enumerate(_ordered_calls(record["calls"])):
                        _publish_call(
                            mlflow, client, record, experiment_id, run_id, export_id, secrets,
                            index, call, f"calls/inline-{index:08d}.json",
                        )
                else:
                    traces = client.search_traces(
                        locations=[experiment_id], run_id=run_id, max_results=1,
                    )
                    if not traces:
                        _spans(mlflow, record, run_id)
                        if not client.search_traces(
                            locations=[experiment_id], run_id=run_id, max_results=1,
                        ):
                            raise RuntimeError(
                                "Exported trace is not available from tracking server",
                            )
        client.set_terminated(run_id, status=record["status"])
        client.set_tag(run_id, "arena_export_complete", "true", synchronous=True)
        return run_id


def _publish_call_files(
    mlflow: Any, client: Any, record: dict[str, Any], experiment_id: str, run_id: str,
    export_id: str, secrets: tuple[str, ...],
) -> None:
    contexts = {
        record.get("call_sources", {}).get(item["path"], f"calls/{Path(item['path']).name}"):
        item.get("attributes", {})
        for item in record.get("call_refs", [])
    }
    for index, filename in enumerate(record.get("call_files", [])):
        path = Path(filename)
        attributes = contexts.get(f"calls/{path.name}", {})
        if path.stat().st_size > MAX_TRACE_BYTES:
            _publish_large_call_file(
                mlflow, client, record, experiment_id, run_id, export_id, secrets, index, path,
                attributes,
            )
            continue
        call = redact(json.loads(path.read_text(encoding="utf-8")), secrets=secrets)
        if attributes:
            call["attributes"] = {**call.get("attributes", {}), **attributes}
            if attributes.get("run_id"):
                call["attributes"]["model_run_id"] = attributes["run_id"]
        _publish_call(
            mlflow, client, record, experiment_id, run_id, export_id, secrets,
            index, call, f"calls/{path.name}",
        )


def _files_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while chunk := left.read(65536):
            if right.read(65536) != chunk:
                return False
    return True


def _publish_large_call_file(
    mlflow: Any, client: Any, record: dict[str, Any], experiment_id: str, run_id: str,
    export_id: str, secrets: tuple[str, ...], index: int, source: Path,
    context: dict[str, Any],
) -> None:
    artifact_path = f"calls/{source.name}"
    call_id = hashlib.sha256(f"{export_id}:{index}".encode()).hexdigest()
    query = {"locations": [experiment_id], "run_id": run_id, "max_results": 1,
             "filter_string": f"tag.arena_call_id = '{call_id}' AND "
             "tag.arena_payload_format = 'json-artifact-v1'", "include_spans": True}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
        path = Path(temporary) / source.name
        rewrite_json(
            source, path, transform=lambda item: redact(item, secrets=secrets),
            secret_key=lambda key: bool(_SECRET_KEY.search(key.replace("-", "_"))),
        )
        call = read_fields(path, {"name", "span_id", "parent_id", "span_type", "status",
                                 "start_time_ns", "end_time_ns"})
        try:
            role = read_fields(path, {"role"}, prefix="attributes").get("role")
        except ValueError:
            role = None
        role = record["params"].get("role", context.get("role", role))
        attributes = {**context, "run_id": run_id, "arena_artifact": artifact_path,
                      "arena_payload_format": "json-artifact-v1"}
        if context.get("run_id"):
            attributes["model_run_id"] = context["run_id"]
        for key, value in (("source_span_id", call.get("span_id")),
                           ("source_parent_id", call.get("parent_id")), ("role", role)):
            if value is not None:
                attributes[key] = value
        reference = {"arena_artifact": artifact_path}
        payload = {"name": call["name"], "inputs": reference, "outputs": reference,
                   "attributes": attributes}
        expected = {"payload": payload, "call_file": path, "artifact_path": artifact_path,
                    "run_id": run_id}
        if _call_trace_available(
            client, query, payload["name"], secrets=secrets, expected=expected,
        ):
            return
        log_json_file_artifact(mlflow, client, run_id, path, artifact_path)
        span = mlflow.start_span_no_context(
            payload["name"], span_type=call.get("span_type", "UNKNOWN"), inputs=reference,
            attributes=attributes, tags={"arena_call_id": call_id,
                                        "arena_payload_format": "json-artifact-v1"},
            experiment_id=experiment_id,
            start_time_ns=call.get("start_time_ns", record["start_time_ns"]),
        )
        span.end(outputs=reference, status=call.get("status", "OK"),
                 end_time_ns=call.get("end_time_ns", record["end_time_ns"]))
        mlflow.flush_trace_async_logging()
        if not _call_trace_available(
            client, query, payload["name"], secrets=secrets, expected=expected,
        ):
            raise RuntimeError("Exported call trace is not available from tracking server")


def _call_payload(call: dict[str, Any], run_id: str, artifact_path: str) -> dict[str, Any]:
    attributes = {
        **call.get("attributes", {}), "run_id": run_id,
        "source_span_id": call.get("span_id"), "source_parent_id": call.get("parent_id"),
    }
    payload = {"name": call["name"], "inputs": call.get("inputs"),
               "outputs": call.get("outputs"), "attributes": attributes}
    if not _large_json(payload):
        return payload
    reference = {"arena_artifact": artifact_path}
    attributes = {"run_id": run_id, "arena_artifact": artifact_path,
                  "arena_payload_format": "json-artifact-v1"}
    for key, value, pointer in (
        ("source_span_id", call.get("span_id"), "span_id"),
        ("source_parent_id", call.get("parent_id"), "parent_id"),
        ("role", call.get("attributes", {}).get("role"), "attributes/role"),
    ):
        if value is not None:
            attributes[key] = f"artifact://{artifact_path}#/{pointer}" \
                if _large_json(value, MAX_TRACE_BYTES // 16) else value
    name = call["name"]
    if _large_json(name, MAX_TRACE_BYTES // 16):
        name = "artifact_call"
        attributes["source_name_artifact"] = f"artifact://{artifact_path}#/name"
    return {"name": name, "inputs": reference, "outputs": reference, "attributes": attributes}


def _publish_call(
    mlflow: Any, client: Any, record: dict[str, Any], experiment_id: str, run_id: str,
    export_id: str, secrets: tuple[str, ...], index: int, call: dict[str, Any], artifact_path: str,
) -> None:
    call_id = hashlib.sha256(f"{export_id}:{index}".encode()).hexdigest()
    query = {"locations": [experiment_id], "run_id": run_id, "max_results": 1,
             "filter_string": f"tag.arena_call_id = '{call_id}'", "include_spans": True}
    payload = _call_payload(call, run_id, artifact_path)
    tags = {"arena_call_id": call_id}
    if payload["attributes"].get("arena_payload_format") == "json-artifact-v1":
        query["filter_string"] += " AND tag.arena_payload_format = 'json-artifact-v1'"
        tags["arena_payload_format"] = "json-artifact-v1"
    expected = {"payload": payload, "call": call, "artifact_path": artifact_path, "run_id": run_id}
    if _call_trace_available(client, query, payload["name"], secrets=secrets, expected=expected):
        return
    log_json_artifact(mlflow, client, run_id, call, artifact_path)
    span = mlflow.start_span_no_context(
        payload["name"], span_type=call.get("span_type", "UNKNOWN"),
        inputs=payload["inputs"], attributes=payload["attributes"],
        tags=tags, experiment_id=experiment_id,
        start_time_ns=call.get("start_time_ns", record["start_time_ns"]),
    )
    span.end(
        outputs=payload["outputs"], status=call.get("status", "OK"),
        end_time_ns=call.get("end_time_ns", record["end_time_ns"]),
    )
    mlflow.flush_trace_async_logging()
    if not _call_trace_available(
        client, query, payload["name"], secrets=secrets, expected=expected,
    ):
        raise RuntimeError("Exported call trace is not available from tracking server")


def _span_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        if isinstance(decoded, (str, dict, list)):
            return decoded
    return value


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(first, second) for first, second in zip(left, right, strict=True)
        )
    return left == right


def _call_trace_available(
    client: Any, query: dict[str, Any], name: str, *, secrets: tuple[str, ...] = (),
    expected: dict[str, Any] | None = None,
) -> bool:
    payload = expected["payload"] if expected is not None else {}
    normalized = {
        key: _span_value(payload[key]) for key in ("inputs", "outputs")
    } if expected is not None else {}
    normalized_attributes = {
        key: _span_value(value) for key, value in payload.get("attributes", {}).items()
    }
    artifact_required = (
        any(not _json_equal(value, payload[key]) for key, value in normalized.items())
        or not _json_equal(normalized_attributes, payload.get("attributes", {}))
        or payload.get("attributes", {}).get("arena_payload_format") == "json-artifact-v1"
    )
    while True:
        traces = client.search_traces(**query)
        for trace in traces:
            if len(trace.data.spans) != 1:
                continue
            span = trace.data.spans[0]
            if redact(span.name, secrets=secrets) != name:
                continue
            if expected is not None:
                received = {
                    key: redact(getattr(span, key, None), secrets=secrets) for key in normalized
                }
                if any(not (
                    _json_equal(received[key], payload[key])
                    or _json_equal(received[key], redact(value, secrets=secrets))
                ) for key, value in normalized.items()):
                    continue
                attributes = redact(getattr(span, "attributes", {}), secrets=secrets)
                if any(not (
                    _json_equal(attributes.get(key), payload["attributes"][key])
                    or _json_equal(attributes.get(key), redact(value, secrets=secrets))
                ) for key, value in normalized_attributes.items() if value is not None):
                    continue
                if artifact_required:
                    if "call_file" in expected:
                        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
                            stored_path = download_json_artifact(
                                client, expected["run_id"], expected["artifact_path"],
                                Path(temporary) / "received.json",
                            )
                            if not _files_equal(stored_path, expected["call_file"]):
                                continue
                        return True
                    stored = read_json_artifact(
                        client, expected["run_id"], expected["artifact_path"],
                    )
                    if not _json_equal(redact(stored, secrets=secrets), expected["call"]):
                        continue
            return True
        token = getattr(traces, "token", None)
        if not token:
            return False
        query = {**query, "page_token": token}


def _digest(value: Any) -> str:
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=False).iterencode(value):
        digest.update(chunk.encode())
    return digest.hexdigest()


def _snapshot_calls(record: dict[str, Any], directory: Path) -> None:
    paths = []
    sources = {}
    secrets = _environment_secrets()
    for filename in record.get("call_files", []):
        source = Path(filename)
        if not source.is_absolute():
            raise ValueError("External call paths must be absolute")
        if source.stat().st_size > MAX_TRACE_BYTES:
            path = directory / "calls" / f"{_record_identity(source, 'call', False)}.json"
            if not path.exists():
                rewrite_json(
                    source, path, transform=lambda item: redact(item, secrets=secrets),
                    secret_key=lambda key: bool(_SECRET_KEY.search(key.replace("-", "_"))),
                )
                _reuse_payload(source, path)
            paths.append(str(path.resolve()))
            sources[str(source)] = f"calls/{path.name}"
            continue
        call = json.loads(source.read_text(encoding="utf-8"))
        if record["params"].get("role"):
            call["attributes"] = {
                **call.get("attributes", {}), "role": record["params"]["role"],
            }
        call = redact(call, secrets=secrets)
        path = directory / "calls" / f"{_digest(call)}.json"
        if not path.exists():
            _save(path, call)
            _reuse_payload(source, path)
        paths.append(str(path.resolve()))
        sources[str(source)] = f"calls/{path.name}"
    if "call_files" in record:
        record["call_files"] = paths
        record["call_sources"] = sources


def _reuse_payload(source: Path, target: Path) -> None:
    if _files_equal(source, target):
        linked = target.with_suffix(".link")
        try:
            with source.open("rb") as input_stream, linked.open("xb") as output:
                os.chmod(linked, 0o600)
                fcntl.ioctl(output.fileno(), 0x40049409, input_stream.fileno())
                output.flush()
                os.fsync(output.fileno())
            linked.replace(target)
        except OSError:
            linked.unlink(missing_ok=True)


def _ordered_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = [{**call, "span_id": call.get("span_id", f"call-{index}")}
               for index, call in enumerate(calls)]
    identifiers = {call["span_id"] for call in pending}
    if len(identifiers) != len(pending):
        raise ValueError("Duplicate span ID")
    if any(call.get("parent_id") not in identifiers | {None} for call in pending):
        raise ValueError("Unknown parent span")
    ordered = []
    completed = {None}
    while pending:
        ready = [call for call in pending if call.get("parent_id") in completed]
        if not ready:
            raise ValueError("Cyclic span hierarchy")
        ordered.extend(ready)
        completed.update(call["span_id"] for call in ready)
        pending = [call for call in pending if call["span_id"] not in completed]
    return ordered


def _spans(mlflow: Any, record: dict[str, Any], run_id: str) -> None:
    calls = _ordered_calls(record["calls"])
    root = mlflow.start_span_no_context(
        f"{record['split']}_evaluation", inputs=record["params"],
        attributes={"run_id": run_id}, start_time_ns=record["start_time_ns"],
    )
    spans = {}
    for call in calls:
        attributes = call.get("attributes", {})
        span = mlflow.start_span_no_context(
            call["name"], parent_span=spans.get(call.get("parent_id"), root),
            span_type=call.get("span_type", "UNKNOWN"), inputs=call.get("inputs"),
            attributes=attributes, start_time_ns=call.get("start_time_ns", record["start_time_ns"]),
        )
        spans[call["span_id"]] = span
    for call in reversed(calls):
        spans[call["span_id"]].end(
            outputs=call.get("outputs"), status=call.get("status", "OK"),
            end_time_ns=call.get("end_time_ns", record["end_time_ns"]),
        )
    root.end(
        outputs=record["metrics"], status="OK" if record["status"] == "FINISHED" else "ERROR",
        end_time_ns=record["end_time_ns"],
    )
    mlflow.flush_trace_async_logging()


def _publish_artifact_files(
    mlflow: Any, client: Any, run_id: str, record: dict[str, Any], secrets: tuple[str, ...],
) -> None:
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
        for filename in record.get("artifact_files", {}).values():
            source = Path(filename)
            target = Path(temporary) / source.name
            rewrite_json(
                source, target, transform=lambda item: redact(item, secrets=secrets),
                secret_key=lambda key: bool(_SECRET_KEY.search(key.replace("-", "_"))),
            )
            log_json_file_artifact(mlflow, client, run_id, target, f"fields/{source.name}")


def _prepare_record(
    source: Path, directory: Path, *, operator: bool, prefix: str = "",
) -> Path:
    metadata = read_fields(source, {"split", "status", "params", "metrics"}, prefix=prefix)
    safe = sanitize_record(metadata, operator=operator)
    if safe["split"] != "train" and not operator:
        raise ValueError("Public/private records require operator export")
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    record_path = directory / "record.json"
    secrets = _environment_secrets()
    fields = split_object(
        source, directory / "fields", prefix=prefix,
        transform=lambda item: redact(item, secrets=secrets),
        secret_key=lambda key: bool(_SECRET_KEY.search(key.replace("-", "_"))),
    )
    record = {}
    inline_paths = []
    for key, path in fields.items():
        if key == "calls" and path.stat().st_size > MAX_TRACE_BYTES:
            inline_paths = split_array(path, directory / "inline-calls")
            metadata = []
            for index, target in enumerate(inline_paths):
                item = read_fields(target, {"span_id", "parent_id"})
                if "span_id" not in item:
                    item["span_id"] = f"call-{index}"
                    with target.open("r+b") as stream:
                        stream.seek(-1, os.SEEK_END)
                        stream.write((", " + json.dumps("span_id") + ": "
                                      + json.dumps(item["span_id"]) + "}").encode())
                        stream.flush()
                        os.fsync(stream.fileno())
                metadata.append({**item, "path": str(target.resolve())})
            record["call_files"] = [item["path"] for item in _ordered_calls(metadata)]
            path.unlink()
        elif key in {"split", "status", "params", "metrics", "call_files", "call_refs"} \
                or path.stat().st_size <= MAX_TRACE_BYTES:
            value = json.loads(path.read_text(encoding="utf-8"))
            if key == "call_files" and record.get("call_files") and value:
                raise ValueError("Tracking record must use either calls or call_files")
            if key != "call_files" or value or "call_files" not in record:
                record[key] = value
            path.unlink()
        else:
            record.setdefault("artifact_files", {})[key] = str(path.resolve())
    record = sanitize_record(record, operator=operator)
    _snapshot_calls(record, directory)
    for path in inline_paths:
        path.unlink(missing_ok=True)
    _save(record_path, record)
    return record_path


def _read_entry(path: Path, *, migrate: bool = True) -> dict[str, Any]:
    fields = {"uri", "operator", "export_id", "status", "run_id", "attempts", "error",
              "record_file", "legacy_file", "created_at_ns", "updated_at_ns"}
    entry = read_fields(path, fields)
    if "record_file" in entry or "legacy_file" in entry or not migrate:
        return entry
    legacy = path.parent / "legacy" / path.name
    legacy.parent.mkdir(mode=0o700, exist_ok=True)
    if not legacy.exists():
        os.link(path, legacy)
    entry["legacy_file"] = str(legacy.resolve())
    if entry.get("status") != "exported":
        record_path = _prepare_record(
            legacy, path.parent / "records" / path.stem, operator=entry["operator"],
            prefix="record",
        )
        entry["record_file"] = str(record_path.resolve())
    _save(path, entry)
    return entry


def _export(path: Path, entry: dict[str, Any]) -> dict[str, str]:
    if entry["status"] == "exported":
        return {"status": "exported", "run_id": entry["run_id"]}
    try:
        record = sanitize_record(
            json.loads(Path(entry["record_file"]).read_text(encoding="utf-8")),
            operator=entry["operator"],
        )
        run_id = _publish(record, entry["uri"], entry["export_id"], entry["operator"])
    except Exception as error:
        entry.update(
            status="tracking_pending", error=redact(str(error), secrets=_environment_secrets()),
            attempts=entry.get("attempts", 0) + 1, updated_at_ns=time.time_ns(),
        )
        _save(path, entry)
        return {"status": "tracking_pending"}
    entry.update(status="exported", run_id=run_id, updated_at_ns=time.time_ns())
    entry.pop("error", None)
    _save(path, entry)
    return {"status": "exported", "run_id": run_id}


def _record_identity(path: Path, uri: str, operator: bool) -> str:
    digest = hashlib.sha256(json.dumps([uri, operator]).encode())
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def export_record(
    record_path: Path, queue_dir: Path | None = None, *,
    operator: bool = False, defer: bool = False,
) -> dict[str, str]:
    uri = _destination(operator)
    if read_fields(record_path, {"split"}).get("split") != "train" and not operator:
        raise ValueError("Public/private records require operator export")
    directory = queue_dir if queue_dir is not None else record_path.parent / ".tracking"
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    export_id = _record_identity(record_path, uri, operator)
    path = directory / f"{export_id}.json"
    with (directory / f"{export_id}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            entry = _read_entry(path)
        else:
            payload = _prepare_record(
                record_path, directory / "records" / export_id, operator=operator,
            )
            now = time.time_ns()
            entry = {
                "uri": uri, "operator": operator, "record_file": str(payload.resolve()),
                "export_id": export_id, "status": "tracking_pending",
                "created_at_ns": now, "updated_at_ns": now,
            }
            _save(path, entry)
        if defer and entry["status"] != "exported":
            return {"status": "tracking_pending"}
        return _export(path, entry)


def export_configured(
    record_path: Path, queue_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    with _EXPORT_LOCK:
        delivery = os.getenv("ARENA_TRACKING_DELIVERY", "inline")
        if delivery not in {"inline", "queued"}:
            raise ValueError("Tracking delivery must be inline or queued")
        configured = {
            "shared": bool(os.getenv("MLFLOW_TRACKING_URI", "").strip()),
            "operator": bool(os.getenv("ARENA_OPERATOR_MLFLOW_TRACKING_URI", "").strip()),
        }
    split = read_fields(record_path, {"split"}).get("split")
    if split not in {"train", "public", "private"}:
        raise ValueError("Unknown tracking split")
    directory = queue_dir if queue_dir is not None else record_path.parent / ".tracking"
    if split == "private" and configured["shared"] and not configured["operator"]:
        raise ValueError("Private export requires the operator tracking destination")
    results = {}
    for name, operator in (("shared", False), ("operator", True)):
        if configured[name] and (split == "train" or operator):
            results[name] = export_record(
                record_path, directory / name, operator=operator, defer=delivery == "queued",
            )
    return results


def flush_queue(
    directory: Path, *, operator: bool = False, continue_on_error: bool = False,
) -> list[dict[str, str]]:
    results = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.with_suffix(".lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                entry = _read_entry(path, migrate=False)
                if entry["operator"] != operator:
                    raise ValueError("Queue requires matching operator mode")
                if entry["uri"].rstrip("/") != _destination(operator):
                    raise ValueError("Queue destination differs from configured destination")
                entry = _read_entry(path)
                if entry["status"] != "exported":
                    results.append(_export(path, entry))
        except Exception as error:
            if not continue_on_error:
                raise
            results.append({"status": "error", "error_type": type(error).__name__})
    return results


def queue_status(directory: Path, *, operator: bool = False) -> dict[str, Any]:
    result = {"total": 0, "exported": 0, "pending": 0, "failed": 0,
              "oldest_pending_at_ns": None}
    pending_times = []
    for path in directory.glob("*.json"):
        result["total"] += 1
        try:
            entry = _read_entry(path, migrate=False)
            if entry["operator"] != operator:
                raise ValueError("Queue requires matching operator mode")
            if entry["status"] == "exported":
                result["exported"] += 1
            else:
                result["pending"] += 1
                result["failed"] += bool(entry.get("error"))
                if entry.get("created_at_ns") is not None:
                    pending_times.append(entry["created_at_ns"])
        except (OSError, ValueError, KeyError):
            result["failed"] += 1
    result["oldest_pending_at_ns"] = min(pending_times) if pending_times else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("record", type=Path)
    export.add_argument("--queue", type=Path)
    export.add_argument("--operator", action="store_true")
    flush = commands.add_parser("flush")
    flush.add_argument("--queue", type=Path, required=True)
    flush.add_argument("--operator", action="store_true")
    args = parser.parse_args()
    result = export_record(args.record, args.queue, operator=args.operator) \
        if args.command == "export" else flush_queue(args.queue, operator=args.operator)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
