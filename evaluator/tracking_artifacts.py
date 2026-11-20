import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from uuid import uuid4

MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
_STRING_CHARS = 32 * 1024
_FORMAT = "arena_json_parts"


def _artifact_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or path.as_posix() != value or "\\" in value
            or ".." in path.parts or path.suffix != ".json"):
        raise ValueError("JSON artifact path must be canonical and relative")
    return path


def _write_json(stream: BinaryIO, value: Any) -> None:
    active: set[int] = set()

    def string(text: str) -> None:
        stream.write(b'"')
        for start in range(0, len(text), _STRING_CHARS):
            stream.write(json.dumps(text[start:start + _STRING_CHARS])[1:-1].encode("ascii"))
        stream.write(b'"')

    def key(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None or isinstance(value, (bool, int, float)):
            return json.dumps(value)
        raise TypeError("JSON object keys must be strings or scalar values")

    def write(item: Any, depth: int) -> None:
        if isinstance(item, str):
            string(item)
        elif isinstance(item, (dict, list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("Circular reference detected")
            active.add(identity)
            mapping = isinstance(item, dict)
            stream.write(b"{" if mapping else b"[")
            for index, entry in enumerate(item.items() if mapping else item):
                stream.write((b",\n" if index else b"\n") + b"  " * (depth + 1))
                if mapping:
                    name, child = entry
                    string(key(name))
                    stream.write(b": ")
                else:
                    child = entry
                write(child, depth + 1)
            if item:
                stream.write(b"\n" + b"  " * depth)
            stream.write(b"}" if mapping else b"]")
            active.remove(identity)
        elif item is None or isinstance(item, (bool, int, float)):
            stream.write(json.dumps(item).encode("ascii"))
        else:
            string(str(item))

    write(value, 0)


def _part_path(prefix: str, index: int) -> str:
    return f"{prefix}/{index:08d}.bin"


def _download(client: Any, run_id: str, path: str, directory: Path) -> Path:
    return Path(client.download_artifacts(run_id, path, dst_path=str(directory)))


def log_json_artifact(
    mlflow: Any, client: Any, run_id: str, value: Any, artifact_path: str,
) -> None:
    _artifact_path(artifact_path)
    with tempfile.TemporaryDirectory(
        prefix="arena-json-", dir=os.environ.get("TMPDIR"),
    ) as temporary:
        directory = Path(temporary)
        source = directory / "source.json"
        with source.open("wb") as stream:
            _write_json(stream, value)
        total = source.stat().st_size
        if total <= MAX_ARTIFACT_BYTES and not (isinstance(value, dict) and _FORMAT in value):
            mlflow.log_dict(value, artifact_path)
            return
        log_json_file_artifact(mlflow, client, run_id, source, artifact_path)


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as source, right.open("rb") as received:
        while block := source.read(min(MAX_ARTIFACT_BYTES, 64 * 1024)):
            if received.read(len(block)) != block:
                return False
    return True


def log_json_file_artifact(
    mlflow: Any, client: Any, run_id: str, source: Path, artifact_path: str,
) -> None:
    target = _artifact_path(artifact_path)
    total = source.stat().st_size
    with tempfile.TemporaryDirectory(
        prefix="arena-json-file-", dir=os.environ.get("TMPDIR"),
    ) as temporary:
        directory = Path(temporary)
        if total <= MAX_ARTIFACT_BYTES:
            with source.open(encoding="utf-8") as stream:
                value = json.load(stream)
            if not (isinstance(value, dict) and _FORMAT in value):
                upload = directory / target.name
                shutil.copyfile(source, upload)
                parent = target.parent.as_posix()
                mlflow.log_artifact(str(upload), artifact_path=None if parent == "." else parent)
                with tempfile.TemporaryDirectory(dir=directory) as download:
                    received = _download(client, run_id, artifact_path, Path(download))
                    if not _same_bytes(source, received):
                        raise ValueError("Artifact read-back differs")
                return
        prefix = artifact_path + ".parts"
        manifest = {_FORMAT: {
            "version": 1, "part_prefix": prefix,
            "part_count": (total + MAX_ARTIFACT_BYTES - 1) // MAX_ARTIFACT_BYTES,
            "part_bytes": MAX_ARTIFACT_BYTES, "total_bytes": total,
        }}
        if len(json.dumps(manifest, indent=2).encode()) > MAX_ARTIFACT_BYTES:
            raise ValueError("Artifact manifest exceeds the transport limit")
        with source.open("rb") as stream:
            for index in range(manifest[_FORMAT]["part_count"]):
                part = directory / f"{index:08d}.bin"
                part.write_bytes(stream.read(MAX_ARTIFACT_BYTES))
                mlflow.log_artifact(str(part), artifact_path=prefix)
                with tempfile.TemporaryDirectory(dir=directory) as download:
                    received = _download(client, run_id, _part_path(prefix, index), Path(download))
                    if not _same_bytes(part, received):
                        raise ValueError("Artifact part read-back differs")
                part.unlink()
        mlflow.log_dict(manifest, artifact_path)
        with tempfile.TemporaryDirectory(dir=directory) as download:
            received = _download(client, run_id, artifact_path, Path(download))
            if received.stat().st_size > MAX_ARTIFACT_BYTES:
                raise ValueError("Artifact manifest read-back exceeds the transport limit")
            with received.open(encoding="utf-8") as stream:
                if json.load(stream) != manifest:
                    raise ValueError("Artifact manifest read-back differs")


def _manifest(value: dict[str, Any], artifact_path: str) -> dict[str, Any]:
    if set(value) != {_FORMAT} or not isinstance(value[_FORMAT], dict):
        raise ValueError("Invalid JSON artifact manifest")
    result = value[_FORMAT]
    if set(result) != {"version", "part_prefix", "part_count", "part_bytes", "total_bytes"}:
        raise ValueError("Invalid JSON artifact manifest fields")
    if type(result["version"]) is not int or result["version"] != 1:
        raise ValueError("Unknown JSON artifact manifest version")
    if result["part_prefix"] != artifact_path + ".parts":
        raise ValueError("Invalid JSON artifact part prefix")
    for field in ("part_count", "part_bytes", "total_bytes"):
        if type(result[field]) is not int or result[field] <= 0:
            raise ValueError("Invalid JSON artifact sizes")
    expected = (result["total_bytes"] + result["part_bytes"] - 1) // result["part_bytes"]
    if result["part_bytes"] > MAX_ARTIFACT_BYTES or result["part_count"] != expected:
        raise ValueError("Inconsistent JSON artifact sizes")
    return result


def download_json_artifact(
    client: Any, run_id: str, artifact_path: str, output: Path,
) -> Path:
    _artifact_path(artifact_path)
    pending = output.with_name(f".{output.name}.{uuid4()}.tmp")
    try:
        with tempfile.TemporaryDirectory(
            prefix="arena-json-read-", dir=os.environ.get("TMPDIR"),
        ) as temporary:
            directory = Path(temporary)
            source = _download(client, run_id, artifact_path, directory)
            value = None
            if source.stat().st_size <= MAX_ARTIFACT_BYTES:
                with source.open(encoding="utf-8") as stream:
                    value = json.load(stream)
            if not isinstance(value, dict) or _FORMAT not in value:
                shutil.copyfile(source, pending)
            else:
                manifest = _manifest(value, artifact_path)
                with pending.open("wb") as destination:
                    for index in range(manifest["part_count"]):
                        expected = min(
                            manifest["part_bytes"],
                            manifest["total_bytes"] - index * manifest["part_bytes"],
                        )
                        with tempfile.TemporaryDirectory(dir=directory) as download:
                            part = _download(
                                client, run_id, _part_path(manifest["part_prefix"], index),
                                Path(download),
                            )
                            if part.stat().st_size != expected:
                                raise ValueError("JSON artifact part size differs")
                            with part.open("rb") as stream:
                                shutil.copyfileobj(stream, destination, 64 * 1024)
                if pending.stat().st_size != manifest["total_bytes"]:
                    raise ValueError("JSON artifact total size differs")
            pending.replace(output)
            return output
    finally:
        pending.unlink(missing_ok=True)


def read_json_artifact(client: Any, run_id: str, artifact_path: str) -> Any:
    with tempfile.TemporaryDirectory(
        prefix="arena-json-value-", dir=os.environ.get("TMPDIR"),
    ) as temporary:
        output = Path(temporary) / "value.json"
        download_json_artifact(client, run_id, artifact_path, output)
        with output.open(encoding="utf-8") as stream:
            return json.load(stream)
