import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

_FORMAT = "arena.examples.v1"


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("checkpoint directories cannot be symlinks")
    if path.exists():
        if not path.is_dir():
            raise ValueError("checkpoint directory must be a directory")
        return
    _directory(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("checkpoint directory must be a directory") from None
    _sync_directory(path.parent)


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint entries must be regular files")
    with path.open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("checkpoint entry must be an object")
    return value


def _write(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    _directory(path.parent)
    if path.is_symlink():
        raise ValueError("checkpoint entries cannot be symlinks")
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".pending-", delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(value, output, ensure_ascii=False, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
            if immutable:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if _read(path) != value:
                        raise ValueError("checkpoint stage is immutable") from None
            else:
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    _sync_directory(path.parent)


def _store(path: Path) -> Path:
    return path.with_name(path.stem + ".examples")


def _validate_ids(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("checkpoint order requires unique nonblank example IDs")
    return value


def _order(path: Path) -> list[str]:
    directory = _store(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint example storage is missing or invalid")
    order = _read(directory / "order.json")
    if set(order) != {"example_ids"}:
        raise ValueError("invalid checkpoint order")
    return _validate_ids(order["example_ids"])


def write_metadata(
    path: Path, metadata: dict[str, Any], *, example_ids: list[str] | None = None,
) -> None:
    if "examples" in metadata:
        raise ValueError("checkpoint metadata cannot contain examples")
    order_path = _store(path) / "order.json"
    if example_ids is not None:
        ids = _validate_ids(example_ids)
        _write(order_path, {"example_ids": ids}, immutable=True)
    else:
        _order(path)
    if path.exists():
        previous = _read(path)
        if "checkpoint_format" not in previous:
            positions = {identifier: index for index, identifier in enumerate(_order(path))}
            for detail in previous.get("examples", []):
                if not isinstance(detail, dict) or detail.get("example_id") not in positions:
                    raise ValueError("legacy checkpoint differs from frozen order")
                write_example(path, positions[detail["example_id"]], detail)
    _write(path, {"checkpoint_format": _FORMAT, "metadata": metadata})


def _update(previous: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    if previous.get("example_id") != detail.get("example_id"):
        raise ValueError("checkpoint example ID changed")
    if "answer" in previous and previous["answer"] != detail.get("answer"):
        raise ValueError("checkpoint solver answer is immutable")
    previous_resources = previous.get("resources", {})
    resources = detail.get("resources", {})
    if not isinstance(previous_resources, dict) or not isinstance(resources, dict):
        raise ValueError("checkpoint resources must be an object")
    if "solver" in previous_resources and previous_resources["solver"] != resources.get("solver"):
        raise ValueError("checkpoint solver resources are immutable")
    previous_calls = previous.get("calls", [])
    calls = detail.get("calls", [])
    if (
        not isinstance(previous_calls, list) or not isinstance(calls, list)
        or calls[:len(previous_calls)] != previous_calls
    ):
        raise ValueError("checkpoint calls must preserve saved history")
    changed = {
        key: value for key, value in detail.items()
        if key not in {"resources", "calls"} and (key not in previous or previous[key] != value)
    }
    removed = [key for key in previous if key not in detail and key not in {"resources", "calls"}]
    changed_resources = {
        key: value for key, value in resources.items()
        if key not in previous_resources or previous_resources[key] != value
    }
    removed_resources = [key for key in previous_resources if key not in resources]
    return {
        "set": changed, "remove": removed, "resources": changed_resources,
        "remove_resources": removed_resources, "calls": calls[len(previous_calls):],
    }


def _apply(previous: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    if (
        set(update) != {"set", "remove", "resources", "remove_resources", "calls"}
        or not isinstance(update["set"], dict) or not isinstance(update["resources"], dict)
        or any(not isinstance(update[key], list) for key in ("remove", "remove_resources", "calls"))
        or any(not isinstance(key, str) for key in [*update["remove"], *update["remove_resources"]])
        or set(update["set"]) & {"resources", "calls"}
    ):
        raise ValueError("invalid checkpoint stage update")
    result = deepcopy(previous)
    for key in update["remove"]:
        result.pop(key, None)
    result.update(update["set"])
    if update["resources"] or update["remove_resources"]:
        resources = result.setdefault("resources", {})
        for key in update["remove_resources"]:
            resources.pop(key, None)
        resources.update(update["resources"])
    if update["calls"]:
        result.setdefault("calls", []).extend(update["calls"])
    _update(previous, result)
    return result


def _stages(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint example entry must be a directory")
    paths = sorted(item for item in directory.iterdir() if not item.name.startswith(".pending-"))
    for index, item in enumerate(paths):
        if item.name != f"stage-{index:06d}.json" or item.is_symlink() or not item.is_file():
            raise ValueError("invalid checkpoint stage inventory")
    return paths


def _example(directory: Path, example_id: str) -> tuple[dict[str, Any], int]:
    paths = _stages(directory)
    if not paths:
        raise ValueError("checkpoint example has no committed stage")
    first = _read(paths[0])
    if set(first) != {"detail"} or not isinstance(first["detail"], dict):
        raise ValueError("invalid initial checkpoint stage")
    detail = first["detail"]
    if detail.get("example_id") != example_id:
        raise ValueError("checkpoint example differs from frozen order")
    for path in paths[1:]:
        stage = _read(path)
        if set(stage) != {"update"} or not isinstance(stage["update"], dict):
            raise ValueError("invalid checkpoint stage")
        detail = _apply(detail, stage["update"])
    return detail, len(paths)


def write_example(path: Path, index: int, detail: dict[str, Any]) -> None:
    ids = _order(path)
    if (
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(ids)
        or detail.get("example_id") != ids[index]
    ):
        raise ValueError("checkpoint example differs from frozen order")
    directory = _store(path) / str(index)
    if directory.exists() or directory.is_symlink():
        paths = _stages(directory)
        if paths:
            previous, revision = _example(directory, ids[index])
            if previous == detail:
                return
            stage = {"update": _update(previous, detail)}
        else:
            revision, stage = 0, {"detail": detail}
    else:
        revision, stage = 0, {"detail": detail}
    _write(directory / f"stage-{revision:06d}.json", stage, immutable=True)


def read_document(path: Path) -> dict[str, Any]:
    document = _read(path)
    if "checkpoint_format" not in document:
        return document
    if (
        set(document) != {"checkpoint_format", "metadata"}
        or document["checkpoint_format"] != _FORMAT
        or not isinstance(document["metadata"], dict)
        or "examples" in document["metadata"]
    ):
        raise ValueError("invalid checkpoint manifest")
    ids = _order(path)
    available = {}
    for item in _store(path).iterdir():
        if item.name == "order.json" or item.name.startswith(".pending-"):
            continue
        if not item.name.isascii() or not item.name.isdigit():
            raise ValueError("unknown checkpoint example entry")
        index = int(item.name)
        if str(index) != item.name or not 0 <= index < len(ids):
            raise ValueError("checkpoint example index is outside frozen order")
        if _stages(item):
            detail, _ = _example(item, ids[index])
            available[index] = detail
    return {
        **document["metadata"],
        "examples": [available[index] for index in sorted(available)],
    }


def materialize_document(path: Path) -> None:
    _write(path, read_document(path))
