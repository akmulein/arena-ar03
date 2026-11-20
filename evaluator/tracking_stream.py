import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import ijson
from ijson.common import ObjectBuilder

Event = tuple[str, str, Any]


def _value_events(events: Iterator[Event], first: Event) -> Iterator[Event]:
    yield first
    depth = 1 if first[1] in {"start_map", "start_array"} else 0
    while depth:
        event = next(events)
        if event[1] in {"start_map", "start_array"}:
            depth += 1
        elif event[1] in {"end_map", "end_array"}:
            depth -= 1
        yield event


def _fields(events: Iterator[Event], prefix: str) -> Iterator[tuple[str, Event]]:
    for path, event, _ in events:
        if path == prefix:
            if event != "start_map":
                raise ValueError("Tracking fields must belong to an object")
            break
    else:
        raise ValueError("Tracking object was not found")
    seen = set()
    for _, event, key in events:
        if event == "end_map":
            break
        if event != "map_key" or key in seen:
            raise ValueError("Tracking object has invalid or duplicate fields")
        seen.add(key)
        yield key, next(events)
    for _ in events:
        pass


def read_fields(source: Path, names: set[str], *, prefix: str = "") -> dict[str, Any]:
    result = {}
    try:
        with source.open("rb") as stream:
            events = iter(ijson.parse(stream, use_float=True))
            for key, first in _fields(events, prefix):
                builder = ObjectBuilder() if key in names else None
                for _, event, value in _value_events(events, first):
                    if builder is not None:
                        builder.event(event, value)
                if builder is not None:
                    result[key] = builder.value
    except (ijson.JSONError, StopIteration) as error:
        raise ValueError("Invalid tracking JSON") from error
    return result


def _write_value(
    events: Iterator[Event], first: Event, output: Any,
    transform: Callable[[Any], Any] | None, secret_key: Callable[[str], bool] | None,
) -> None:
    event, value = first[1:]
    if event == "start_map":
        output.write("{")
        count = 0
        for _, event, key in events:
            if event == "end_map":
                break
            if event != "map_key":
                raise ValueError("Invalid tracking JSON object")
            if count:
                output.write(",")
            json.dump(transform(key) if transform is not None else key, output, ensure_ascii=False)
            output.write(":")
            child = next(events)
            if secret_key is not None and secret_key(key):
                for _ in _value_events(events, child):
                    pass
                output.write('"[REDACTED]"')
            else:
                _write_value(events, child, output, transform, secret_key)
            count += 1
        output.write("}")
    elif event == "start_array":
        output.write("[")
        count = 0
        for child in events:
            if child[1] == "end_array":
                break
            if count:
                output.write(",")
            _write_value(events, child, output, transform, secret_key)
            count += 1
        output.write("]")
    elif event in {"string", "number", "boolean", "null"}:
        json.dump(transform(value) if transform is not None else value, output, ensure_ascii=False)
    else:
        raise ValueError("Invalid tracking JSON value")


def split_object(
    source: Path, output_dir: Path, *, prefix: str = "",
    transform: Callable[[Any], Any] | None = None,
    secret_key: Callable[[str], bool] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    result = {}
    try:
        with source.open("rb") as stream:
            events = iter(ijson.parse(stream, use_float=True))
            for index, (key, first) in enumerate(_fields(events, prefix)):
                path = output_dir / f"{index:08d}.json"
                temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
                try:
                    temporary.touch(mode=0o600, exist_ok=False)
                    with temporary.open("w", encoding="utf-8") as output:
                        if secret_key is not None and secret_key(key):
                            for _ in _value_events(events, first):
                                pass
                            output.write('"[REDACTED]"')
                        else:
                            _write_value(events, first, output, transform, secret_key)
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                result[key] = path
    except (ijson.JSONError, StopIteration) as error:
        raise ValueError("Invalid tracking JSON") from error
    descriptor = os.open(output_dir, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return result


def rewrite_json(
    source: Path, target: Path, *, transform: Callable[[Any], Any] | None = None,
    secret_key: Callable[[str], bool] | None = None,
) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4()}.tmp")
    try:
        temporary.touch(mode=0o600, exist_ok=False)
        with source.open("rb") as stream, temporary.open("w", encoding="utf-8") as output:
            events = iter(ijson.parse(stream, use_float=True))
            _write_value(events, next(events), output, transform, secret_key)
            for _ in events:
                pass
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(target)
        descriptor = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except (ijson.JSONError, StopIteration) as error:
        raise ValueError("Invalid tracking JSON") from error
    finally:
        temporary.unlink(missing_ok=True)


def split_array(
    source: Path, output_dir: Path, *, transform: Callable[[Any], Any] | None = None,
    secret_key: Callable[[str], bool] | None = None,
) -> list[Path]:
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    result = []
    try:
        with source.open("rb") as stream:
            events = iter(ijson.parse(stream, use_float=True))
            if next(events)[1] != "start_array":
                raise ValueError("Tracking items must belong to an array")
            for first in events:
                if first[1] == "end_array":
                    break
                path = output_dir / f"{len(result):08d}.json"
                temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
                try:
                    temporary.touch(mode=0o600, exist_ok=False)
                    with temporary.open("w", encoding="utf-8") as output:
                        _write_value(events, first, output, transform, secret_key)
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                result.append(path)
            for _ in events:
                pass
    except (ijson.JSONError, StopIteration) as error:
        raise ValueError("Invalid tracking JSON") from error
    descriptor = os.open(output_dir, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return result
