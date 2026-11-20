"""Operator-only rendering of saved Arena checkpoints; never runs the solver.

Run ``python operator_artifacts.py --config /operator/artifacts.json once``
or ``... watch``. ``... supervise -- python -m evaluator.train ...`` starts
the normal service, watches its checkpoints, and performs a final scan on exit.
The configuration contains output_root, metric_runtime, and sources. Each
source contains split, state_dir, dataset_path, and optional record_globs.
All paths, records, logs, and generated artifacts belong in the closed operator
account. Do not import this module from the runner or metric.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any

DEFAULT_GLOBS = {
    "train": ["evaluations/*/record.json"],
    "public": [
        "*/public/candidate-results/*/details.json",
        "*/public/*/details/*.json",
    ],
    "private": ["*/details/*.json"],
}
METRIC_FIELDS = (
    "score",
    "statistics",
    "status",
    "failure",
    "error",
    "error_type",
    "diagnostics",
    "resources",
    "resources_complete",
)


def component(value: str) -> str:
    """Retain ordinary IDs; encode other IDs without traversal or collisions."""
    if not isinstance(value, str) or not value or len(value) > 100:
        raise ValueError("invalid identifier")
    if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        return value
    return "id-" + sha256(value.encode("utf-8")).hexdigest()


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("artifact directories cannot be symlinks")
    if path.parent != path:
        private_directory(path.parent)
    if not path.exists():
        path.mkdir(mode=0o700)
    if not path.is_dir():
        raise ValueError("artifact path must be a directory")


def write_bytes(path: Path, value: bytes) -> None:
    private_directory(path.parent)
    if path.is_symlink():
        raise ValueError("artifact files cannot be symlinks")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as out:
        pending = Path(out.name)
        try:
            out.write(value)
            out.flush()
            os.fsync(out.fileno())
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode())


def json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def private_log(path: Path) -> Any:
    """Append logs without following an existing symlink."""
    private_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "ab")


@dataclass(frozen=True)
class Source:
    split: str
    state_dir: Path
    dataset_path: Path
    record_globs: tuple[str, ...]


@dataclass(frozen=True)
class Configuration:
    output_root: Path
    metric_runtime: Path
    sources: tuple[Source, ...]

    @classmethod
    def load(cls, path: Path) -> Configuration:
        data = json_object(path.read_text())

        def resolve(value: str) -> Path:
            result = Path(value).expanduser()
            return (path.parent / result).absolute() if not result.is_absolute() else result

        sources = []
        for item in data["sources"]:
            split = item["split"]
            if split not in DEFAULT_GLOBS:
                raise ValueError("unknown split")
            configured = item.get("record_globs", DEFAULT_GLOBS[split])
            if not isinstance(configured, list):
                raise ValueError("record patterns must be a list")
            patterns = tuple(configured)
            if not patterns or any(
                not isinstance(p, str) or Path(p).is_absolute() or ".." in Path(p).parts
                for p in patterns
            ):
                raise ValueError("record patterns must be relative")
            sources.append(
                Source(
                    split,
                    resolve(item["state_dir"]),
                    resolve(item["dataset_path"]),
                    patterns,
                )
            )
        if not sources:
            raise ValueError("at least one source is required")
        result = cls(
            resolve(data["output_root"]),
            resolve(data["metric_runtime"]),
            tuple(sources),
        )
        if any(
            result.output_root.resolve().is_relative_to(s.state_dir.resolve() / "candidates")
            for s in result.sources
        ):
            raise ValueError("artifacts cannot be written into candidate storage")
        return result


class FrozenMetric:
    """Import only the operator-selected frozen runtime, never candidate code."""

    def __init__(self, runtime: Path):
        runtime = runtime.resolve(strict=True)
        for name, module in tuple(sys.modules.items()):
            if name == "ar03_metric" or name.startswith("ar03_metric."):
                location = getattr(module, "__file__", None)
                if location is None or not Path(location).resolve().is_relative_to(runtime):
                    raise ValueError("a different metric runtime is already imported")
        sys.path.insert(0, str(runtime))
        try:
            modules = [
                importlib.import_module(name)
                for name in (
                    "ar03_metric.common.postprocess",
                    "ar03_metric.common.schema",
                    "ar03_metric.evaluation.teds",
                )
            ]
            if any(
                not Path(module.__file__).resolve().is_relative_to(runtime) for module in modules
            ):
                raise ValueError("a different metric runtime is already imported")
        finally:
            sys.path.pop(0)
        self.postprocess, self.schema, self.teds = modules

    def document(self, page: dict[str, Any]) -> str:
        return self.postprocess.to_html(page)

    def trees(self, page: dict[str, Any]) -> list[dict[str, Any]]:
        """Use the exact TEDS parser/tokenizer/tree class from the frozen metric."""
        from lxml import html

        def serialize(node: Any) -> dict[str, Any]:
            return {
                "tag": node.tag,
                "colspan": node.colspan,
                "rowspan": node.rowspan,
                "content": node.content,
                "children": [serialize(n) for n in node.children],
            }

        trees = []
        for table in page["tables"]:
            document = "<html><body>" + self.postprocess.table_html(table) + "</body></html>"
            parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
            node = html.fromstring(document, parser=parser).xpath("body/table")[0]
            full = self.teds.TEDS().load_html_tree(node)
            structure = self.teds.TEDS(structure_only=True).load_html_tree(node)
            trees.append(
                {
                    "html": document,
                    "descendant_count": len(node.xpath(".//*")),
                    "full": serialize(full),
                    "structure_only": serialize(structure),
                }
            )
        return trees


def grid_svg(page: dict[str, Any]) -> str:
    """Logical grid with explicit spans, independent of predicted bbox validity."""
    width, unit, top = 960, 42, 25
    height = sum(table["rows"] * unit + 55 for table in page["tables"]) + 30
    shapes = []
    for index, table in enumerate(page["tables"]):
        cell_width = (width - 20) / table["cols"]
        shapes.append(f'<text x="10" y="{top}" font-size="16">Table {index}</text>')
        top += 10
        for cell in table["cells"]:
            rs, cs = cell["r1"] - cell["r0"] + 1, cell["c1"] - cell["c0"] + 1
            x, y = 10 + cell["c0"] * cell_width, top + cell["r0"] * unit
            fill = "#dceafa" if rs > 1 or cs > 1 else "#ffffff"
            label = f"r{cell['r0']}:{cell['r1']} c{cell['c0']}:{cell['c1']}"
            text = escape(cell["text"])
            shapes.append(
                f'<rect x="{x}" y="{y}" width="{cs * cell_width}" height="{rs * unit}" '
                f'fill="{fill}" stroke="#345"/><text x="{x + 4}" y="{y + 14}" '
                f'font-size="10">{label}<title>{text}</title></text>'
            )
        top += table["rows"] * unit + 45
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" '
        f'fill="white"/>{"".join(shapes)}</svg>'
    )


def image_overlays(
    image: Any,
    prediction: dict[str, Any] | None,
    truth: dict[str, Any],
    directory: Path,
) -> int:
    import cv2

    missing = 0
    colors = {"prediction": (40, 180, 20), "truth": (190, 30, 190)}

    def box(canvas: Any, values: Any, color: tuple[int, int, int], label: str) -> bool:
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            return False
        height, width = canvas.shape[:2]
        x0, y0, x1, y1 = [
            round(max(0, min(v, width if i % 2 == 0 else height))) for i, v in enumerate(values)
        ]
        if x0 >= x1 or y0 >= y1:
            return False
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            canvas,
            label,
            (x0, max(12, y0 + 13)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        return True

    def save(name: str, canvas: Any) -> None:
        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:
            raise ValueError("overlay encoding failed")
        write_bytes(directory / name, encoded.tobytes())

    tables = image.copy()
    for name, page in (("truth", truth), ("prediction", prediction)):
        if page is None:
            continue
        for index, table in enumerate(page["tables"]):
            if not box(tables, table.get("bbox"), colors[name], f"{name} T{index}"):
                missing += 1
        grid = image.copy()
        for index, table in enumerate(page["tables"]):
            for cell in table["cells"]:
                label = f"T{index} r{cell['r0']}:{cell['r1']} c{cell['c0']}:{cell['c1']}"
                if not box(grid, cell.get("bbox"), colors[name], label):
                    missing += 1
        save(name + "-cells-overlay.png", grid)
    cv2.putText(
        tables,
        "truth: purple | prediction: green",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    save("tables-overlay.png", tables)
    return missing


class ArtifactConsumer:
    def __init__(self, config: Configuration):
        self.config = config
        private_directory(config.output_root)
        config.output_root.chmod(0o700)
        self.metric = FrozenMetric(config.metric_runtime)
        self.datasets: dict[Source, dict[str, tuple[int, int]]] = {}
        self.dataset_versions: dict[Source, tuple[int, int]] = {}
        self.seen: dict[tuple[str, str, str, int | None], bytes] = {}
        self.scan_errors: list[dict[str, Any]] = []
        self.processed = 0
        self.errors = 0

    def example(self, source: Source, page_id: str) -> dict[str, Any]:
        """Index byte offsets, keeping image payloads out of persistent memory/storage."""
        stat = source.dataset_path.stat()
        version = (stat.st_size, stat.st_mtime_ns)
        if source not in self.datasets:
            result = {}
            with source.dataset_path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    item = json_object(json.loads(line))
                    identifier = item["id"]
                    if identifier in result:
                        raise ValueError("dataset IDs are not unique")
                    result[identifier] = (offset, len(line))
            self.datasets[source] = result
            self.dataset_versions[source] = version
        current = source.dataset_path.stat()
        if self.dataset_versions[source] != (current.st_size, current.st_mtime_ns):
            raise ValueError("dataset changed while rendering checkpoints")
        offset, length = self.datasets[source][page_id]
        with source.dataset_path.open("rb") as stream:
            stream.seek(offset)
            return json_object(json.loads(stream.read(length)))

    def render(
        self,
        source: Source,
        candidate: str,
        row: dict[str, Any],
        example: dict[str, Any],
        record_path: Path,
        attempt: int | None = None,
    ) -> None:
        import cv2
        import numpy as np

        page_id = row["example_id"]
        directory = (
            self.config.output_root / source.split / component(candidate) / component(page_id)
        )
        if attempt is not None:
            directory = directory / "interrupted-attempts" / str(attempt)
        private_directory(directory)
        evaluation = {key: row[key] for key in METRIC_FIELDS if key in row}
        evaluation.update(
            {
                "metrics_available": "score" in row or "statistics" in row,
                "split": source.split,
                "candidate_id": candidate,
                "page_id": page_id,
                "record_path": str(record_path),
            }
        )
        errors = []
        prediction = None
        truth = None
        write_bytes(directory / "answer.txt", str(row.get("answer", "")).encode())
        if "answer" in row:
            try:
                answer = json_object(row["answer"])
                write_json(directory / "answer.json", answer)
                evaluation["recognition_error"] = answer.get("error")
                prediction = json_object(answer["prediction"])
                write_json(directory / "prediction.json", prediction)
                self.metric.schema.validate(prediction)
            except Exception as error:
                prediction = None
                errors.append(
                    {
                        "stage": "prediction",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
        try:
            truth = json_object(example["reference"])
            write_json(directory / "truth.json", truth)
            self.metric.schema.validate(truth)
        except Exception as error:
            truth = None
            errors.append({"stage": "truth", "type": type(error).__name__, "message": str(error)})
        for name, page in (("prediction", prediction), ("truth", truth)):
            if page is None:
                continue
            try:
                write_bytes(directory / (name + ".html"), self.metric.document(page).encode())
                write_json(directory / (name + "-teds-trees.json"), self.metric.trees(page))
                write_bytes(directory / (name + "-logical-grid.svg"), grid_svg(page).encode())
            except Exception as error:
                errors.append(
                    {
                        "stage": name + "-render",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
        try:
            payload = json_object(example["input"])
            if payload.get("version") != "ar03-page-v1":
                raise ValueError("unsupported image input")
            image = cv2.imdecode(
                np.frombuffer(base64.b64decode(payload["image_base64"], validate=True), np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError("cannot decode image")
            evaluation["missing_geometry"] = image_overlays(
                image, prediction, truth or {"tables": []}, directory
            )
        except Exception as error:
            errors.append(
                {
                    "stage": "overlays",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        evaluation["artifact_errors"] = errors
        evaluation["answer_available"] = "answer" in row
        evaluation["prediction_available"] = prediction is not None
        evaluation["truth_available"] = truth is not None
        self.errors += bool(errors)
        write_json(directory / "evaluation.json", evaluation)
        self.processed += 1

    def record_error(self, path: Path, error: Exception, page_id: str | None = None) -> None:
        detail = {
            "record_path": str(path),
            "page_id": page_id,
            "type": type(error).__name__,
            "message": str(error),
        }
        if detail not in self.scan_errors:
            self.scan_errors.append(detail)
            self.errors += 1
            write_json(self.config.output_root / "operator-errors.json", self.scan_errors)

    def consume_row(
        self,
        source: Source,
        path: Path,
        candidate: str,
        row: dict[str, Any],
        attempt: int | None = None,
    ) -> None:
        if "answer" not in row and not row.get("failure"):
            return  # A solving marker is not a processed page.
        key = (str(path), candidate, row["example_id"], attempt)
        stamp = sha256(
            json.dumps(
                {k: v for k, v in row.items() if k not in {"calls", "input", "reference"}},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).digest()
        if self.seen.get(key) == stamp:
            return
        self.render(
            source,
            candidate,
            row,
            self.example(source, row["example_id"]),
            path,
            attempt=attempt,
        )
        self.seen[key] = stamp

    def scan(self) -> None:
        from evaluator.checkpoints import read_document

        for source in self.config.sources:
            paths = sorted(
                {p for pattern in source.record_globs for p in source.state_dir.glob(pattern)}
            )
            for path in paths:
                try:
                    if path.is_symlink() or not path.resolve().is_relative_to(
                        source.state_dir.resolve()
                    ):
                        raise ValueError("record must remain inside its state directory")
                    document = read_document(path)
                    candidate = document.get(
                        "candidate_id", document.get("params", {}).get("candidate_id")
                    )
                    if not isinstance(candidate, str):
                        raise ValueError("record has no candidate ID")
                    rows = [(row, None) for row in document.get("examples", [])]
                    rows += [
                        (row, index)
                        for index, row in enumerate(document.get("interrupted_attempts", []))
                    ]
                    for row, attempt in rows:
                        try:
                            if not isinstance(row, dict):
                                raise ValueError("invalid example checkpoint")
                            self.consume_row(source, path, candidate, row, attempt)
                        except Exception as error:
                            self.record_error(
                                path,
                                error,
                                row.get("example_id") if isinstance(row, dict) else None,
                            )
                except Exception as error:
                    self.record_error(path, error)


def run(
    config: Configuration,
    mode: str,
    *,
    interval: float = 2,
    command: list[str] | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Watch committed page stages even when the final candidate record fails."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    stop = stop or threading.Event()
    consumer = ArtifactConsumer(config)
    start = time.monotonic()
    child = None
    stdout = stderr = None
    previous = (-1, -1)
    try:
        if mode == "supervise":
            if not command:
                raise ValueError("supervise requires a service command")
            stdout = private_log(config.output_root / "service.stdout.log")
            stderr = private_log(config.output_root / "service.stderr.log")
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        while True:
            consumer.scan()
            current = (consumer.processed, consumer.errors)
            if current != previous:
                print(
                    json.dumps(
                        {
                            "elapsed_seconds": round(time.monotonic() - start, 3),
                            "rendered_page_updates": current[0],
                            "artifact_errors": current[1],
                        }
                    ),
                    flush=True,
                )
                previous = current
            if mode == "once":
                return int(consumer.errors > 0)
            if child is not None and child.poll() is not None:
                consumer.scan()  # Include final atomic writes made just before service exit.
                return child.returncode or int(consumer.errors > 0)
            if stop.wait(interval):
                break
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        consumer.scan()
        return int(consumer.errors > 0)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("mode", choices=["once", "watch", "supervise"])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    os.umask(0o077)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    try:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return run(
            Configuration.load(args.config),
            args.mode,
            interval=args.interval,
            command=command,
            stop=stop,
        )
    except Exception:
        # No page IDs, inputs, scores, paths, or exception payloads on the console.
        print(json.dumps({"operator_artifact_errors": 1}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
