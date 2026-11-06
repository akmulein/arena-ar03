"""Two examples through Arena DockerRunner and a separate metric container."""

import argparse
import base64
import json
import shutil
import struct
import time
import zlib
from pathlib import Path

from evaluator.contracts import Example, Metric
from evaluator.metrics import aggregate_results, evaluate_answer
from evaluator.runner import DockerRunner

CASE = Path(__file__).resolve().parent


def synthetic():
    width, height = 400, 300
    rows = []
    for y in range(height):
        row = bytearray([255] * width)
        for x in range(width):
            horizontal = 30 <= x <= 330 and any(abs(y - v) <= 1 for v in (40, 140, 240))
            vertical = 40 <= y <= 240 and any(abs(x - v) <= 1 for v in (30, 180, 330))
            if horizontal or vertical:
                row[x] = 0
        rows.append(b"\x00" + row)

    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack(
            "!I", zlib.crc32(kind + data),
        )

    image = b"\x89PNG\r\n\x1a\n"
    image += chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 0, 0, 0, 0))
    image += chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")
    cells = [
        {"r0": r, "r1": r, "c0": c, "c1": c, "text": "",
         "bbox": [30 + c * 150, 40 + r * 100, 180 + c * 150, 140 + r * 100]}
        for r in range(2) for c in range(2)
    ]
    truth = {"tables": [{"bbox": [30, 40, 330, 240], "rows": 2, "cols": 2, "cells": cells}]}
    return image, json.dumps(truth)


def example(page_id, image, reference):
    value = json.dumps({"image_base64": base64.b64encode(image).decode("ascii")})
    return Example(id=page_id, input=value, reference=reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--runner-image", default="ar03-runner:local")
    parser.add_argument("--metric-image", default="ar03-metric:local")
    args = parser.parse_args()
    config = json.loads((CASE / "metric/metric.json").read_text())
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker CLI is not installed")
    config["command"]["argv"] = [
        docker, "run", "--rm", "--interactive", "--init", "--network", "none",
        "--read-only", "--user", "10001:10001", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "128",
        "--memory", "2g", "--cpus", "2", "--log-driver", "none",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=128m,uid=10001,gid=10001",
        args.metric_image,
    ]
    metric = Metric.model_validate(config)
    runner = DockerRunner(image=args.runner_image, uses_model=False, timeout=300)
    image, reference = synthetic()
    artificial = example("synthetic-grid", image, reference)
    identity = evaluate_answer(metric, artificial, reference)
    assert identity["score"] == 1.0 and identity["statistics"]["valid"]
    invalid = evaluate_answer(metric, artificial, "not JSON")
    assert invalid["score"] == 0.0 and not invalid["statistics"]["valid"]
    print(json.dumps({"identity_teds": identity["score"], "invalid_score": invalid["score"],
                      "invalid_valid": invalid["statistics"]["valid"]}), flush=True)

    def check(item):
        # Only image input goes to DockerRunner. Reference stays with the judge.
        solved = runner(CASE / "ways/baseline", item.input)
        started = time.perf_counter()
        judged = evaluate_answer(metric, item, solved.answer)
        final = aggregate_results(metric, [{"example_id": item.id, **judged}])
        judge_seconds = time.perf_counter() - started
        assert judged["statistics"]["valid"]
        print(json.dumps({
            "id": item.id, "teds": final, "prediction_valid": True,
            "solver_seconds": solved.wall_seconds, "judge_seconds": judge_seconds,
            "solver_usage": solved.usage, "judge_tokens": 0,
        }), flush=True)

    check(artificial)
    with (args.train / "manifest.jsonl").open() as stream:
        record = json.loads(next(stream))
    page_id = record["id"]
    check(example(
        page_id, (args.train / "images" / f"{page_id}.jpg").read_bytes(),
        (args.train / "truth" / f"{page_id}.json").read_text(),
    ))


if __name__ == "__main__":
    main()
