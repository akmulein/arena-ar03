"""One train page through the Arena contracts, without Docker or services."""

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path

from evaluator.contracts import Example, Metric
from evaluator.metrics import aggregate_results, evaluate_answer

CASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar3", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    runtime = str(args.python.absolute())
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    config = json.loads((CASE / "metric/metric.json").read_text())
    # Only interpreter/path resolution changes for this unfrozen local check.
    config["command"]["argv"] = [runtime, "-B", str(CASE / "metric/score.py")]
    metric = Metric.model_validate(config)
    synthetic = Example(id="empty", input="", reference='{"tables": []}')
    assert evaluate_answer(metric, synthetic, '{"tables": []}')["score"] == 1
    assert evaluate_answer(metric, synthetic, 'invalid JSON')["score"] == 0
    with (args.ar3 / "data/train/manifest.jsonl").open() as stream:
        record = json.loads(next(stream))
    page_id = record["id"]
    image = (args.ar3 / "data/train/images" / f"{page_id}.jpg").read_bytes()
    value = json.dumps({"image_base64": base64.b64encode(image).decode("ascii")})
    reference = (args.ar3 / "data/train/truth" / f"{page_id}.json").read_text()
    example = Example(id=page_id, input=value, reference=reference)
    payload = json.dumps({"input": example.input, "assets_dir": "/assets"})
    assert len(payload.encode()) <= 8 * 1024 * 1024
    started = time.perf_counter()
    completed = subprocess.run(
        [runtime, "-B", "-E", "-s", str(CASE / "ways/baseline/solve.py")],
        input=payload, text=True, capture_output=True, check=True,
        cwd="/tmp", timeout=300,
    )
    solver_seconds = time.perf_counter() - started
    assert len(completed.stdout.encode()) <= 1024 * 1024
    answer = json.loads(completed.stdout)["answer"]
    assert isinstance(answer, str)
    started = time.perf_counter()
    result = evaluate_answer(metric, example, answer)
    final = aggregate_results(metric, [{"example_id": page_id, **result}])
    judge_seconds = time.perf_counter() - started
    print(json.dumps({
        "page_id": page_id, "input_characters": len(value),
        "prediction_valid": result["statistics"]["valid"], "teds": final,
        "solver_seconds": solver_seconds, "judge_seconds": judge_seconds,
        "solver_tokens": 0, "judge_tokens": 0,
        "synthetic_adapter_checks": "passed",
    }, indent=2))
    assert result["statistics"]["valid"]


if __name__ == "__main__":
    main()
