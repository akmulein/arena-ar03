"""Evaluate one Arena answer using the approved AR-03 page metric."""

import json
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / "runtime"
sys.path.insert(0, str(RUNTIME))

from ar03_metric.evaluation.detection import score_detection  # noqa: E402
from ar03_metric.evaluation.metrics import score  # noqa: E402

QUALITY_FIELDS = ("teds", "teds_struct", "cell_f1", "numeric_accuracy", "valid")
COUNTER_FIELDS = (
    "cell_correct", "cell_predicted", "cell_true", "numeric_correct", "numeric_total",
    "detection_tp", "detection_fp", "detection_fn",
)


def evaluate_payload(payload):
    """Keep invalid-prediction policy inside the original scoring function."""
    example = payload["example"]
    truth = json.loads(example["reference"])
    answer = json.loads(payload["answer"])
    if not isinstance(answer, dict) or set(answer) != {"prediction", "detections", "error"}:
        raise ValueError("AR-03 answer must contain prediction, detections and error")
    if not isinstance(answer["detections"], list):
        raise ValueError("AR-03 detections must be a list")
    error = answer["error"]
    if error is not None and not isinstance(error, str):
        raise ValueError("AR-03 recognition error must be a string or null")
    result = score(answer["prediction"], truth)
    result.update(score_detection(answer["detections"], truth["tables"]))
    return {
        "score": result["teds"],
        "statistics": {
            "quality": {key: result[key] for key in QUALITY_FIELDS},
            "counters": {key: result[key] for key in COUNTER_FIELDS},
            "recognition_error": error is not None,
        },
    }


def main():
    try:
        result = evaluate_payload(json.load(sys.stdin))
        json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    except Exception as error:
        # Closed references or text must not leak through command diagnostics.
        sys.stderr.write(f"AR-03 metric failed ({type(error).__name__})\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
