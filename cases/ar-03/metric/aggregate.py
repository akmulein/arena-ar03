"""Preserve macro page TEDS and micro corpus diagnostics for Arena."""

import json
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent / "runtime"
sys.path.insert(0, str(RUNTIME))

from ar03_metric.evaluation.reporting import summarize  # noqa: E402

QUALITY_FIELDS = ("teds", "teds_struct", "cell_f1", "numeric_accuracy", "valid")
COUNTER_FIELDS = (
    "cell_correct", "cell_predicted", "cell_true", "numeric_correct", "numeric_total",
    "detection_tp", "detection_fp", "detection_fn",
)
SUMMARY_FIELDS = (
    "pages", "teds", "teds_struct", "cell_f1", "numeric_accuracy", "errors",
    "detection_precision", "detection_recall", "detection_f1",
    "detection_tp", "detection_fp", "detection_fn",
)


def aggregate_payload(payload):
    reports = payload["results"]
    if not isinstance(reports, list) or not reports:
        raise ValueError("AR-03 aggregation requires nonempty reports")
    results = []
    seen = set()
    for report in reports:
        example_id = report["example_id"]

        if not isinstance(example_id, str) or not example_id or example_id in seen:
            raise ValueError("AR-03 aggregation requires unique example IDs")
        
        if report.get("failure"):
            raise ValueError("AR-03 aggregation cannot omit a failed page")
        
        seen.add(example_id)
        statistics = report["statistics"]
        quality = statistics["quality"]
        counters = statistics["counters"]
        recognition_error = statistics["recognition_error"]

        if type(recognition_error) is not bool:
            raise ValueError("AR-03 recognition_error must be a boolean")
        
        if report.get("score") != quality["teds"]:
            raise ValueError("AR-03 report score differs from page TEDS")
        
        results.append({
            **{key: quality[key] for key in QUALITY_FIELDS},
            **{key: counters[key] for key in COUNTER_FIELDS},
            # Source summarize counts non-null errors. Text itself is unnecessary.
            "error": "recognition_error" if recognition_error else None,
            # Arena measures solver resources independently; never trust candidate timing.
            # This placeholder satisfies source summarize; its timing fields are omitted.
            "seconds": 0.0,
        })

    summary = summarize(
        results, fingerprint="arena-command-metric", split="arena",
        population_pages=len(results), sampling="complete supplied set",
    )

    return {
        "score": summary["teds"],
        "statistics": {key: summary[key] for key in SUMMARY_FIELDS},
    }


def main():
    try:
        result = aggregate_payload(json.load(sys.stdin))
        json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    
    except Exception as error:
        sys.stderr.write(f"AR-03 aggregation failed ({type(error).__name__})\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
