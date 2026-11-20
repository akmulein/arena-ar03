"""Render permitted Arena journal data; never read public/private evaluator state.

Run once after a measurement, or use --watch in a separate process. Page progress
accepts only an explicitly supplied train API result/checkpoint. The journal
remains the authority for registration order and public AUC.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from html import escape
from pathlib import Path

ARENA_ROOT = Path(__file__).resolve().parents[2]
if (ARENA_ROOT / "evaluator" / "search.py").is_file():
    sys.path.insert(0, str(ARENA_ROOT))

from evaluator.checkpoints import read_document  # noqa: E402
from evaluator.search import Journal  # noqa: E402

STAGES = ("detection", "structure", "ocr", "end-to-end")
COLORS = {
    "succeeded": "#087f5b", "failed": "#c92a2a",
    "running": "#1971c2", "unmeasured": "#868e96",
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def train_progress(record, *, now_ns=None):
    """Produce counters only; input must come from the open train service."""
    if record.get("split") != "train":
        raise ValueError("page progress accepts only train records")
    rows = record.get("examples", [])
    total = len(record.get("example_ids", []))
    if not total:
        total = record.get("progress", {}).get("total_examples", len(rows))
    processed = len(rows)
    evaluated = sum("score" in row or "statistics" in row for row in rows)
    errors = 0
    for row in rows:
        failed = bool(row.get("failure") or row.get("error"))
        stats = row.get("statistics", {})
        failed |= bool(stats.get("recognition_error") or stats.get("error"))
        if isinstance(row.get("answer"), str):
            try:
                answer = json.loads(row["answer"])
                failed |= isinstance(answer, dict) and bool(answer.get("error"))
            except (ValueError, TypeError):
                failed = True
        errors += bool(failed)
    started = record.get("start_time_ns")
    ended = record.get("end_time_ns")
    current = time.time_ns() if now_ns is None else now_ns
    if _finite(started):
        elapsed = max(0, ((ended if _finite(ended) else current) - started) / 1e9)
    else:
        elapsed = record.get("progress", {}).get("elapsed_seconds")
    remaining = (
        elapsed / processed * max(0, total - processed)
        if processed and _finite(elapsed) else None
    )
    return {
        "status": record.get("status"), "processed_pages": processed,
        "evaluated_pages": evaluated, "total_pages": total,
        "percent": 100 * processed / total if total else None,
        "elapsed_seconds": elapsed, "remaining_seconds": remaining, "errors": errors,
    }


def snapshot(journal, comparison, *, stage, train_record=None):
    """Use Arena's curve unchanged; expose no hypotheses, evidence or raw errors."""
    if comparison.get("split") not in ("train", "public"):
        raise ValueError("monitor comparison must be train or public")
    if stage not in STAGES:
        raise ValueError("unknown pipeline stage")
    exported = journal.export()
    campaign = exported["campaign"]
    curve = journal.curve(**comparison, direction="maximize")
    points = {point["candidate_id"]: point for point in curve["points"]}
    lineage = []
    for candidate in exported["candidates"]:
        point = points[candidate["candidate_id"]]
        state = point["state"]
        if candidate.get("snapshot_error") or candidate.get("bundle_error"):
            state = "failed"
        lineage.append({
            "candidate_id": candidate["candidate_id"],
            "solution_id": candidate["solution_id"], "proposal": candidate["proposal"],
            "parents": candidate["parents"], "state": state, "score": point["score"],
        })
    measured = [point for point in curve["points"] if point["state"] in ("succeeded", "failed")]
    summary = {
        "stage": stage, "split": comparison["split"],
        "max_solutions": campaign["max_solutions"],
        "proposed_candidates": campaign["proposed"],
        "evaluated_candidates": len(measured),
        "successful_candidates": sum(point["state"] == "succeeded" for point in measured),
        "failed_candidates": sum(point["state"] == "failed" for point in measured),
        "running_candidates": sum(point["state"] == "running" for point in curve["points"]),
        "best_teds": curve["final_quality"], "auc": curve["auc"],
        "observed_auc": curve["observed_auc"], "auc_applicable": curve["auc_applicable"],
        "auc_complete": curve["complete"] and curve["auc_applicable"],
        "stopped": campaign["stopped"], "finalized": campaign["finalization"] is not None,
    }
    if train_record is not None:
        if comparison["split"] != "train":
            raise ValueError("train page progress belongs to a train comparison")
        summary["train_progress"] = train_progress(train_record)
    return {"summary": summary, "curve": curve, "lineage": lineage}


def _svg(title, body, *, width=900, height=420):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        '<style>text{font-family:system-ui,sans-serif;font-size:13px;fill:#212529}</style>'
        f'<text x="30" y="30" font-size="18">{escape(title)}</text>{body}</svg>\n'
    )


def _number(value):
    return "unknown" if value is None else f"{value:.6g}"


def curve_svg(curve, *, area=False):
    """Missing scores stay gaps; failed proposals remain visible without a zero."""
    horizon = max(1, curve["horizon"])
    x0, x1, y0, y1 = 65, 865, 335, 70
    body = [
        f'<path d="M{x0} {y1}V{y0}H{x1}" fill="none" stroke="#868e96"/>',
        f'<text x="{x0}" y="385">Proposal order / fixed horizon {horizon}</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = y0 - value * (y0 - y1)
        body.append(f'<path d="M{x0} {y}H{x1}" stroke="#e9ecef"/>')
        body.append(f'<text x="28" y="{y + 4}">{value:.1f}</text>')
    def x(proposal):
        return x0 + proposal / horizon * (x1 - x0)

    def y(quality):
        return y0 - quality * (y0 - y1)
    if area and not curve["auc_applicable"]:
        body.append('<text x="90" y="190">AUC is defined only for public.</text>')
        return _svg("Train TEDS: AUC not applicable", "".join(body))
    previous = None
    for point in curve["points"]:
        proposal, score = point["proposal"], point["score"]
        quality = point["quality"]
        if area and quality is not None:
            left, right = x(proposal - 1), x(proposal)
            body.append(
                f'<rect x="{left}" y="{y(quality)}" width="{right - left}" '
                f'height="{y0 - y(quality)}" fill="#d0ebff" stroke="#74c0fc"/>'
            )
        if not area and score is not None:
            body.append(
                f'<circle cx="{x(proposal)}" cy="{y(score)}" r="4" fill="#1971c2">'
                f'<title>{escape(point["candidate_id"])} TEDS={score}</title></circle>'
            )
        if not area and quality is not None:
            if previous is not None:
                previous_proposal, previous_quality = previous
                body.append(
                    f'<path d="M{x(previous_proposal)} {y(previous_quality)}'
                    f'H{x(proposal)}V{y(quality)}" fill="none" stroke="#087f5b" '
                    'stroke-width="2"/>'
                )
            previous = (proposal, quality)
        elif quality is None:
            previous = None
        if point["state"] == "failed":
            xx = x(proposal)
            body.append(
                f'<path d="M{xx - 4} 45l8 8m-8 0l8 -8" stroke="#c92a2a"/>'
                f'<text x="{xx + 8}" y="54">{escape(point["solution_id"])} failed</text>'
            )
    if area:
        title = (
            f'Public AUC={_number(curve["auc"])}; '
            f'observed prefix={_number(curve["observed_auc"])}'
        )
    else:
        title = (
            f'{curve["comparison"]["split"]} TEDS; best observed='
            f'{_number(curve["final_quality"])}; blue=score, green=covered best'
        )
    return _svg(title, "".join(body))


def lineage_svg(candidates):
    """All registered versions, including failed and unmeasured descendants."""
    levels = {}
    positions = {}
    buckets = {}
    for candidate in candidates:
        level = 1 + max((levels[parent] for parent in candidate["parents"]), default=-1)
        levels[candidate["candidate_id"]] = level
        buckets.setdefault(level, []).append(candidate)
    width = max(900, max((len(items) for items in buckets.values()), default=1) * 190 + 40)
    height = max(220, len(buckets) * 105 + 90)
    for level, items in buckets.items():
        for index, candidate in enumerate(items):
            positions[candidate["candidate_id"]] = (
                (index + 0.5) * width / len(items), 85 + level * 105,
            )
    body = [
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto"><path d="M0 0L8 4L0 8" fill="#adb5bd"/></marker></defs>',
    ]
    for candidate in candidates:
        xx, yy = positions[candidate["candidate_id"]]
        for parent in candidate["parents"]:
            px, py = positions[parent]
            body.append(
                f'<path d="M{px} {py + 32}L{xx} {yy - 16}" stroke="#adb5bd" '
                'marker-end="url(#arrow)"/>'
            )
    for candidate in candidates:
        xx, yy = positions[candidate["candidate_id"]]
        state = candidate["state"]
        color = COLORS[state]
        label = candidate["solution_id"]
        if candidate["candidate_id"] != label:
            label += " / " + candidate["candidate_id"]
        body.extend([
            f'<rect x="{xx - 85}" y="{yy - 16}" width="170" height="64" '
            f'rx="8" fill="white" stroke="{color}" stroke-width="2"/>',
            f'<text x="{xx}" y="{yy + 4}" text-anchor="middle">{escape(label)}</text>',
            f'<text x="{xx}" y="{yy + 23}" text-anchor="middle">{state}</text>',
            f'<text x="{xx}" y="{yy + 41}" text-anchor="middle">'
            f'TEDS={_number(candidate["score"])}</text>',
        ])
    if not candidates:
        body.append('<text x="30" y="95">No registered proposals.</text>')
    return _svg("Candidate lineage", "".join(body), width=width, height=height)


def _atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_artifacts(result, output):
    _atomic_text(output / "summary.json", json.dumps(result["summary"], indent=2) + "\n")
    _atomic_text(output / "curve.json", json.dumps(result["curve"], indent=2) + "\n")
    _atomic_text(output / "lineage.json", json.dumps(result["lineage"], indent=2) + "\n")
    _atomic_text(output / "teds.svg", curve_svg(result["curve"]))
    _atomic_text(output / "auc.svg", curve_svg(result["curve"], area=True))
    _atomic_text(output / "lineage.svg", lineage_svg(result["lineage"]))


def _bar(completed, total):
    fraction = min(1, max(0, completed / total)) if total else 0
    count = round(fraction * 20)
    return "[" + "=" * count + "." * (20 - count) + "]"


def progress_line(summary):
    limit = summary["max_solutions"]
    proposed, evaluated = summary["proposed_candidates"], summary["evaluated_candidates"]
    line = (
        f'{_bar(proposed, limit)} proposed={proposed}/{limit} '
        f'evaluated={evaluated}/{limit} failed={summary["failed_candidates"]} '
        f'stage={summary["stage"]} best_{summary["split"]}_TEDS={_number(summary["best_teds"])}'
    )
    pages = summary.get("train_progress")
    if pages is not None:
        line += (
            f' | pages {_bar(pages["processed_pages"], pages["total_pages"])} '
            f'{pages["processed_pages"]}/{pages["total_pages"]} '
            f'({_number(pages["percent"])}%) elapsed={_number(pages["elapsed_seconds"])}s '
            f'ETA={_number(pages["remaining_seconds"])}s errors={pages["errors"]}'
        )
    return line


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "public"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--train-record", type=Path, help="open train API result/checkpoint only")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args(argv)
    if not _finite(args.interval) or args.interval <= 0:
        parser.error("interval must be finite and positive")
    if args.train_record is not None and args.split != "train":
        parser.error("train-record is supported only with split=train")
    comparison = {
        "split": args.split, "dataset_id": args.dataset,
        "subset_id": args.subset, "conditions_id": args.conditions,
    }
    result = None
    try:
        journal = Journal(args.journal)
        while True:
            record = read_document(args.train_record) if args.train_record is not None else None
            result = snapshot(journal, comparison, stage=args.stage, train_record=record)
            save_artifacts(result, args.output)
            print(progress_line(result["summary"]), file=sys.stderr, flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"AR03 monitoring failed ({type(error).__name__})", file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result["summary"], allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
