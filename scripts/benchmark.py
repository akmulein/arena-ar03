import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from evaluator.metrics import score
from scripts.prepare import SPLITS
from solution.pipeline import solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "public"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    root = Path("data") / args.split
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    if len(records) != SPLITS[args.split][1]:
        raise ValueError("Full benchmark requires the complete split")
    population_pages = len(records)
    if args.manifest:
        selected = [
            json.loads(line) for line in args.manifest.read_text().splitlines()
        ]
        available = {r["id"]: r for r in records}
        if len({r["id"] for r in selected}) != len(selected):
            raise ValueError("Subset contains duplicate pages")
        if any(available.get(r["id"]) != r for r in selected):
            raise ValueError("Subset records must match the source manifest")
        records = selected
    fingerprint = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path("solution").glob("*.py")))
        + Path("requirements.txt").read_bytes()
    ).hexdigest()
    if args.limit:
        records = records[: args.limit]
    out = Path("reports/predictions") / args.split
    out.mkdir(parents=True, exist_ok=True)
    if args.cached_only:
        records = [r for r in records if (out / f"{r['id']}.json").exists()]
    if not records:
        raise ValueError("No pages to evaluate")
    results = []
    for record in records:
        page_id = record["id"]
        saved = out / f"{page_id}.json"
        run = json.loads(saved.read_text()) if saved.exists() else {}
        if (
            run.get("fingerprint") != fingerprint
            or run.get("image_sha256") != record["sha256"]
        ):
            if args.cached_only:
                raise ValueError(f"Stale cache: {page_id}")
            image = cv2.imread(str(root / "images" / f"{page_id}.jpg"))
            start = time.perf_counter()
            try:
                prediction = solve(image)
                error = None
            except (RuntimeError, ValueError) as exc:
                prediction, error = {"tables": []}, str(exc)
            run = dict(
                fingerprint=fingerprint,
                image_sha256=record["sha256"],
                prediction=prediction,
                seconds=time.perf_counter() - start,
                error=error,
            )
            saved.write_text(json.dumps(run, ensure_ascii=False))
        truth = json.loads((root / "truth" / f"{page_id}.json").read_text())
        metrics = score(run["prediction"], truth)
        results.append(
            dict(
                id=page_id,
                seconds=run["seconds"],
                error=run["error"],
                **metrics,
            )
        )
        if len(results) % 100 == 0:
            print(args.split, len(results), flush=True)
    summary = dict(
        fingerprint=fingerprint,
        split=args.split,
        pages=len(results),
        population_pages=population_pages,
        sampling=(
            "cached available pages; not a full-train estimate"
            if args.cached_only
            else str(args.manifest)
            if args.manifest
            else "first pages"
            if args.limit
            else "complete split"
        ),
        teds=float(np.mean([r["teds"] for r in results])),
        teds_struct=float(np.mean([r["teds_struct"] for r in results])),
        cell_f1=2
        * sum(r["cell_correct"] for r in results)
        / max(1, sum(r["cell_predicted"] + r["cell_true"] for r in results)),
        numeric_accuracy=sum(r["numeric_correct"] for r in results)
        / max(1, sum(r["numeric_total"] for r in results)),
        seconds_mean=float(np.mean([r["seconds"] for r in results])),
        seconds_total=sum(r["seconds"] for r in results),
        errors=sum(r["error"] is not None for r in results),
    )
    name = args.manifest.stem if args.manifest else args.split
    if args.cached_only:
        name += f"-cached-{len(records)}"
    elif args.limit:
        name += f"-first-{args.limit}"
    Path(f"reports/{name}.json").write_text(json.dumps(summary, indent=2))
    Path(f"reports/{name}-pages.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
