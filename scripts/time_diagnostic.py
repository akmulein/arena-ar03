"""Time normal inference on the first 20 fixed diagnostic train pages."""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    args = parser.parse_args()
    root = Path("reports/diagnostics")
    if args.phase == "before":
        sys.path.insert(0, str((root / "before/source").resolve()))
    solve = importlib.import_module("solution.pipeline").solve
    records = [
        json.loads(line)
        for line in (root / "train-200.jsonl").read_text().splitlines()
    ][:20]
    rows = []
    for record in records:
        image = cv2.imread(f"data/train/images/{record['id']}.jpg")
        start = time.perf_counter()
        solve(image)
        rows.append(dict(id=record["id"], seconds=time.perf_counter() - start))
    (root / args.phase / "timing-20.json").write_text(
        json.dumps(rows, indent=2)
    )
    print(args.phase, sum(r["seconds"] for r in rows) / len(rows))


if __name__ == "__main__":
    main()
