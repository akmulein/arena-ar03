"""Time normal inference on the first 20 fixed diagnostic train pages."""

import argparse
import json
import time

import cv2

from baseline.pipeline import solve
from dataset.config import ROOT as DATA_ROOT
from evaluation.diagnose import ROOT, sample
from evaluation.cache import baseline_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    from pathlib import Path
    if not args.phase or Path(args.phase).name != args.phase or args.phase in {".", ".."}:
        parser.error("phase must be a single directory name")
    root = ROOT
    phase_dir = root / args.phase
    run_lock = phase_dir / "run-config.json"
    if not run_lock.is_file():
        parser.error("run evaluation.diagnose for this phase first")
    if json.loads(run_lock.read_text())["fingerprint"] != baseline_fingerprint():
        parser.error("phase belongs to a different baseline")
    records = sample()[:20]
    rows = []
    for record in records:
        image = cv2.imread(str(DATA_ROOT / f"train/images/{record['id']}.jpg"))
        start = time.perf_counter()
        solve(image)
        rows.append(dict(id=record["id"], seconds=time.perf_counter() - start))
    (root / args.phase / "timing-20.json").write_text(
        json.dumps(rows, indent=2)
    )
    print(args.phase, sum(r["seconds"] for r in rows) / len(rows))


if __name__ == "__main__":
    main()
