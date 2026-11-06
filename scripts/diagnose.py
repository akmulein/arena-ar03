"""Train-only, fixed-sample diagnosis; never loads public or private."""

import argparse
import copy
import hashlib
import inspect
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from evaluator.metrics import match_tables, score
from solution.detection import detect
from solution.geometry import transform_box
from solution.ocr import recognize
from solution.preprocessing import preprocess
from solution.structure import recover

ROOT = Path("reports/diagnostics")


def sample():
    path = ROOT / "train-200.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    records = [
        json.loads(line)
        for line in Path("data/train/manifest.jsonl").read_text().splitlines()
    ]
    records = random.Random(42).sample(records, 200)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def frozen_hashes(records):
    paths = list(Path("evaluator").glob("*.py"))
    paths += [
        Path("solution/schema.py"),
        Path("solution/postprocess.py"),
        Path("data/train/manifest.jsonl"),
        ROOT / "train-200.jsonl",
    ]
    paths += [Path(f"data/train/truth/{r['id']}.json") for r in records]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def draw_boxes(image, truth, predicted):
    image = image.copy()
    for boxes, color in [(truth, (0, 170, 0)), (predicted, (0, 0, 255))]:
        for table in boxes:
            x0, y0, x1, y1 = map(round, table["bbox"])
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    return image


def restore(table, inverse, width, height):
    table["bbox"] = transform_box(table["bbox"], inverse, width, height)
    for cell in table["cells"]:
        cell["bbox"] = transform_box(cell["bbox"], inverse, width, height)
    return table


def evaluate(image, truth, modes, stage):
    start = time.perf_counter()
    gray, binary, inverse = preprocess(image)
    boxes, horizontal, vertical = detect(binary)
    h, w = gray.shape
    forward = cv2.invertAffineTransform(inverse)
    raw = [{"bbox": transform_box(b, inverse, w, h)} for b in boxes]
    predictions = {}
    for mode in modes:
        tables = []
        if mode == "identity":
            predictions[mode] = copy.deepcopy(truth)
            continue
        if mode == "oracle_structure":
            for original in truth["tables"]:
                table = copy.deepcopy(original)
                for cell in table["cells"]:
                    cell["text"] = ""
                    cell["bbox"] = transform_box(cell["bbox"], forward, w, h)
                recognize(gray, table)
                for cell, gt in zip(table["cells"], original["cells"]):
                    cell["bbox"] = gt["bbox"]
                tables.append(table)
        else:
            regions = (
                boxes
                if mode == "normal"
                else [
                    transform_box(t["bbox"], forward, w, h)
                    for t in truth["tables"]
                ]
            )
            for index, box in enumerate(regions):
                x0, y0, x1, y1 = box
                box = (
                    max(0, int(np.floor(x0))),
                    max(0, int(np.floor(y0))),
                    min(w, int(np.ceil(x1))),
                    min(h, int(np.ceil(y1))),
                )
                table = (
                    recover(box, horizontal, vertical, binary)
                    if "binary" in inspect.signature(recover).parameters
                    else recover(box, horizontal, vertical)
                )
                if table is not None:
                    recognize(gray, table)
                    restore(table, inverse, w, h)
                    if mode == "oracle_detection":
                        table["bbox"] = truth["tables"][index]["bbox"]
                    tables.append(table)
        predictions[mode] = {"tables": tables}
    seconds = time.perf_counter() - start
    if stage:
        stage.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(stage / "original.jpg"), image)
        cv2.imwrite(str(stage / "deskew.jpg"), gray)
        for name, mask in [
            ("binary", binary),
            ("horizontal", horizontal),
            ("vertical", vertical),
        ]:
            cv2.imwrite(str(stage / f"{name}.png"), mask)
        cv2.imwrite(
            str(stage / "boxes.jpg"), draw_boxes(image, truth["tables"], raw)
        )
    result = dict(
        raw_detection=raw,
        predictions=predictions,
        seconds_all_modes=seconds,
        estimated_angle=float(
            np.degrees(np.arctan2(inverse[0, 1], inverse[0, 0]))
        ),
    )
    return result


def aggregate(rows, modes):
    summary = {"pages": len(rows)}
    for mode in ["raw_detection"] + modes:
        entries = [r[mode] for r in rows]
        true = sum(r["true_tables"] for r in entries)
        pred = sum(r["pred_tables"] for r in entries)
        matched = sum(r["matched_tables"] for r in entries)
        result = dict(
            recall=matched / true,
            precision=matched / pred if pred else 0,
            true_tables=true,
            predicted_tables=pred,
            matched_tables=matched,
            empty_fraction=sum(r["pred_tables"] == 0 for r in entries)
            / len(rows),
        )
        if mode != "raw_detection":
            result.update(
                teds=float(np.mean([r["teds"] for r in entries])),
                teds_struct=float(
                    np.mean([r["teds_struct"] for r in entries])
                ),
                cell_f1=2
                * sum(r["cell_correct"] for r in entries)
                / max(
                    1,
                    sum(r["cell_predicted"] + r["cell_true"] for r in entries),
                ),
                numeric_accuracy=sum(r["numeric_correct"] for r in entries)
                / max(1, sum(r["numeric_total"] for r in entries)),
            )
        summary[mode] = result
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["normal", "oracle_detection", "oracle_structure", "identity"],
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    records = sample()
    hashes = frozen_hashes(records)
    lock = ROOT / "frozen.json"
    if lock.exists():
        assert json.loads(lock.read_text()) == hashes, "Frozen files changed"
    else:
        lock.write_text(json.dumps(hashes, indent=2))
    if args.limit:
        records = records[: args.limit]
    out = ROOT / args.phase
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        page_id = record["id"]
        truth = json.loads(
            Path(f"data/train/truth/{page_id}.json").read_text()
        )
        cached = out / f"{page_id}.json"
        if cached.exists():
            result = json.loads(cached.read_text())
        else:
            image = cv2.imread(f"data/train/images/{page_id}.jpg")
            result = evaluate(
                image, truth, args.modes, out / "stages" / page_id
            )
            cached.write_text(json.dumps(result, ensure_ascii=False))
        row = dict(
            id=page_id,
            angle_error=abs(
                result["estimated_angle"] - record["distortion"]["angle"]
            ),
        )
        for mode in ["raw_detection"] + args.modes:
            pred = (
                result["raw_detection"]
                if mode == "raw_detection"
                else result["predictions"][mode]["tables"]
            )
            entry = dict(
                true_tables=len(truth["tables"]),
                pred_tables=len(pred),
                matched_tables=len(match_tables(pred, truth["tables"])),
            )
            if mode != "raw_detection":
                entry.update(score(result["predictions"][mode], truth))
            row[mode] = entry
        rows.append(row)
        if len(rows) % 10 == 0:
            print(args.phase, len(rows), flush=True)
        (out / "pages.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    summary = aggregate(rows, args.modes)
    summary["angle_mae"] = float(np.mean([r["angle_error"] for r in rows]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
