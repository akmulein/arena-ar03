"""Stream archives, retaining selected full pages and complete annotations."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np

from scripts.augment import distort
from solution.schema import validate

ROOT = Path("data")
BASE = "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/35b1c097807e0b07ec5313879b85956b7b3890db/"
SPLITS = {
    "train": ("Train_Part1", 15998),
    "public": ("Val", 2000),
    "private": ("Test", 2000),
}

# Retained on disk, excluded from active manifests by the Arena input limit.
EXCLUDED_PAGES = {"PMC3287911_1", "PMC2727000_2"}


def check_storage():
    used = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    if used > 118_000_000_000 or shutil.disk_usage(ROOT).free < 2_000_000_000:
        raise RuntimeError("Storage budget reached; preparation stopped")


def index_pdf():
    db = sqlite3.connect(ROOT / "annotations.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS pages "
        "(id TEXT PRIMARY KEY, doc TEXT, split TEXT, tables TEXT)"
    )
    if db.execute("SELECT count(*) FROM pages").fetchone()[0]:
        return db
    verify_archive("pdf.tar.gz", "PubTables-1M-PDF_Annotations.tar.gz")
    with tarfile.open(ROOT / "raw/pdf.tar.gz", "r|gz") as archive:
        for number, member in enumerate(archive):
            if number % 10000 == 0:
                check_storage()
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            tables = json.load(archive.extractfile(member))
            pages = {}
            for table in tables:
                table["cells"] = [
                    {
                        k: cell[k]
                        for k in (
                            "row_nums",
                            "column_nums",
                            "pdf_bbox",
                            "pdf_text_content",
                        )
                    }
                    for cell in table["cells"]
                ]
                page_id = f"{table['pmc_id']}_{table['pdf_page_index']}"
                pages.setdefault(page_id, []).append(table)
            for page_id, tables in pages.items():
                db.execute(
                    "INSERT INTO pages VALUES (?, ?, ?, ?)",
                    (
                        page_id,
                        tables[0]["pmc_id"],
                        tables[0]["split"],
                        json.dumps(tables),
                    ),
                )
    db.commit()
    print(
        "PDF index complete:",
        db.execute("SELECT count(*) FROM pages").fetchone()[0],
        flush=True,
    )
    return db


def convert(tables, width, height):
    result = {"tables": []}
    for table in tables:
        if table["exclude_for_detection"] or table["exclude_for_structure"]:
            raise ValueError("excluded source annotation")
        x0, y0, x1, y1 = table["pdf_full_page_bbox"]

        def box(values):
            return [
                (values[0] - x0) * width / (x1 - x0),
                (values[1] - y0) * height / (y1 - y0),
                (values[2] - x0) * width / (x1 - x0),
                (values[3] - y0) * height / (y1 - y0),
            ]

        cells = []
        for cell in table["cells"]:
            rr, cc = sorted(cell["row_nums"]), sorted(cell["column_nums"])
            if rr != list(range(rr[0], rr[-1] + 1)) or cc != list(
                range(cc[0], cc[-1] + 1)
            ):
                raise ValueError("nonrectangular source cell")
            cells.append(
                dict(
                    r0=rr[0],
                    r1=rr[-1],
                    c0=cc[0],
                    c1=cc[-1],
                    text=cell["pdf_text_content"],
                    bbox=box(cell["pdf_bbox"]),
                )
            )
        result["tables"].append(
            dict(
                bbox=box(table["pdf_table_bbox"]),
                rows=len(table["rows"]),
                cols=len(table["columns"]),
                cells=cells,
            )
        )
    return validate(result)


def prepare(split, db, seed):
    source, target = SPLITS[split]
    official = {"train": "train", "public": "val", "private": "test"}[split]
    with tarfile.open(ROOT / "raw/filelists.tar.gz") as archive:
        allowed = {
            Path(line.decode().strip()).stem
            for line in archive.extractfile(f"{official}_filelist.txt")
        }
    directory = ROOT / split
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "truth").mkdir(exist_ok=True)
    manifest_path = directory / "manifest.jsonl"
    records = (
        [json.loads(line) for line in manifest_path.read_text().splitlines()]
        if manifest_path.exists()
        else []
    )
    selected = {r["id"] for r in records}
    if any(r["seed"] != seed for r in records):
        raise ValueError("Existing manifest uses a different seed")
    if len(records) >= target:
        return
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB free")
    url = BASE + f"PubTables-1M-Detection_Images_{source}.tar.gz?download=true"
    process = subprocess.Popen(
        ["curl", "-sSL", "--fail", "--retry", "3", url], stdout=subprocess.PIPE
    )
    try:
        with (
            tarfile.open(fileobj=process.stdout, mode="r|gz") as archive,
            manifest_path.open("a") as manifest,
        ):
            for member in archive:
                page_id = Path(member.name).stem
                if (
                    not member.isfile()
                    or page_id not in allowed
                    or page_id in selected
                    or page_id in EXCLUDED_PAGES
                ):
                    continue
                row = db.execute(
                    "SELECT doc, split, tables FROM pages WHERE id=?",
                    (page_id,),
                ).fetchone()
                if row is None or row[1] != official:
                    continue
                image = cv2.imdecode(
                    np.frombuffer(
                        archive.extractfile(member).read(), np.uint8
                    ),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise ValueError(f"corrupt image: {page_id}")
                try:
                    truth = convert(
                        json.loads(row[2]), image.shape[1], image.shape[0]
                    )
                    encoded, truth, profile = distort(
                        image, truth, page_id, seed
                    )
                    validate(truth)
                except (ValueError, TypeError):
                    continue
                image_path = directory / "images" / f"{page_id}.jpg"
                image_path.write_bytes(encoded.tobytes())
                (directory / "truth" / f"{page_id}.json").write_text(
                    json.dumps(truth, ensure_ascii=False)
                )
                record = dict(
                    id=page_id,
                    document=row[0],
                    source=official,
                    seed=seed,
                    sha256=hashlib.sha256(encoded).hexdigest(),
                    distortion=profile,
                )
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                records.append(record)
                selected.add(page_id)
                if len(records) % 100 == 0:
                    print(split, len(records), flush=True)
                if len(records) % 1000 == 0:
                    check_storage()
                if len(records) == target:
                    break
        if len(records) != target:
            raise RuntimeError(
                f"{split}: only {len(records)}/{target} complete pages"
            )
    finally:
        process.terminate()
        process.wait()


def verify_archive(local, remote):
    metadata = json.loads(Path("docs/upstream/dataset-files.json").read_text())
    expected = next(item for item in metadata if item["path"] == remote)
    with (ROOT / "raw" / local).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected["lfs"]["oid"]:
        raise ValueError(f"Archive checksum mismatch: {local}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    verify_archive(
        "filelists.tar.gz", "PubTables-1M-Detection_Filelists.tar.gz"
    )
    manifests = [ROOT / split / "manifest.jsonl" for split in SPLITS]
    if all(path.exists() for path in manifests):
        records = [
            [json.loads(line) for line in path.read_text().splitlines()]
            for path in manifests
        ]
        if any(r["seed"] != args.seed for split in records for r in split):
            raise ValueError("Existing manifest uses a different seed")
        if all(
            len(rows) == target
            for rows, (_, target) in zip(records, SPLITS.values())
        ):
            from scripts.audit import audit

            audit()
            return
    db = index_pdf()
    for split in SPLITS:
        prepare(split, db, args.seed)
    db.close()


if __name__ == "__main__":
    main()
