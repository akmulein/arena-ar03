"""Verify manifest integrity, labels, images, and split independence."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import cv2

from common.schema import validate
from dataset.config import (
    DEFAULT_SEED,
    EXCLUDED_PAGES,
    OFFICIAL_SPLITS,
    REPORTS_DIR,
    ROOT,
    SPLITS,
)
from dataset.storage import directory_size, read_filelist, read_manifest, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_split_files(directory: Path, active_ids: set[str]) -> None:
    for folder, pattern in (("images", "*.jpg"), ("truth", "*.json")):
        stored_ids = {path.stem for path in (directory / folder).glob(pattern)}
        missing = active_ids - stored_ids
        unexpected = stored_ids - active_ids - EXCLUDED_PAGES
        require(
            not missing,
            f"{directory.name}/{folder}: missing IDs {sorted(missing)[:10]}",
        )
        require(
            not unexpected,
            f"{directory.name}/{folder}: unexpected IDs {sorted(unexpected)[:10]}",
        )


def check_bbox(box, width: int, height: int, page_id: str) -> None:
    require(
        len(box) == 4
        and 0 <= box[0] < box[2] <= width
        and 0 <= box[1] < box[3] <= height,
        f"{page_id}: bbox outside image or degenerate: {box}",
    )


def check_truth(truth: dict, width: int, height: int, page_id: str) -> None:
    validate(truth)
    for table in truth["tables"]:
        check_bbox(table["bbox"], width, height, page_id)
        for cell in table["cells"]:
            check_bbox(cell["bbox"], width, height, page_id)


def audit_split(split: str, root: Path, seed: int) -> tuple[int, set[str], str]:
    directory = root / split
    official = OFFICIAL_SPLITS[split]
    _, target = SPLITS[split]
    allowed = read_filelist(official, root)
    records = read_manifest(directory / "manifest.jsonl")
    active_ids = {record["id"] for record in records}

    require(
        len(records) == target,
        f"{split}: expected {target} pages, got {len(records)}",
    )
    require(len(active_ids) == target, f"{split}: duplicate page IDs")
    require(not active_ids & EXCLUDED_PAGES, f"{split}: excluded pages in manifest")
    check_split_files(directory, active_ids)

    documents = set()
    fingerprint = hashlib.sha256()
    for record in records:
        page_id = record["id"]
        document = record["document"]
        require(page_id in allowed, f"{split}/{page_id}: page not in official filelist")
        require(record["source"] == official, f"{page_id}: incorrect source split")
        require(record["seed"] == seed, f"{page_id}: unexpected seed")
        require(
            page_id.rsplit("_", 1)[0] == document,
            f"{page_id}: incorrect document ID",
        )
        documents.add(document)

        image_path = directory / "images" / f"{page_id}.jpg"
        require(
            sha256_file(image_path) == record["sha256"],
            f"{page_id}: image checksum mismatch",
        )
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"{page_id}: cannot decode image")
        height, width = image.shape[:2]

        truth_bytes = (directory / "truth" / f"{page_id}.json").read_bytes()
        check_truth(json.loads(truth_bytes), width, height, page_id)
        # Preserve the original fingerprint format and manifest order.
        fingerprint.update(record["sha256"].encode("utf-8"))
        fingerprint.update(truth_bytes)

    print(f"{split}: verified {len(records)} pages", flush=True)
    return len(records), documents, fingerprint.hexdigest()


def audit(
    root: Path = ROOT,
    *,
    seed: int = DEFAULT_SEED,
    report_path: Path = REPORTS_DIR / "dataset-audit.json",
) -> dict:
    counts = {}
    documents = {}
    fingerprints = {}

    for split in SPLITS:
        count, document_ids, fingerprint = audit_split(split, root, seed)
        counts[split] = count
        documents[split] = document_ids
        fingerprints[split] = fingerprint

    for first, second in combinations(SPLITS, 2):
        overlap = documents[first] & documents[second]
        require(
            not overlap,
            f"Document overlap between {first} and {second}: {sorted(overlap)[:10]}",
        )

    result = {
        "counts": counts,
        "document_counts": {split: len(ids) for split, ids in documents.items()},
        "document_intersections": 0,
        "fingerprints": fingerprints,
        "bytes": directory_size(root),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(result, indent=2)
    report_path.write_text(report, encoding="utf-8")
    print(report, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    audit(seed=args.seed)


if __name__ == "__main__":
    main()
