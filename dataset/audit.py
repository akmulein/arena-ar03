"""Verify manifest integrity, labels, images, and split independence."""

import argparse
import hashlib
import json
from itertools import combinations
from collections import Counter

import numpy as np
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
    REVISION,
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


def perceptual_hash(image) -> int:
    """64-bit DCT pHash; a screening heuristic, not a duplicate proof."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coeff = cv2.dct(small.astype(np.float32))[:8, :8].ravel()
    bits = coeff > np.median(coeff[1:])
    bits[0] = False  # ignore the DC brightness coefficient
    return sum(int(bit) << i for i, bit in enumerate(bits))


def audit_subsets(root, manifests, report_path, *, seed=DEFAULT_SEED, phash_distance=6):
    """Audit only chosen pages. Run in a trusted operator environment."""
    from dataset.subsets import SIZES, load_population, profile, differences

    require(0 <= phash_distance <= 63, "Invalid pHash distance")
    errors = []
    near = []
    found = []
    details = {}
    for split, path in manifests.items():
        path = Path(path)
        records = read_manifest(path)
        population, features = load_population(root, split)
        available = {r["id"]: r for r in population}
        allowed = read_filelist(OFFICIAL_SPLITS[split], root)
        require(len(records) == SIZES[split], f"{split}: expected {SIZES[split]} pages")
        require(len({r["id"] for r in records}) == len(records), f"{split}: duplicate IDs")
        # Check hashes and seed against the selection report, if one was created.
        selection_path = path.with_name(path.stem + "-selection.json")
        cap = 2
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            require(selection["manifest_sha256"] == sha256_file(path), f"{split}: manifest changed")
            require(selection["source_manifest_sha256"] == sha256_file(root / split / "manifest.jsonl"),
                    f"{split}: source manifest changed")
            require(selection["seed"] == seed and selection["dataset_revision"] == REVISION,
                    f"{split}: selection seed or revision mismatch")
            require(selection["split"] == split, f"{split}: incorrect selection report")
            cap = selection["max_pages_per_document"]
        doc_counts = Counter(r["document"] for r in records)
        require(max(doc_counts.values()) <= cap, f"{split}: document cap exceeded")
        fingerprint = hashlib.sha256()
        for record in records:
            pid = record["id"]
            require(available.get(pid) == record, f"{split}/{pid}: not an unchanged source record")
            require(pid in allowed and pid not in EXCLUDED_PAGES, f"{split}/{pid}: outside official split")
            require(record["seed"] == seed, f"{split}/{pid}: unexpected augmentation seed")
            directory = root / split
            image_path = directory / "images" / f"{pid}.jpg"
            actual_hash = sha256_file(image_path)
            require(actual_hash == record["sha256"], f"{split}/{pid}: image hash mismatch")
            image = cv2.imread(str(image_path))
            require(image is not None, f"{split}/{pid}: cannot decode image")
            h, w = image.shape[:2]
            truth_bytes = (directory / "truth" / f"{pid}.json").read_bytes()
            check_truth(json.loads(truth_bytes), w, h, pid)
            fingerprint.update(actual_hash.encode("ascii"))
            fingerprint.update(hashlib.sha256(truth_bytes).digest())
            found.append((split, pid, record["document"], actual_hash, perceptual_hash(image)))
        pop, chosen = profile(population, features), profile(records, features)
        details[split] = {
            "manifest_sha256": sha256_file(path), "image_truth_fingerprint": fingerprint.hexdigest(),
            "population": pop, "subset": chosen, "fraction_differences": differences(pop, chosen),
        }
    for a, b in combinations(found, 2):
        sa, pa, da, ha, pha = a
        sb, pb, db, hb, phb = b
        pair = {"first": f"{sa}/{pa}", "second": f"{sb}/{pb}"}
        cross = sa != sb
        if cross and (pa == pb or da == db):
            errors.append({**pair, "reason": "cross-split page/document overlap"})
        if ha == hb:
            errors.append({**pair, "reason": "exact image duplicate"})
        elif (distance := (pha ^ phb).bit_count()) <= phash_distance:
            near.append({**pair, "distance": distance, "cross_split": cross})
    near.sort(key=lambda p: (not p["cross_split"], p["distance"], p["first"], p["second"]))
    result = {
        "status": "failed" if errors else "review_required" if near else "passed",
        "dataset_revision": REVISION, "seed": seed, "splits": details,
        "errors": errors, "phash_distance": phash_distance,
        "potential_visual_duplicates": near,
        "note": "pHash is a screening heuristic, not proof of absence of near duplicates or pretraining leakage",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit: {result['status']}; exact/overlap errors={len(errors)}, pHash pairs={len(near)}; {report_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--subsets", action="store_true", help="Audit 500/200/200 instead of full dataset")
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--public-manifest", type=Path)
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--phash-distance", type=int, default=6)
    args = parser.parse_args()
    if not args.subsets:
        audit(root=args.root, seed=args.seed, report_path=args.report or REPORTS_DIR / "dataset-audit.json")
        return
    directory = args.manifest_dir or args.root / "subsets"
    manifests = {
        "train": args.train_manifest or directory / "train-500.jsonl",
        "public": args.public_manifest or directory / "public-200.jsonl",
        "private": args.private_manifest or directory / "private-200.jsonl",
    }
    result = audit_subsets(args.root, manifests, args.report or directory / "arena-subsets-audit.json",
                           seed=args.seed, phash_distance=args.phash_distance)
    if result["status"] != "passed":
        raise SystemExit(1 if result["status"] == "failed" else 2)


if __name__ == "__main__":
    main()
