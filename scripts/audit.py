import hashlib
import json
import tarfile
from pathlib import Path

import cv2

from scripts.prepare import EXCLUDED_PAGES, SPLITS
from solution.schema import validate


def audit(root=Path("data")):
    documents = {}
    counts = {}
    fingerprints = {}
    for split, (_, target) in SPLITS.items():
        official = {"train": "train", "public": "val", "private": "test"}[
            split
        ]
        with tarfile.open(root / "raw/filelists.tar.gz") as archive:
            allowed = {
                Path(line.decode().strip()).stem
                for line in archive.extractfile(f"{official}_filelist.txt")
            }
        records = [
            json.loads(line)
            for line in (root / split / "manifest.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(records) == target, (split, len(records), target)
        assert len({r["id"] for r in records}) == target
        active_ids = {r["id"] for r in records}
        assert not active_ids & EXCLUDED_PAGES
        for folder, pattern in (("images", "*.jpg"), ("truth", "*.json")):
            stored_ids = {p.stem for p in (root / split / folder).glob(pattern)}
            assert active_ids <= stored_ids
            assert stored_ids - active_ids <= EXCLUDED_PAGES
        documents[split] = {r["document"] for r in records}
        fingerprint = hashlib.sha256()
        for record in records:
            assert record["id"] in allowed
            assert record["source"] == official
            assert record["seed"] == 42
            assert record["id"].rsplit("_", 1)[0] == record["document"]
            image_path = root / split / "images" / (record["id"] + ".jpg")
            assert (
                hashlib.sha256(image_path.read_bytes()).hexdigest()
                == record["sha256"]
            )
            image = cv2.imread(str(image_path))
            assert image is not None
            truth_path = root / split / "truth" / (record["id"] + ".json")
            fingerprint.update(record["sha256"].encode())
            fingerprint.update(truth_path.read_bytes())
            truth = validate(
                json.loads(
                    (
                        root / split / "truth" / (record["id"] + ".json")
                    ).read_text()
                )
            )
            for table in truth["tables"]:
                for box in [table["bbox"]] + [
                    c["bbox"] for c in table["cells"]
                ]:
                    assert 0 <= box[0] < box[2] <= image.shape[1]
                    assert 0 <= box[1] < box[3] <= image.shape[0]
        counts[split] = len(records)
        fingerprints[split] = fingerprint.hexdigest()
        print(split, "verified", len(records), flush=True)
    for a, b in [
        ("train", "public"),
        ("train", "private"),
        ("public", "private"),
    ]:
        assert not documents[a] & documents[b], (a, b)
    result = dict(
        counts=counts,
        document_counts={k: len(v) for k, v in documents.items()},
        document_intersections=0,
        fingerprints=fingerprints,
        bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
    )
    Path("reports/dataset-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
