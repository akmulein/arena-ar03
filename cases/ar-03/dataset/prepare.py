"""Build the fixed dataset splits from streamed page archives."""

import argparse
import hashlib
import json
import sqlite3
import subprocess
import tarfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ar03_solver.common.schema import validate
from dataset.annotations import convert, index_pdf
from dataset.audit import audit
from dataset.augment import distort
from dataset.config import (
    ARCHIVES,
    BASE,
    DEFAULT_SEED,
    EXCLUDED_PAGES,
    MIN_PREPARE_FREE_BYTES,
    OFFICIAL_SPLITS,
    ROOT,
    SPLITS,
)
from dataset.storage import (
    check_storage,
    read_filelist,
    read_manifest,
    require_free_space,
    verify_archive,
)


@contextmanager
def stream_images(source: str) -> Iterator[tarfile.TarFile]:
    """Read pages in upstream archive order without storing the archive."""
    url = f"{BASE}PubTables-1M-Detection_Images_{source}.tar.gz?download=true"
    process = subprocess.Popen(
        ["curl", "-sSL", "--fail", "--retry", "3", url],
        stdout=subprocess.PIPE,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("Cannot open curl output stream")
        with (
            process.stdout,
            tarfile.open(fileobj=process.stdout, mode="r|gz") as archive,
        ):
            yield archive
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()


def existing_records(path: Path, seed: int) -> list[dict]:
    records = read_manifest(path) if path.exists() else []
    if any(record["seed"] != seed for record in records):
        raise ValueError(f"Existing manifest uses a different seed: {path}")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError(f"Duplicate page IDs in manifest: {path}")
    return records


def decode_page(archive: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Cannot read image: {member.name}")
    with stream:
        encoded = np.frombuffer(stream.read(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Corrupt image: {member.name}")
    return image


def prepare_page(
    image: np.ndarray, annotations: str, page_id: str, seed: int
) -> tuple[np.ndarray, dict, dict]:
    """Convert source labels and apply the fixed scan simulation."""
    height, width = image.shape[:2]
    truth = convert(json.loads(annotations), width, height)
    encoded, truth, profile = distort(image, truth, page_id, seed)
    return encoded, validate(truth), profile


def write_page(
    directory: Path,
    page_id: str,
    document: str,
    official: str,
    seed: int,
    encoded: np.ndarray,
    truth: dict,
    profile: dict,
) -> dict:
    image_bytes = encoded.tobytes()
    (directory / "images" / f"{page_id}.jpg").write_bytes(image_bytes)
    (directory / "truth" / f"{page_id}.json").write_text(
        json.dumps(truth, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "id": page_id,
        "document": document,
        "source": official,
        "seed": seed,
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "distortion": profile,
    }


def prepare(split: str, db: sqlite3.Connection, seed: int, root: Path = ROOT) -> None:
    source, target = SPLITS[split]
    official = OFFICIAL_SPLITS[split]
    allowed = read_filelist(official, root)

    directory = root / split
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "truth").mkdir(exist_ok=True)
    manifest_path = directory / "manifest.jsonl"
    records = existing_records(manifest_path, seed)
    selected = {record["id"] for record in records}
    count = len(records)

    if count > target:
        raise ValueError(
            f"{split}: manifest has {count} pages; expected at most {target}"
        )
    if count == target:
        return

    require_free_space(root, MIN_PREPARE_FREE_BYTES)

    with (
        stream_images(source) as archive,
        manifest_path.open("a", encoding="utf-8") as manifest,
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
                "SELECT doc, split, tables FROM pages WHERE id = ?", (page_id,)
            ).fetchone()
            if row is None or row[1] != official:
                continue

            document, _, annotations = row
            image = decode_page(archive, member)
            try:
                encoded, truth, profile = prepare_page(
                    image, annotations, page_id, seed
                )
            except (ValueError, TypeError):
                # Preserve the original policy: skip invalid source annotations.
                continue

            record = write_page(
                directory, page_id, document, official, seed, encoded, truth, profile
            )
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            selected.add(page_id)
            count += 1

            if count % 100 == 0:
                print(f"{split}: {count}/{target}", flush=True)
            if count % 1000 == 0:
                check_storage(root)
            if count == target:
                break

    if count != target:
        raise RuntimeError(f"{split}: only {count}/{target} complete pages")


def dataset_is_complete(root: Path, seed: int) -> bool:
    paths = {split: root / split / "manifest.jsonl" for split in SPLITS}
    if not all(path.exists() for path in paths.values()):
        return False
    records = {split: existing_records(path, seed) for split, path in paths.items()}
    return all(len(records[split]) == target for split, (_, target) in SPLITS.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    cv2.setNumThreads(1)
    verify_archive(*ARCHIVES["filelists"])
    if dataset_is_complete(ROOT, args.seed):
        audit(seed=args.seed)
        return

    with closing(index_pdf()) as db:
        for split in SPLITS:
            prepare(split, db, args.seed)

    audit(seed=args.seed)


if __name__ == "__main__":
    main()
