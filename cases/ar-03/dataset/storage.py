"""File and archive operations shared by dataset commands."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from dataset.config import (
    ARCHIVES,
    MAX_DATA_BYTES,
    METADATA_PATH,
    MIN_FREE_BYTES,
    ROOT,
)


def directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def require_free_space(root: Path, minimum: int) -> None:
    if shutil.disk_usage(root).free < minimum:
        raise RuntimeError(
            f"Insufficient free disk space: {root} (need {minimum} bytes)"
        )


def check_storage(root: Path = ROOT) -> None:
    if directory_size(root) > MAX_DATA_BYTES:
        raise RuntimeError(f"Dataset storage budget exceeded: {root}")
    require_free_space(root, MIN_FREE_BYTES)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_metadata(remote: str) -> dict:
    entries = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    for entry in entries:
        if entry["path"] == remote:
            return entry
    raise ValueError(f"Archive missing from metadata: {remote}")


def verify_archive(local: str, remote: str, root: Path = ROOT) -> None:
    path = root / "raw" / local
    expected = archive_metadata(remote)
    if sha256_file(path) != expected["lfs"]["oid"]:
        raise ValueError(f"Archive checksum mismatch: {path}")


def read_filelist(official_split: str, root: Path = ROOT) -> set[str]:
    local, _ = ARCHIVES["filelists"]
    with tarfile.open(root / "raw" / local) as archive:
        member = archive.extractfile(f"{official_split}_filelist.txt")
        if member is None:
            raise ValueError(f"Filelist missing for split: {official_split}")
        with member:
            return {Path(line.decode("utf-8").strip()).stem for line in member}


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]
