"""Download and verify the two required annotation archives."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests

from scripts.prepare import BASE, ROOT, verify_archive

ARCHIVES = {
    "pdf": ("pdf.tar.gz", "PubTables-1M-PDF_Annotations.tar.gz"),
    "filelists": (
        "filelists.tar.gz",
        "PubTables-1M-Detection_Filelists.tar.gz",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=ARCHIVES)
    args = parser.parse_args()
    metadata = json.loads(Path("docs/upstream/dataset-files.json").read_text())
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    for key, (local, remote) in ARCHIVES.items():
        if args.only and args.only != key:
            continue
        path = ROOT / "raw" / local
        if path.exists():
            verify_archive(local, remote)
            continue
        expected = next(m for m in metadata if m["path"] == remote)
        if shutil.disk_usage(ROOT).free < expected["size"] + 2_000_000_000:
            raise RuntimeError("Insufficient free disk space")
        temporary = path.with_suffix(".part")
        digest = hashlib.sha256()
        with requests.get(
            BASE + remote, stream=True, timeout=(30, 60)
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected["lfs"]["oid"]:
            raise ValueError(f"Checksum mismatch: {remote}")
        temporary.replace(path)
        print(local, "verified", flush=True)


if __name__ == "__main__":
    main()
