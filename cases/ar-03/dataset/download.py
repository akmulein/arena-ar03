"""Download and verify the required PubTables-1M metadata archives."""

import argparse
import hashlib
from pathlib import Path

import requests

from dataset.config import ARCHIVES, BASE, MIN_FREE_BYTES, ROOT
from dataset.storage import archive_metadata, require_free_space, verify_archive

CHUNK_SIZE = 1024 * 1024


def download_archive(key: str, root: Path = ROOT) -> Path:
    local, remote = ARCHIVES[key]
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / local

    if destination.exists():
        verify_archive(local, remote, root)
        print(f"{local}: already downloaded and verified", flush=True)
        return destination

    metadata = archive_metadata(remote)
    require_free_space(root, metadata["size"] + MIN_FREE_BYTES)
    temporary = destination.with_suffix(".part")
    digest = hashlib.sha256()

    try:
        with requests.get(BASE + remote, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        output.write(chunk)
                        digest.update(chunk)

        if digest.hexdigest() != metadata["lfs"]["oid"]:
            raise ValueError(f"Archive checksum mismatch: {remote}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"{local}: downloaded and verified", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=ARCHIVES, help="Download one archive only")
    args = parser.parse_args()

    for key in ([args.only] if args.only else ARCHIVES):
        download_archive(key)


if __name__ == "__main__":
    main()
