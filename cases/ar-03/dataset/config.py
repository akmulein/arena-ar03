"""Fixed dataset configuration and project paths."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT_ROOT.parents[1]
RESOURCES = Path(
    os.environ.get("AR03_RESOURCES", str(REPOSITORY.parent / f"{REPOSITORY.name}-operator"))
)
ROOT = RESOURCES / "data"
REPORTS_DIR = RESOURCES / "reports"
METADATA_PATH = Path(__file__).resolve().with_name("dataset-files.json")

REVISION = "35b1c097807e0b07ec5313879b85956b7b3890db"
BASE = f"https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/{REVISION}/"

DEFAULT_SEED = 42

# Local split -> (upstream image archive suffix, number of pages).
SPLITS = {
    "train": ("Train_Part1", 15_998),
    "public": ("Val", 2_000),
    "private": ("Test", 2_000),
}
OFFICIAL_SPLITS = {"train": "train", "public": "val", "private": "test"}

# These pages exceed the Arena input size limit.
EXCLUDED_PAGES = frozenset({"PMC3287911_1", "PMC2727000_2"})

# Archive key -> (local filename, upstream filename).
ARCHIVES = {
    "pdf": ("pdf.tar.gz", "PubTables-1M-PDF_Annotations.tar.gz"),
    "filelists": ("filelists.tar.gz", "PubTables-1M-Detection_Filelists.tar.gz"),
}

MAX_DATA_BYTES = 118_000_000_000
MIN_FREE_BYTES = 2_000_000_000
MIN_PREPARE_FREE_BYTES = 10 * 1024**3
