"""Export existing fixed AR-03 manifests to Arena Example JSONL.

All paths are explicit. This command never selects pages, reruns an audit,
changes source files, or decodes/re-encodes images. Closed data must be exported
inside the operator's isolated environment. --operator records the intended
role and does not provide filesystem or account isolation.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

DATASET_REVISION = "35b1c097807e0b07ec5313879b85956b7b3890db"
SPLIT_SIZES = {"train": 500, "public": 200, "private": 200}
INPUT_VERSION = "ar03-page-v1"
MAX_EXAMPLE_CHARACTERS = 500000
MAX_RUNNER_TASK_BYTES = 8 * 1024 * 1024
PAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")


class ExportError(ValueError):
    """An export failure whose message contains no dataset contents."""


def existing_audit(audit_path, manifest_path, split):
    """Check the existing audit's manifest recording; do not rerun its audit."""
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ExportError("Existing dataset audit must be an object")
    status = audit.get("status")
    if status not in {"passed", "review_required"}:
        raise ExportError("Existing dataset audit has an unusable status")
    if audit.get("dataset_revision") != DATASET_REVISION or audit.get("seed") != 42:
        raise ExportError("Existing dataset audit revision or seed differs")
    if not isinstance(audit.get("errors"), list) or audit["errors"]:
        raise ExportError("Existing dataset audit records integrity errors")
    details = audit["splits"][split]
    pages = details["subset"]["pages"]
    if type(pages) is not int or pages != SPLIT_SIZES[split]:
        raise ExportError("Existing dataset audit records a different split size")
    recorded = details["manifest_sha256"]
    if not isinstance(recorded, str) or re.fullmatch(r"[a-f0-9]{64}", recorded) is None:
        raise ExportError("Existing dataset audit lacks a valid manifest recording")
    manifest_bytes = Path(manifest_path).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != recorded:
        raise ExportError("Manifest differs from the existing dataset audit recording")
    records = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines()]
    if len(records) != SPLIT_SIZES[split]:
        raise ExportError("Manifest contains a different number of pages")
    seen = set()
    for record in records:
        page_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(page_id, str) or PAGE_ID.fullmatch(page_id) is None:
            raise ExportError("Manifest contains an invalid page ID")
        if page_id in seen:
            raise ExportError("Manifest contains duplicate page IDs")
        seen.add(page_id)
    return records, status


def source_file(split_directory, relative):
    path = split_directory / relative
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ExportError("Input must be a regular source file")
    if not path.resolve().is_relative_to(split_directory):
        raise ExportError("Input escapes the selected split directory")
    return path


def example_record(root, split, page_id):
    directory = root / split
    if directory.is_symlink() or not directory.is_dir():
        raise ExportError("Selected split must be a regular source directory")
    directory = directory.resolve()
    image = source_file(directory, Path("images") / f"{page_id}.jpg").read_bytes()
    truth_path = source_file(directory, Path("truth") / f"{page_id}.json")
    reference = truth_path.read_bytes().decode("utf-8")
    truth = json.loads(reference)
    if not isinstance(truth, dict) or not isinstance(truth.get("tables"), list):
        raise ExportError("Source truth does not have the existing table schema")
    input_text = json.dumps({
        "version": INPUT_VERSION,
        "image_base64": base64.b64encode(image).decode("ascii"),
    }, ensure_ascii=False, separators=(",", ":"))
    if len(input_text) > MAX_EXAMPLE_CHARACTERS:
        raise ExportError("Image envelope exceeds the Arena input character limit")
    if len(reference) > MAX_EXAMPLE_CHARACTERS:
        raise ExportError("Reference exceeds the Arena reference character limit")
    # Complete runner tasks also contain trusted configuration and candidate parameters.
    # The runner checks their full size at execution; this is the minimum solve task.
    minimum_task_bytes = len(json.dumps({
        "input": input_text, "uses_model": False,
    }, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if minimum_task_bytes > MAX_RUNNER_TASK_BYTES:
        raise ExportError("Input exceeds the Arena runner transport byte limit")
    return {"id": page_id, "input": input_text, "reference": reference}, minimum_task_bytes


def export_dataset(*, root, manifest, split, output, audit, operator=False):
    if split not in SPLIT_SIZES:
        raise ExportError("Unknown split")
    if split != "train" and not operator:
        raise ExportError("Closed export requires --operator in an isolated operator environment")
    root = Path(root).resolve(strict=True)
    output = Path(output)
    if not root.is_dir():
        raise ExportError("Source dataset root must be a directory")
    if output.resolve().is_relative_to(root):
        raise ExportError("Export output must be outside the source dataset")
    if output.exists() or output.is_symlink():
        raise ExportError("Export output already exists; overwriting is not supported")
    records, audit_status = existing_audit(audit, manifest, split)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    maximum_input = maximum_reference = maximum_minimum_task = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".ar03-export-",
            dir=output.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for record in records:
                example, minimum_task_bytes = example_record(root, split, record["id"])
                stream.write(json.dumps(
                    example, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
                ) + "\n")
                maximum_input = max(maximum_input, len(example["input"]))
                maximum_reference = max(maximum_reference, len(example["reference"]))
                maximum_minimum_task = max(maximum_minimum_task, minimum_task_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic create-if-absent on the same filesystem; never replace an output.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    summary = {
        "exported": True, "split": split, "input_version": INPUT_VERSION,
        "audit_status": audit_status,
        "known_limitations": (
            ["Existing audit status is review_required; the audit and selection were not rerun."]
            if audit_status == "review_required" else []
        ),
    }
    if split == "train":
        summary.update({
            "pages": len(records), "max_input_characters": maximum_input,
            "max_reference_characters": maximum_reference,
            "max_minimum_runner_task_bytes": maximum_minimum_task,
        })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True, help="Existing subset audit JSON")
    parser.add_argument("--split", choices=list(SPLIT_SIZES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operator", action="store_true",
        help="Attest the operator role for closed export; this flag does not enforce isolation",
    )
    arguments = parser.parse_args(argv)
    if arguments.split != "train" and not arguments.operator:
        parser.error("Closed export requires --operator in an isolated operator environment")
    try:
        summary = export_dataset(**vars(arguments))
    except ExportError as error:
        sys.stderr.write(f"AR-03 export failed: {error}\n")
        return 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        # Exception messages may contain a closed path, ID, reference or JSON fragment.
        sys.stderr.write(f"AR-03 export failed ({type(error).__name__})\n")
        return 1
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
