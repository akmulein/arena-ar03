"""Synthetic tests for immutable, explicit-path Arena dataset export."""

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "data.py"
REVISION = "35b1c097807e0b07ec5313879b85956b7b3890db"
JPEG = b"\xff\xd8synthetic-jpeg-byte-sequence\x00\x80\xff\xd9"


def run(arguments, *, check=True):
    process = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), *map(str, arguments)],
        capture_output=True, text=True, check=check, cwd="/",
    )
    return json.loads(process.stdout) if check else process


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "source-data"
    manifest = tmp_path / "train-500.jsonl"
    rows = [{"id": f"PMC_SYNTH_{i}_1"} for i in range(500)]
    images = root / "train" / "images"
    truth = root / "train" / "truth"
    images.mkdir(parents=True)
    truth.mkdir()
    reference = '{\n "tables": [], "synthetic": "тест"\n}\n'
    for row in rows:
        (images / (row["id"] + ".jpg")).write_bytes(JPEG)
        (truth / (row["id"] + ".json")).write_text(reference)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = tmp_path / "existing-audit.json"
    audit.write_text(json.dumps({
        "status": "passed", "dataset_revision": REVISION, "seed": 42,
        "errors": [], "potential_visual_duplicates": [],
        "splits": {"train": {"subset": {"pages": 500},
                             "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}},
    }))
    output = tmp_path / "exported.jsonl"
    return {"root": root, "manifest": manifest, "audit": audit, "output": output,
            "rows": rows, "reference": reference}


def args(fixture, *extras):
    return ["--root", fixture["root"], "--manifest", fixture["manifest"],
            "--audit", fixture["audit"], "--split", "train", "--output", fixture["output"],
            *extras]


def update_audit(fixture, update):
    data = json.loads(fixture["audit"].read_text())
    update(data)
    fixture["audit"].write_text(json.dumps(data))


def update_manifest_hash(fixture):
    update_audit(fixture, lambda data: data["splits"]["train"].update({
        "manifest_sha256": hashlib.sha256(fixture["manifest"].read_bytes()).hexdigest(),
    }))


def test_export_preserves_all_ids_order_reference_and_original_jpeg_bytes(fixture):
    summary = run(args(fixture))
    records = [json.loads(line) for line in fixture["output"].read_text().splitlines()]
    assert [row["id"] for row in records] == [row["id"] for row in fixture["rows"]]
    assert len(records) == summary["pages"] == 500
    assert summary["audit_status"] == "passed"
    for record in records:
        assert record["reference"] == fixture["reference"]
        envelope = json.loads(record["input"])
        assert envelope["version"] == "ar03-page-v1"
        assert set(envelope) == {"version", "image_base64"}
        assert base64.b64decode(envelope["image_base64"], validate=True) == JPEG
        assert len(record["input"]) <= 500000
        assert len(record["reference"]) <= 500000


def test_export_does_not_overwrite_existing_output(fixture):
    fixture["output"].write_text("keep-existing")
    completed = run(args(fixture), check=False)
    assert completed.returncode != 0
    assert fixture["output"].read_text() == "keep-existing"


@pytest.mark.parametrize("status", ["failed", "unknown"])
def test_unusable_audit_status_rejected(fixture, status):
    update_audit(fixture, lambda data: data.update(status=status))
    assert run(args(fixture), check=False).returncode != 0
    assert not fixture["output"].exists()


def test_review_required_is_preserved_as_a_known_limitation(fixture):
    update_audit(fixture, lambda data: data.update(status="review_required"))
    result = run(args(fixture))
    assert result["pages"] == 500
    assert result["audit_status"] == "review_required"
    assert result["known_limitations"]


@pytest.mark.parametrize("split", ["public", "private"])
def test_operator_closed_export_uses_synthetic_data_and_emits_no_counters(fixture, split):
    (fixture["root"] / "train").rename(fixture["root"] / split)
    rows = fixture["rows"][:200]
    fixture["manifest"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = json.loads(fixture["audit"].read_text())
    audit["status"] = "review_required"
    audit["splits"] = {split: {"subset": {"pages": 200},
                              "manifest_sha256": hashlib.sha256(
                                  fixture["manifest"].read_bytes()).hexdigest()}}
    fixture["audit"].write_text(json.dumps(audit))
    arguments = args(fixture, "--operator")
    arguments[arguments.index("--split") + 1] = split
    summary = run(arguments)
    assert summary["audit_status"] == "review_required"
    assert "pages" not in summary
    assert "max_input_characters" not in summary
    assert "PMC_SYNTH" not in json.dumps(summary)
    assert len(fixture["output"].read_text().splitlines()) == 200


@pytest.mark.parametrize("split", ["public", "private"])
def test_closed_export_requires_operator_before_any_paths_are_read(tmp_path, split):
    result = run([
        "--root", tmp_path / "does-not-exist", "--manifest", tmp_path / "missing.jsonl",
        "--audit", tmp_path / "missing-audit.json", "--split", split,
        "--output", tmp_path / "no-output.jsonl",
    ], check=False)
    assert result.returncode != 0
    assert "operator" in result.stderr.lower()
    assert "missing.jsonl" not in result.stderr


@pytest.mark.parametrize("field", ["revision", "count", "hash", "errors"])
def test_existing_audit_invariants_are_checked(fixture, field):
    def change(data):
        if field == "revision":
            data["dataset_revision"] = "wrong-version"
        elif field == "count":
            data["splits"]["train"]["subset"]["pages"] = 499
        elif field == "hash":
            data["splits"]["train"]["manifest_sha256"] = "0" * 64
        else:
            data["errors"] = [{"reason": "synthetic failure"}]
    update_audit(fixture, change)
    assert run(args(fixture), check=False).returncode != 0
    assert not fixture["output"].exists()


@pytest.mark.parametrize("bad_id", ["../outside", "/absolute", "a/b", "a\\b", "x" * 101, ""])
def test_unsafe_ids_rejected_without_reading_outside_paths(fixture, bad_id):
    rows = fixture["rows"]
    rows[0]["id"] = bad_id
    fixture["manifest"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    update_manifest_hash(fixture)
    completed = run(args(fixture), check=False)
    assert completed.returncode != 0
    assert not fixture["output"].exists()


def test_duplicate_ids_and_wrong_manifest_size_fail(fixture):
    rows = fixture["rows"]
    rows[1] = rows[0]
    fixture["manifest"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    update_manifest_hash(fixture)
    assert run(args(fixture), check=False).returncode != 0
    fixture["manifest"].write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    update_manifest_hash(fixture)
    assert run(args(fixture), check=False).returncode != 0


@pytest.mark.parametrize("kind", ["image", "reference"])
def test_transport_limits_fail_atomically(fixture, kind):
    page_id = fixture["rows"][-1]["id"]
    if kind == "image":
        (fixture["root"] / "train/images" / f"{page_id}.jpg").write_bytes(JPEG + b"x" * 375000)
    else:
        (fixture["root"] / "train/truth" / f"{page_id}.json").write_text(
            json.dumps({"tables": [], "extra": "x" * 500000}))
    assert run(args(fixture), check=False).returncode != 0
    assert not fixture["output"].exists()
    assert not list(fixture["output"].parent.glob(".ar03-export-*"))


def test_largest_base64_envelope_below_the_limit_is_accepted(fixture):
    envelope_overhead = len(json.dumps(
        {"version": "ar03-page-v1", "image_base64": ""}, separators=(",", ":")))
    byte_count = ((500000 - envelope_overhead) // 4) * 3
    image = JPEG[:-2] + b"x" * (byte_count - len(JPEG)) + JPEG[-2:]
    page_id = fixture["rows"][-1]["id"]
    (fixture["root"] / "train/images" / f"{page_id}.jpg").write_bytes(image)
    result = run(args(fixture))
    assert 499995 <= result["max_input_characters"] <= 500000
    assert result["max_minimum_runner_task_bytes"] < 8 * 1024 * 1024


def test_reference_limit_counts_unicode_characters(fixture):
    page_id = fixture["rows"][-1]["id"]
    reference = json.dumps({"tables": [], "value": "я" * 300000}, ensure_ascii=False)
    assert len(reference.encode("utf-8")) > 500000
    (fixture["root"] / "train/truth" / f"{page_id}.json").write_text(reference)
    assert run(args(fixture))["max_reference_characters"] == len(reference)


@pytest.mark.parametrize("failure", ["missing-image", "invalid-truth"])
def test_source_failure_leaves_no_partial_output(fixture, failure):
    page_id = fixture["rows"][-1]["id"]
    if failure == "missing-image":
        (fixture["root"] / "train/images" / f"{page_id}.jpg").unlink()
    else:
        (fixture["root"] / "train/truth" / f"{page_id}.json").write_text("not-json")
    assert run(args(fixture), check=False).returncode != 0
    assert not fixture["output"].exists()
    assert not list(fixture["output"].parent.glob(".ar03-export-*"))


def test_symlink_image_outside_split_is_rejected(fixture, tmp_path):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(JPEG)
    image = fixture["root"] / "train/images" / (fixture["rows"][0]["id"] + ".jpg")
    image.unlink()
    image.symlink_to(outside)
    assert run(args(fixture), check=False).returncode != 0
    assert not fixture["output"].exists()


def test_output_cannot_be_written_inside_source_dataset(fixture):
    fixture["output"] = fixture["root"] / "exported.jsonl"
    assert run(args(fixture), check=False).returncode != 0
    assert not fixture["output"].exists()


def test_cli_has_no_audit_or_selection_operations(tmp_path):
    completed = run(["--help"], check=False)
    assert completed.returncode == 0
    assert "--overwrite" not in completed.stdout
    assert "--seed" not in completed.stdout
    assert "--pages" not in completed.stdout
    assert "--create" not in completed.stdout
    assert "--allow-review-required" not in completed.stdout
    assert "--review-decision" not in completed.stdout


def test_cli_requires_explicit_all_paths(tmp_path):
    completed = run([], check=False)
    assert completed.returncode != 0
    assert "required" in completed.stderr
    assert not list(tmp_path.iterdir())
