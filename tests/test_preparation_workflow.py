import json
import time

import pytest

from evaluator.build import BuildStore
from evaluator.build_receipts import bound_build_receipt
from evaluator.client import pack_candidate
from evaluator.reliability import empty_usage
from evaluator.runner import SolveResult
from runner import main as sample_runner


@pytest.fixture
def preparation_workflow(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "source.txt").write_text("alpha,beta")
    active = {}
    run_program = sample_runner.run_program

    def mounted_program(path, payload, timeout=None):
        payload = {**payload, "assets_dir": str(assets)}
        if "build_dir" in payload:
            assert payload["build_dir"] == "/output"
            payload["build_dir"] = str(active["output"])
        return run_program(active["candidate"] / path.name, payload, timeout)

    def build(candidate_dir, output):
        active.update(candidate=candidate_dir, output=output)
        started = time.monotonic()
        task = {"uses_model": False, "prepare_timeout": 10}
        manifest = candidate_dir / "solution.json"
        if manifest.exists():
            task["parameters"] = json.loads(manifest.read_text())["parameters"]
        answer = sample_runner.build(task)
        return SolveResult(answer, empty_usage(), time.monotonic() - started)

    def dependencies(candidate_dir):
        manifest = candidate_dir / "solution.json"
        if not manifest.exists():
            return []
        return json.loads(manifest.read_text())["parameters"].get("asset_digests", [])

    monkeypatch.setattr(sample_runner, "run_program", mounted_program)
    builder = BuildStore(
        tmp_path / "builds", assets / "derived", "a" * 64, "b" * 64,
        build, dependency_assets=dependencies,
    )
    return builder, active


def submit_build(builder, candidate):
    receipt, completion = builder.submit(candidate)
    assert completion is not None
    completion.result(timeout=10)
    return builder.read(receipt["candidate_digest"])


def producer(tmp_path, key="build_dir"):
    candidate = tmp_path / "producer"
    candidate.mkdir()
    (candidate / "build.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        "source = Path(payload['assets_dir']).joinpath('source.txt').read_text()\n"
        f"Path(payload[{key!r}]).joinpath('table.json').write_text(\n"
        "    json.dumps(source.upper().split(',')))\n"
        "print('{}')\n"
    )
    return pack_candidate("producer", candidate)


def test_real_build_publication_and_selected_solver_preserve_dependency_costs(
    tmp_path, preparation_workflow,
):
    builder, active = preparation_workflow
    source = producer(tmp_path)
    prepared = submit_build(builder, source)
    assert prepared["status"] == "FINISHED"
    assert bound_build_receipt(builder.root, prepared["output_digest"]) == prepared
    published = builder.publisher.store_root / prepared["output_digest"] / "table.json"
    assert json.loads(published.read_text()) == ["ALPHA", "BETA"]
    assert published.stat().st_mode & 0o222 == 0

    consumer = tmp_path / "consumer"
    consumer.mkdir()
    parameters = {"asset_digests": [prepared["output_digest"]]}
    (consumer / "solution.json").write_text(json.dumps({
        "harness": "program", "parameters": parameters,
    }))
    (consumer / "build.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        "Path(payload['build_dir']).joinpath('selection.json').write_text(\n"
        "    json.dumps(payload['parameters']['asset_digests']))\n"
        "print('{}')\n"
    )
    (consumer / "solve.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        "selected = payload['parameters']['asset_digests'][0]\n"
        "table = json.loads(Path(payload['assets_dir']).joinpath(\n"
        "    'derived', selected, 'table.json').read_text())\n"
        "print(json.dumps({'answer': table[int(payload['input'])]}))\n"
    )
    measured = submit_build(builder, pack_candidate("consumer", consumer))
    assert measured["status"] == "FINISHED"
    assert measured["dependency_receipts"] == [prepared["candidate_digest"]]
    assert measured["resources"]["solver"]["wall_seconds"] == pytest.approx(
        measured["own_resources"]["solver"]["wall_seconds"]
        + prepared["own_resources"]["solver"]["wall_seconds"],
    )
    active["candidate"] = builder.root / "jobs" / measured["candidate_digest"] / "candidate"
    assert sample_runner.solve({
        "uses_model": False, "input": "1", "parameters": parameters,
    }) == "BETA"


def test_wrong_build_output_field_preserves_failed_receipt_without_publication(
    tmp_path, preparation_workflow,
):
    builder, _ = preparation_workflow
    candidate = producer(tmp_path, key="output_dir")
    failed = submit_build(builder, candidate)
    assert failed["status"] == "FAILED"
    assert failed["error"]["type"] == "CalledProcessError"
    assert "output_digest" not in failed
    assert list((builder.root / "bindings").iterdir()) == []
    assert failed["resources"]["solver"]["wall_seconds"] > 0
    assert builder.read(candidate.digest()) == failed
