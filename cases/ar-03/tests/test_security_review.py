"""Only the four security-review regressions; no solver or candidate is executed."""

import importlib
import json
import os
import time
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


@pytest.fixture
def security(monkeypatch):
    monkeypatch.syspath_prepend(str(DEPLOY))
    monkeypatch.delenv("AR03_BOOT_FINGERPRINT", raising=False)
    return importlib.import_module("security")


@pytest.fixture
def trusted_tree(tmp_path, security, monkeypatch):
    repo = tmp_path / "arena"
    deploy = repo / "cases/ar-03/deploy"
    for name in (
        "cases/ar-03/deploy/security.py",
        "cases/ar-03/train.py",
        "cases/ar-03/monitor.py",
        "cases/ar-03/operator_artifacts.py",
        "evaluator/runner.py",
        "evaluator/execution.py",
        "runner/main.py",
        "pyproject.toml",
        "uv.lock",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original trusted source\n")
    root = tmp_path / "operator/runs/test"
    (root / "private/metric").mkdir(parents=True)
    (root / "private/metric/score.py").write_text("# frozen metric\n")
    (root / "config.json").write_text('{"services":{"train":"local"}}\n')
    monkeypatch.setattr(security, "DEPLOY", deploy)
    return root, deploy / "security.py"


def test_changed_config_requires_restart_even_after_restore(security, trusted_tree):
    root, _ = trusted_tree
    gate = security.SafetyGate(root)
    path = root / "config.json"
    original, metadata = path.read_bytes(), path.stat()
    path.write_bytes(original + b" ")
    with pytest.raises(HTTPException, match="restart required"):
        gate.verify()
    path.write_bytes(original)
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    with pytest.raises(HTTPException, match="restart required"):
        gate.verify()


def test_changed_trusted_code_cannot_be_reverified(security, trusted_tree):
    root, source = trusted_tree
    gate = security.SafetyGate(root)
    source.write_text("# changed trusted source\n")
    with pytest.raises(HTTPException, match="restart required"):
        gate.verify()


def test_new_control_cannot_authorize_old_backend(security, trusted_tree):
    root, source = trusted_tree
    original = security.SafetyGate(root)
    source.write_text("# newer operator source\n")
    replacement = security.SafetyGate(root)
    (root / "private/processes.json").write_text(
        json.dumps(
            [
                {
                    "role": "train",
                    "pid": 123,
                    "started_at": "old-process",
                    "startup_fingerprint": original.startup_fingerprint,
                }
            ]
        )
    )
    with pytest.raises(HTTPException, match="restart required"):
        replacement.verify()


def test_container_restart_changes_identity_without_changing_id(security):
    row = {
        "Id": "same-id",
        "Image": "same-image",
        "Config": {},
        "HostConfig": {},
        "NetworkSettings": {"Networks": {}},
        "Mounts": [],
        "State": {"StartedAt": "first"},
    }
    first = security.digest(security.runtime_signature(row))
    row["State"]["StartedAt"] = "second"
    assert first != security.digest(security.runtime_signature(row))


def test_pid_reuse_with_different_start_time_is_rejected(security, trusted_tree, monkeypatch):
    root, _ = trusted_tree
    (root / "private/processes.json").write_text(
        json.dumps(
            [
                {
                    "role": "train",
                    "pid": 123,
                    "started_at": "first",
                    "startup_fingerprint": "same-code",
                }
            ]
        )
    )
    monkeypatch.setattr(
        security,
        "process_identity",
        lambda pid: {"started_at": "second", "command": str(root) + " serve"},
    )
    with pytest.raises(ValueError, match="start time"):
        security.operator_processes(root, "same-code")


@pytest.fixture
def model_assets(tmp_path, security):
    root = tmp_path / "runner-assets"
    for name in security.APPROVED_MODEL_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic model placeholder")
    return root


def test_original_eight_model_files_are_accepted(security, model_assets):
    assert len(security.APPROVED_MODEL_FILES) == 8
    security.validate_assets(model_assets)


def test_extra_asset_file_blocks_validation(security, model_assets):
    (model_assets / "models/notes.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="approved model files"):
        security.validate_assets(model_assets)


def test_missing_asset_file_blocks_validation(security, model_assets):
    (model_assets / sorted(security.APPROVED_MODEL_FILES)[0]).unlink()
    with pytest.raises(ValueError, match="approved model files"):
        security.validate_assets(model_assets)


@pytest.fixture
def live_operator():
    configured = os.environ.get("AR03_LIVE_SAFETY_ROOT")
    if not configured:
        pytest.skip("explicit AR03_LIVE_SAFETY_ROOT required; never arms a campaign implicitly")
    root = Path(configured).resolve()
    config = json.loads((root / "config.json").read_text())
    admin = {"Authorization": "Bearer " + (root / "private/admin-token").read_text().strip()}
    research = {"Authorization": "Bearer " + (root / "research-token").read_text().strip()}
    with httpx.Client(
        base_url=f"http://127.0.0.1:{config['port']}", trust_env=False, timeout=60
    ) as client:
        status = client.get("/status", headers=research).json()
        assert not status["authorized"] and status["proposed"] == 0 and not status["running"]
        if not status["verified"]:
            client.post("/admin/verify", headers=admin, json={}).raise_for_status()
        try:
            yield root, client, admin, research
        finally:
            client.post("/admin/pause", headers=admin, json={}).raise_for_status()


def test_live_valid_permission_is_automatically_revoked(live_operator):
    root, client, admin, research = live_operator
    response = client.post(
        "/admin/authorize",
        headers=admin,
        json={
            "source_ref": "Owner-authorized revocation test; no orchestrator or candidates"
        },
    )
    response.raise_for_status()
    assert response.json()["authorized"] is True
    assert client.get("/status", headers=research).json()["authorized"] is True
    evidence = json.loads((root / "private/security/evidence.json").read_text())
    permission_path = root / "private/security/permission.json"
    permission = json.loads(permission_path.read_text())
    assert permission["fingerprint"] == evidence["fingerprint"]
    assert permission["generation"] == evidence["generation"]
    config_path = root / "config.json"
    original = config_path.read_bytes()
    try:
        config_path.write_bytes(original + b" ")
        deadline = time.monotonic() + 15
        while permission_path.exists():
            assert time.monotonic() < deadline, "valid permission was not automatically revoked"
            time.sleep(0.2)
        assert not (root / "private/security/evidence.json").exists()
        assert client.post("/admin/verify", headers=admin, json={}).status_code == 423
    finally:
        config_path.write_bytes(original)
    assert client.post("/admin/verify", headers=admin, json={}).status_code == 423
    status = client.get("/status", headers=research).json()
    assert not status["authorized"] and status["proposed"] == 0 and not status["running"]
    (root / "private/security/review-valid-permission.json").write_text(
        json.dumps(
            {
                "initial_permission_valid": True,
                "automatic_revocation": True,
                "stale_process_reverification_blocked": True,
                "orchestrator_started": False,
                "candidates_started": False,
                "proposed": status["proposed"],
            },
            indent=2,
        )
        + "\n"
    )
