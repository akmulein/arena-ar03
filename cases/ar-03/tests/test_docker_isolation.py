"""Only the new Docker execution boundary; synthetic inputs, no dataset/metric."""

import importlib.util
import json
import os
import shutil
import textwrap
from pathlib import Path

import pytest
from evaluator.runner import DockerRunner

CASE = Path(__file__).resolve().parents[1]
IMAGE = "arena-ar03-runner:arm64-cpu-v1"
ARENA = CASE.parents[1]
ASSETS = (
    Path(os.environ.get("AR03_OPERATOR_HOME", str(ARENA.parent / f"{ARENA.name}-operator")))
    / "runner-assets"
)


@pytest.fixture
def docker_runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ar03_docker_train", CASE / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "DOCKER_CONFIG",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    ):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    module.configure_local_docker(tmp_path)
    return DockerRunner(
        image=IMAGE,
        uses_model=False,
        assets_dir=ASSETS,
        memory="6g",
        cpus="2",
        tmpfs_size="512m",
        timeout=300,
    )


def test_candidate_cannot_access_host_closed_files_env_or_network(
    docker_runner, tmp_path, monkeypatch
):
    host_files = []
    for relative in (
        "host-secret",
        "closed/public/truth.json",
        "closed/private/result.json",
        "closed/artifacts/visualization.html",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic-host-canary")
        host_files.append(str(path))
    monkeypatch.setenv("AR03_HOST_SECRET_CANARY", "synthetic-host-only-value")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source = """
        import json, os, socket, sys
        from pathlib import Path
        task = json.load(sys.stdin)
        def writable(path):
            try:
                Path(path).write_text("probe")
                return True
            except OSError:
                return False
        def reachable(host):
            try:
                with socket.create_connection((host, 443), timeout=.2):
                    return True
            except OSError:
                return False
        status = Path("/proc/self/status").read_text()
        result = {
            "host_files_hidden": all(not Path(p).exists() for p in HOST_FILES),
            "host_env_hidden": "AR03_HOST_SECRET_CANARY" not in os.environ,
            "proxy_env_absent": not any(k.lower() in ("http_proxy", "https_proxy", "all_proxy")
                                        for k in os.environ),
            "socket_hidden": not Path("/var/run/docker.sock").exists(),
            "root_readonly": not writable("/host-write-probe"),
            "candidate_readonly": not writable("/candidate/write-probe"),
            "assets_readonly": not writable("/assets/write-probe"),
            "tmp_writable": writable("/tmp/write-probe"),
            "network_blocked": not any(reachable(h) for h in
                                       ("1.1.1.1", "192.168.65.254", "host.docker.internal")),
            "unprivileged": os.getuid() == 10001,
            "no_capabilities": "CapEff:\\t0000000000000000" in status,
            "no_new_privileges": "NoNewPrivs:\\t1" in status,
            "only_current_input": set(task) == {"input", "assets_dir"}
                                  and task["input"] == "synthetic-current-input",
        }
        print(json.dumps({"answer": json.dumps(result)}))
    """
    (candidate / "solve.py").write_text(
        "HOST_FILES = " + repr(host_files) + "\n" + textwrap.dedent(source)
    )
    result = docker_runner(candidate, "synthetic-current-input")
    checks = json.loads(result.answer)
    assert all(checks.values()), checks
    assert result.usage["total_tokens"] == 0


def test_arm64_paddle_and_local_weights_execute_offline(docker_runner, tmp_path):
    candidate = tmp_path / "compatibility"
    candidate.mkdir()
    shutil.copytree(CASE / "ways/baseline/runtime", candidate / "runtime")
    shutil.copyfile(CASE / "ways/baseline/solve.py", candidate / "adapter.py")
    script = textwrap.dedent("""
        import json, sys
        from pathlib import Path
        import adapter
        task = json.load(sys.stdin)
        with adapter.diagnostics_to_stderr():
            import cv2, numpy as np, paddle, platform
            sys.path.insert(0, str(Path(__file__).parent / "runtime"))
            from ar03_solver.baseline.detection import table_detector
            from ar03_solver.baseline.structure import structure_model
            from ar03_solver.baseline.ocr import ocr_model
            image = np.full((128, 256, 3), 255, dtype=np.uint8)
            for y in (8, 64, 120): cv2.line(image, (8, y), (248, y), (0, 0, 0), 2)
            for x in (8, 128, 248): cv2.line(image, (x, 8), (x, 120), (0, 0, 0), 2)
            cv2.putText(image, "12", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
            detector = table_detector().predict(image, imgsz=1024, conf=.25,
                                                device="cpu", verbose=False)
            structure = list(structure_model().predict(image, batch_size=1))
            text = list(ocr_model().predict([image[12:60, 12:124].copy()], batch_size=1))
            result = {"paddle": paddle.__version__, "arch": platform.machine(),
                      "detector": len(detector), "structure": len(structure),
                      "ocr": len(text), "text_is_string": isinstance(text[0]["rec_text"], str)}
        print(json.dumps({"answer": json.dumps(result)}))
    """)
    (candidate / "solve.py").write_text(
        "try:\n"
        + textwrap.indent(script, "    ")
        + "except Exception as error:\n"
        + "    print(json.dumps({'answer': json.dumps({'runtime_error': "
        "type(error).__name__, 'message': str(error)})}))\n"
    )
    result = json.loads(docker_runner(candidate, "synthetic-compatibility-input").answer)
    assert result == {
        "paddle": "3.2.2",
        "arch": "aarch64",
        "detector": 1,
        "structure": 1,
        "ocr": 1,
        "text_is_string": True,
    }
