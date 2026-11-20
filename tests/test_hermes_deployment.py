import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES = Path("integrations/hermes")


def test_generic_image_uses_common_entrypoint_and_no_case_sources():
    dockerfile = (ROOT / HERMES / "Dockerfile").read_text()
    assert "cases/" not in dockerfile
    assert "integrations/hermes/config.yaml /opt/arena/integrations/hermes/" in dockerfile
    assert "COPY integrations/hermes/source-cache/ /opt/hermes-source-cache/" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "integrations.hermes.researcher"]' in dockerfile
    assert "uv sync --frozen --no-dev --extra tracking" in dockerfile


def test_generic_hermes_build_context_excludes_all_cases_and_private_materials(tmp_path):
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker BuildKit is required to inspect the build context")
    for arguments in (("buildx", "version"), ("info", "--format", "{{.ServerVersion}}")):
        if subprocess.run([docker, *arguments], capture_output=True).returncode:
            pytest.skip("Docker BuildKit is required to inspect the build context")
    included = {
        "pyproject.toml", "uv.lock", "evaluator/source.py", "runner/main.py",
        "integrations/hermes/runtime.py", "integrations/hermes/config.yaml",
        "integrations/hermes/source-cache/source.tar", "integrations/hermes/install.sh",
    }
    excluded = {
        "cases/template/README.md", "cases/sc71/harness/hermes/runtime.py",
        "cases/custom/dataset/train.jsonl", "integrations/docker/monitor.py",
        "integrations/hermes/.env", "integrations/hermes/access-token.txt",
        "integrations/hermes/source-cache/operator.env",
        "integrations/hermes/source-cache/private/data.json",
        "integrations/hermes/source-cache/.git/config", "archive/report.md",
    }
    context = tmp_path / "context"
    for name in included | excluded:
        path = context / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic")
    dockerfile = context / HERMES / "Dockerfile"
    dockerfile.write_text("FROM scratch\nCOPY . /\n")
    shutil.copyfile(ROOT / HERMES / "Dockerfile.dockerignore", dockerfile.with_suffix(
        ".dockerignore"
    ))
    output = tmp_path / "output"
    result = subprocess.run([
        docker, "buildx", "build", "--progress=quiet", "--file", str(dockerfile),
        "--output", f"type=local,dest={output}", str(context),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert all((output / name).is_file() for name in included)
    assert not any((output / name).exists() for name in excluded)
