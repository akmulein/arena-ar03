import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HERMES = Path("cases/sc71/harness/hermes")


def test_hermes_researcher_can_install_the_tracking_client() -> None:
    dockerfile = (ROOT / HERMES / "Dockerfile").read_text()
    arena = dockerfile.split("WORKDIR /opt/arena\n", 1)[1]
    assert "RUN uv sync --frozen --no-dev --extra tracking\n" in arena


@pytest.mark.parametrize("hermes", [False, True])
def test_build_context_excludes_other_cases_and_sensitive_files(
    tmp_path: Path, hermes: bool,
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker BuildKit is required to inspect the build context")
    for arguments in (("buildx", "version"), ("info", "--format", "{{.ServerVersion}}")):
        available = subprocess.run(
            [docker, *arguments], capture_output=True, check=False, timeout=30,
        )
        if available.returncode:
            pytest.skip("Docker BuildKit is required to inspect the build context")

    context = tmp_path / "context"
    context.mkdir()
    common = {
        "pyproject.toml", "uv.lock", "evaluator/source.py", "runner/main.py",
        "integrations/hermes/runtime.py", "integrations/hermes/config.yaml",
        "integrations/hermes/install.sh",
    }
    template = {"cases/template/README.md"}
    adapter = {str(HERMES / path) for path in (
        "main.py", "source-cache/source.tar",
    )}
    excluded = {
        "cases/other/harness/main.py", "cases/sc71/dataset/train.jsonl",
        "cases/sc71/campaigns/run/state.json", "cases/sc71/proposals/method.py",
        "archive/cases/sc71/result.json", "evaluator/.env", "evaluator/access-token.txt",
        "integrations/hermes/source-cache/source.tar",
        str(HERMES / "config.yaml"),
        *(str(HERMES / path) for path in (
            ".env", "secret.json", "private/data.json", "source-cache/.env",
            "source-cache/operator.env", "source-cache/private/value.txt",
            "source-cache/.git/config", "source-cache/data/value.json",
        )),
    }
    for name in common | template | adapter | excluded:
        path = context / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic")
    shutil.copyfile(ROOT / ".dockerignore", context / ".dockerignore")
    dockerfile = HERMES / "Dockerfile" if hermes else Path("Dockerfile")
    if hermes:
        shutil.copyfile(
            ROOT / HERMES / "Dockerfile.dockerignore",
            context / HERMES / "Dockerfile.dockerignore",
        )
    (context / dockerfile).write_text("FROM scratch\nCOPY . /\n")
    output = tmp_path / "output"
    result = subprocess.run(
        [docker, "buildx", "build", "--progress=quiet", "--file", str(context / dockerfile),
         "--output", f"type=local,dest={output}", str(context)],
        text=True, capture_output=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    allowed = common | (adapter if hermes else template)
    denied = excluded | (template if hermes else adapter)
    for name in allowed:
        assert (output / name).is_file(), name
    for name in denied:
        assert not (output / name).exists(), name


@pytest.mark.parametrize("recipe,old_file", [
    ("cases/sc71/harness/hermes/Dockerfile", False),
    ("cases/sc35/harness/Dockerfile", True),
])
def test_case_images_preserve_legacy_config_path_using_shared_profile(tmp_path, recipe, old_file):
    instructions = [line.removeprefix("RUN ") for line in (ROOT / recipe).read_text().splitlines()
                    if line.startswith("RUN ln ") and "config.yaml" in line]
    assert len(instructions) == 1
    docker = shutil.which("docker")
    if docker is None or subprocess.run(
        [docker, "image", "inspect", "python:3.12-slim"], capture_output=True,
    ).returncode:
        pytest.skip("A local Python image and Docker are required for the image wiring check")
    arena = tmp_path / "arena"
    shared = arena / "integrations/hermes/config.yaml"
    shared.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "integrations/hermes/config.yaml", shared)
    adapter = arena / HERMES
    adapter.mkdir(parents=True)
    (adapter / "runtime.py").write_text("")
    if old_file:
        (adapter / "config.yaml").write_text("old case profile")
    probe = (
        "from pathlib import Path; "
        "runtime = Path('/opt/arena/cases/sc71/harness/hermes/runtime.py'); "
        "config = runtime.with_name('config.yaml'); "
        "shared = Path('/opt/arena/integrations/hermes/config.yaml'); "
        "assert config.is_symlink(); assert config.read_text() == shared.read_text()"
    )
    result = subprocess.run([
        docker, "run", "--rm", "--network", "none", "--entrypoint", "sh",
        "--mount", f"type=bind,source={arena},target=/opt/arena",
        "python:3.12-slim", "-c", instructions[0] + " && python -c " + shlex.quote(probe),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
