"""Train service using Arena's isolated, CPU-only Docker candidate runner."""

import json
import os
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parent
ARENA = CASE.parents[1]
sys.path.insert(0, str(ARENA))


DATASET_ID = "ar03-train-500"
CONDITIONS_ID = "ar03-teds-v1-docker-arm64-cpu-train-only"


def configure_local_docker(runtime):
    """Do not inject host proxy settings or choose a remote daemon implicitly."""
    sockets = (Path("/var/run/docker.sock"), Path.home() / ".docker/run/docker.sock")
    daemon = next((path for path in sockets if path.exists() and path.is_socket()), None)
    if daemon is None:
        raise ValueError("a local Docker daemon socket is required; no native fallback")
    docker_config = runtime / "docker-config"
    docker_config.mkdir(mode=0o700, parents=True, exist_ok=True)
    plugins = Path("/Applications/Docker.app/Contents/Resources/cli-plugins")
    options = {"cliPluginsExtraDirs": [str(plugins)]} if plugins.is_dir() else {}
    (docker_config / "config.json").write_text(json.dumps(options) + "\n")
    for name in ("DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        os.environ.pop(name, None)
    os.environ.update(DOCKER_CONFIG=str(docker_config), DOCKER_HOST=f"unix://{daemon}")


def main():
    raise SystemExit(
        "Standalone train launch is disabled. Use cases/ar-03/deploy/operatorctl.py run."
    )


if __name__ == "__main__":
    main()
