import argparse
import importlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path


def container_command(
    image: str, network: str, model: str, proxy: str | None = None,
) -> list[str]:
    command = [
        "docker", "run", "--rm", "--network", network, "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
    ]
    if proxy:
        for name in ("DDGS_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
            command.extend(["-e", f"{name}={proxy}"])
    return [
        *command, "--entrypoint", "python", image,
        "-m", "integrations.hermes.network", "--model", model, "--probe",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the researcher's public web search")
    parser.add_argument("--model", default=os.environ.get("ARENA_MODEL"))
    parser.add_argument("--image")
    parser.add_argument("--network")
    parser.add_argument("--proxy")
    parser.add_argument("--home", type=Path, default=Path("/tmp/arena-web-check"))
    parser.add_argument("--probe", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.model:
        parser.error("--model or ARENA_MODEL is required")
    if arguments.probe:
        if arguments.image or arguments.network or arguments.proxy:
            parser.error("--probe cannot configure a Docker environment")
        from integrations.hermes.runtime import prepare_home

        prepare_home(arguments.home, model=arguments.model)
        web_search_tool = importlib.import_module("tools.web_tools").web_search_tool
        result = json.loads(web_search_tool("Python documentation"))
        hits = result.get("data", {}).get("web", [])
        success = result.get("success") is True and isinstance(hits, list) and bool(hits)
        print(json.dumps({
            "success": success, "results": len(hits) if isinstance(hits, list) else 0,
            "error": result.get("error"),
        }))
        return 0 if success else 1
    if not arguments.image or not arguments.network:
        parser.error("--image and --network are required")
    result = subprocess.run(container_command(
        arguments.image, arguments.network, arguments.model, arguments.proxy,
    ), capture_output=True, text=True)
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    return int(result.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
