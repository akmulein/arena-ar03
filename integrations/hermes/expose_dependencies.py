import site
import subprocess
from pathlib import Path


def main() -> None:
    dependency_path = subprocess.check_output([
        "/opt/arena/.venv/bin/python", "-c", "import site; print(site.getsitepackages()[0])",
    ], text=True)
    Path(site.getsitepackages()[0], "arena.pth").write_text(dependency_path, encoding="utf-8")


if __name__ == "__main__":
    main()
