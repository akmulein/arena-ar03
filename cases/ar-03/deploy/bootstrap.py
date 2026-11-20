"""Operator-only resource preparation; never starts services or a campaign."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import httpx

CASE = Path(__file__).resolve().parents[1]
ARENA = CASE.parents[1]
MODELS = (
    (
        "DocLayout-YOLO",
        "juliozhao/DocLayout-YOLO-DocStructBench",
        "8c3299a30b8ff29a1503c4431b035b93220f7b11",
    ),
    ("SLANet_plus", "PaddlePaddle/SLANet_plus", "bae6e5f8c3c4e7da0c0b7639fdf3228fe76184e2"),
    (
        "PP-OCRv6_small_rec",
        "PaddlePaddle/PP-OCRv6_small_rec",
        "bd619643acac4b9650c040234da8d944476ee3f1",
    ),
)


def external_resources(path):
    path = Path(path).resolve()
    if path.is_relative_to(ARENA) or ARENA.is_relative_to(path):
        raise ValueError("operator resources must be outside the source repository")
    return path


def exporter():
    spec = importlib.util.spec_from_file_location("ar03_data_export", CASE / "data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_data(resources):
    """Reuse original preparation and selection; reject a different frozen selection."""
    os.environ["AR03_RESOURCES"] = str(resources)
    sys.path[:0] = [str(CASE), str(CASE / "ways/baseline/runtime")]
    import cv2

    from dataset import annotations, download, prepare, storage, subsets

    cv2.setNumThreads(1)
    root = resources / "data"
    root.mkdir(exist_ok=True)
    for key in download.ARCHIVES:
        download.download_archive(key, root)
    if not prepare.dataset_is_complete(root, 42):
        with closing(annotations.index_pdf(root)) as database:
            for split in subsets.SIZES:
                prepare.prepare(split, database, 42, root)
    recipe = json.loads((CASE / "dataset/selection.json").read_text())
    directory = root / "subsets"
    for split, size in subsets.SIZES.items():
        frozen = recipe[split]
        if storage.sha256_file(root / split / "manifest.jsonl") != frozen["source_manifest_sha256"]:
            raise ValueError("prepared source population differs from the frozen selection")
        manifest = directory / f"{split}-{size}.jsonl"
        if not manifest.exists():
            subsets.write_subset(root, split, size, 42, directory, frozen["max_pages_per_document"])
        if storage.sha256_file(manifest) != frozen["manifest_sha256"]:
            raise ValueError("restored subset differs from the frozen selection")
    audit = directory / "arena-subsets-audit.json"
    if not audit.exists():
        shutil.copyfile(CASE / "dataset/audit-record.json", audit)
    data_export = exporter()
    for split, size in subsets.SIZES.items():
        data_export.existing_audit(audit, directory / f"{split}-{size}.jsonl", split)
    exported = root / "arena"
    exported.mkdir(exist_ok=True)
    if not (exported / "train-500.jsonl").exists():
        with tempfile.TemporaryDirectory(dir=resources) as temporary:
            destination = Path(temporary) / "train-500.jsonl"
            data_export.export_dataset(
                root=root,
                manifest=directory / "train-500.jsonl",
                split="train",
                output=destination,
                audit=audit,
            )
            destination.replace(exported / "train-500.jsonl")
    identifiers = CASE / "dataset/pilot-train-ids.json"
    target = exported / "evaluation-ids.json"
    if target.exists() and json.loads(target.read_text()) != json.loads(identifiers.read_text()):
        raise ValueError("existing pilot selection differs; refusing to replace it")
    if not target.exists():
        shutil.copyfile(identifiers, target)


def download_file(url, destination):
    if destination.is_symlink():
        raise ValueError("model source must not be a symlink")
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("xb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_models(resources):
    from security import APPROVED_MODEL_FILES, validate_assets

    assets = resources / "runner-assets"
    if assets.exists():
        validate_assets(assets)
        return
    for directory, repository, revision in MODELS:
        prefix = "models/" + directory + "/"
        for name in sorted(APPROVED_MODEL_FILES):
            if name.startswith(prefix):
                download_file(
                    f"https://huggingface.co/{repository}/resolve/{revision}/{name[len(prefix) :]}",
                    resources / name,
                )
    with tempfile.TemporaryDirectory(prefix=".runner-assets-", dir=resources) as temporary:
        staged = Path(temporary) / "assets"
        for name in sorted(APPROVED_MODEL_FILES):
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resources / name, target)
        validate_assets(staged)
        staged.rename(assets)


def inspect_image(reference):
    result = subprocess.run(
        ["docker", "image", "inspect", reference], capture_output=True, text=True, check=False
    )
    if result.returncode:
        return None
    record = json.loads(result.stdout)[0]
    if record["Os"] != "linux" or record["Architecture"] != "arm64":
        raise ValueError("the fixed AR-03 image configuration requires Linux ARM64")
    return record["Id"]


def prepare_research_python(compose, environment, resources):
    """Install only NumPy into the existing writable researcher volume, offline."""
    execute = compose + ["exec", "-T", "--user", "10001:10001", "researcher"]
    probe = (
        "import sys; sys.path.insert(0,'/codex/research-packages'); "
        "import numpy; assert numpy.__version__=='2.3.5'"
    )
    present = subprocess.run(
        execute + ["/opt/arena/.venv/bin/python", "-I", "-c", probe],
        env=environment,
        capture_output=True,
        check=False,
    )
    if present.returncode == 0:
        return
    metadata = httpx.get("https://pypi.org/pypi/numpy/2.3.5/json", timeout=30)
    metadata.raise_for_status()
    wheel = next(
        row
        for row in metadata.json()["urls"]
        if "-cp313-cp313-manylinux" in row["filename"] and row["filename"].endswith("aarch64.whl")
    )
    cached = resources / "research-wheels" / wheel["filename"]
    download_file(wheel["url"], cached)
    writer = (
        "import sys; from pathlib import Path; "
        "p=Path('/codex/research-wheels')/sys.argv[1]; "
        "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())"
    )
    with cached.open("rb") as stream:
        subprocess.run(
            execute + ["python", "-I", "-c", writer, wheel["filename"]],
            stdin=stream,
            env=environment,
            check=True,
        )
    subprocess.run(
        execute
        + [
            "/usr/local/bin/python",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-cache-dir",
            "--no-compile",
            "--target",
            "/codex/research-packages",
            "/codex/research-wheels/" + wheel["filename"],
        ],
        env=environment,
        check=True,
    )


def prepare_images(resources):
    from security import PINS

    sys.path.insert(0, str(CASE))
    from train import configure_local_docker

    configure_local_docker(resources / "build")
    path = resources / "images.json"
    images = json.loads(path.read_text()) if path.exists() else dict(PINS)
    for name, dockerfile in (
        ("researcher", "deploy/container.Dockerfile"),
        ("runner", "harness/Dockerfile"),
    ):
        installed = inspect_image(images[name])
        if installed is None:
            tag = f"arena-ar03-{name}:arm64-reproduced"
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--platform",
                    "linux/arm64",
                    "-t",
                    tag,
                    "-f",
                    str(CASE / dockerfile),
                    str(ARENA),
                ],
                check=True,
            )
            installed = inspect_image(tag)
        if installed is None:
            raise ValueError("built image is unavailable")
        images[name] = installed
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(images, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def bootstrap(args):
    resources = external_resources(args.resources)
    resources.mkdir(parents=True, exist_ok=True, mode=0o700)
    steps = {"data": prepare_data, "models": prepare_models, "images": prepare_images}
    selected = list(steps) if args.step == "all" else [args.step]
    for step in selected:
        print(f"Preparing operator resources: {step}", file=sys.stderr, flush=True)
        steps[step](resources)
    print(json.dumps({"prepared_steps": selected, "campaign_started": False}))
