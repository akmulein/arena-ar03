"""Prediction cache and reproducible baseline identification."""

import hashlib
import json
import logging
import math
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile

from dataset.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MODEL_PATHS = {
    'AR03_TABLE_MODEL_PATH': (
        'models/DocLayout-YOLO/doclayout_yolo_docstructbench_imgsz1024.pt',
        False,
    ),
    'AR03_STRUCTURE_MODEL_DIR': ('models/SLANet_plus', True),
    'AR03_OCR_MODEL_DIR': ('models/PP-OCRv6_small_rec', True),
}
MODEL_EXTENSIONS = {
    '.pt', '.pdmodel', '.pdiparams', '.json', '.yaml', '.yml',
    '.txt', '.onnx', '.bin', '.safetensors', '.pdparams', '.pdopt',
}
RUNTIME_PACKAGES = (
    'doclayout-yolo', 'torch', 'torchvision', 'paddlex', 'paddlepaddle',
    'paddlepaddle-gpu', 'numpy', 'opencv-python', 'opencv-python-headless',
    'opencv-contrib-python', 'opencv-contrib-python-headless', 'Pillow',
)
REQUIRED_FIELDS = {'prediction', 'detections', 'seconds', 'error'}


def _file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _model_hashes(project_root: Path) -> dict:
    models = {}
    for variable, (default, is_directory) in MODEL_PATHS.items():
        path = Path(os.environ.get(variable, str(project_root / default))).expanduser()
        if is_directory:
            if not path.is_dir():
                raise FileNotFoundError(f'Model directory not found: {path}')
            files = sorted(
                file for file in path.rglob('*')
                if file.is_file()
                and not any(part.startswith('.') for part in file.relative_to(path).parts)
                and file.suffix.lower() in MODEL_EXTENSIONS
            )
            if not files:
                raise FileNotFoundError(f'No model files found: {path}')
            models[variable] = {
                file.relative_to(path).as_posix(): _file_hash(file) for file in files
            }
        else:
            if not path.is_file():
                raise FileNotFoundError(f'Model weights not found: {path}')
            models[variable] = _file_hash(path)
    return models


def baseline_fingerprint(project_root: Path = PROJECT_ROOT) -> str:
    """Hash recognition code, local model files and relevant runtime versions once."""
    sources = []
    for package in ('baseline', 'common'):
        files = sorted((project_root / package).rglob('*.py'))
        if not files:
            raise FileNotFoundError(f'No Python sources in {project_root / package}')
        sources.extend(files)
    sources.append(project_root / 'requirements.txt')

    payload = {
        'sources': {
            path.relative_to(project_root).as_posix(): _file_hash(path)
            for path in sources
        },
        'models': _model_hashes(project_root),
        'packages': _package_versions(),
        'python': sys.version,
        'platform': platform.system(),
        'machine': platform.machine(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_prediction(path: Path, fingerprint: str, image_sha256: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        run = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning('Ignoring invalid prediction cache: %s', path)
        return None
    if not isinstance(run, dict):
        return None
    if (
        run.get('fingerprint') != fingerprint
        or run.get('image_sha256') != image_sha256
        or not REQUIRED_FIELDS.issubset(run)
    ):
        return None
    if (
        not isinstance(run['prediction'], dict)
        or not isinstance(run['detections'], list)
        or not isinstance(run['seconds'], (int, float))
        or not math.isfinite(run['seconds'])
        or run['seconds'] < 0
        or not (run['error'] is None or isinstance(run['error'], str))
    ):
        return None
    return run


def save_prediction(path: Path, run: dict) -> None:
    """Write atomically so an interrupted run cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(run, stream, ensure_ascii=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
