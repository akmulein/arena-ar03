"""Arena program adapter. The recognition algorithm lives in ar03_solver."""

import base64
import binascii
import contextlib
import ctypes
import json
import os
import sys
import tempfile
from pathlib import Path

MAX_INPUT_CHARS = 500_000
MAX_OUTPUT_BYTES = 1024 * 1024
RUNTIME = Path(__file__).resolve().parent / "runtime"


def decode_input(value: str) -> bytes:
    if not isinstance(value, str) or len(value) > MAX_INPUT_CHARS:
        raise ValueError("invalid AR03 input length")
    envelope = json.loads(value)
    if not isinstance(envelope, dict) or envelope.get("version") != "ar03-page-v1":
        raise ValueError("unsupported AR03 input version")
    try:
        encoded = base64.b64decode(envelope["image_base64"], validate=True)
    except (KeyError, ValueError, TypeError, binascii.Error) as error:
        raise ValueError("invalid encoded image") from error
    if not encoded:
        raise ValueError("empty encoded image")
    return encoded


def solve_request(task: dict, *, solver=None) -> dict:
    encoded = decode_input(task["input"])
    if task.get("parameters", {}):
        raise ValueError("baseline has no candidate-controlled parameters")
    device = os.environ.get("AR03_DETECTOR_DEVICE", "cpu")
    if device not in ("cpu", "mps"):
        raise ValueError("unsupported detector device")
    import cv2

    if solver is None:
        sys.path.insert(0, str(RUNTIME))
        from ar03_solver.baseline.pipeline import solve

        solver = solve
    with tempfile.TemporaryDirectory(prefix="ar03-page-") as directory:
        image_path = Path(directory) / "page.jpg"
        image_path.write_bytes(encoded)
        image = cv2.imread(str(image_path))
    if image is None:
        raise OSError("cannot decode input image")
    detections = []
    try:
        prediction = solver(image, detections=detections)
        error = None
    except (RuntimeError, ValueError) as failure:
        prediction = {"tables": []}
        error = str(failure)
    answer = json.dumps({"prediction": prediction, "detections": detections, "error": error},
                        ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    result = {"answer": answer}
    # Match the stock runner's final ASCII-escaped envelope, not only the inner JSON.
    if len(json.dumps(result).encode()) + 1 > MAX_OUTPUT_BYTES:
        raise ValueError("answer exceeds Arena stdout limit")
    return result


@contextlib.contextmanager
def diagnostics_to_stderr():
    """Protect JSON stdout from both Python and native model-library messages."""
    sys.stdout.flush()
    saved = os.dup(1)
    library = ctypes.CDLL(None)
    library.fflush.argtypes = [ctypes.c_void_p]
    library.fflush.restype = ctypes.c_int
    try:
        os.dup2(2, 1)
        yield
    finally:
        try:
            sys.stdout.flush()
            library.fflush(None)
        finally:
            os.dup2(saved, 1)
            os.close(saved)


def main() -> None:
    raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("task exceeds Arena stdin limit")
    task = json.loads(raw)
    with diagnostics_to_stderr():
        result = solve_request(task)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
