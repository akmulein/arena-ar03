"""Arena transport adapter; recognition stays in solution.pipeline."""

import base64
import json
import sys

import cv2
import numpy as np
from solution.pipeline import solve


def main():
    task = json.load(sys.stdin)
    value = task["input"]
    if not isinstance(value, str) or len(value) > 500000:
        raise ValueError("input must be a string of at most 500000 characters")
    encoded = json.loads(value)["image_base64"]
    data = base64.b64decode(encoded, validate=True)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot decode input image")
    answer = json.dumps(solve(image), ensure_ascii=False, separators=(",", ":"))
    output = json.dumps({"answer": answer}, ensure_ascii=False)
    if len(output.encode("utf-8")) > 1024 * 1024:
        raise ValueError("answer exceeds Arena's 1 MiB limit")
    print(output)


if __name__ == "__main__":
    main()
