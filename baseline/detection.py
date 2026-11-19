import os
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def table_detector():
    from doclayout_yolo import YOLOv10

    default_path = (
        Path(__file__).resolve().parents[1]
        / "models/DocLayout-YOLO"
        / "doclayout_yolo_docstructbench_imgsz1024.pt"
    )

    model_path = Path(
        os.environ.get(
            "AR03_TABLE_MODEL_PATH",
            str(default_path),
        )
    ).expanduser()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {model_path}"
        )

    return YOLOv10(str(model_path))


def detect(image):
    """Detect tables in deskewed BGR image."""

    result = table_detector().predict(
        image,
        imgsz=1024,
        conf=0.25,
        device="cpu",
        verbose=False,
    )[0]

    height, width = image.shape[:2]
    boxes = []

    for item in result.boxes:
        label = result.names[int(item.cls.item())]

        if label.casefold() != "table":
            continue

        x0, y0, x1, y1 = np.clip(
            item.xyxy[0].cpu().numpy(),
            0,
            [width, height, width, height],
        )

        box = (
            int(np.floor(x0)),
            int(np.floor(y0)),
            int(np.ceil(x1)),
            int(np.ceil(y1)),
        )

        if box[0] < box[2] and box[1] < box[3]:
            boxes.append(box)

    return sorted(boxes, key=lambda b: (b[1], b[0]))