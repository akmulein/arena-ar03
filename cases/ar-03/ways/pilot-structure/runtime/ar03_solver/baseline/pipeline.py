import cv2

from ar03_solver.baseline.detection import detect
from ar03_solver.baseline.ocr import recognize
from ar03_solver.baseline.preprocessing import preprocess
from ar03_solver.baseline.structure import recover
from ar03_solver.common.geometry import transform_box
from ar03_solver.common.schema import validate


def solve(image, *, detections=None):
    """Recognize tables in a document image."""

    cv2.setNumThreads(1)

    deskewed, inverse = preprocess(image)
    boxes = detect(deskewed)

    tables = []
    height, width = image.shape[:2]

    if detections is not None:
        detections.extend(
            {"bbox": transform_box(box, inverse, width, height)}
            for box in boxes
        )

    for box in boxes:
        table = recover(box, deskewed)

        if table is None:
            continue

        recognize(deskewed, table)

        table["bbox"] = transform_box(
            table["bbox"],
            inverse,
            width,
            height,
        )

        for cell in table["cells"]:
            cell["bbox"] = transform_box(
                cell["bbox"],
                inverse,
                width,
                height,
            )

        tables.append(table)

    return validate({"tables": tables})