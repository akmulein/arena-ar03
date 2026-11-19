import cv2

from baseline.detection import detect
from baseline.ocr import recognize
from baseline.preprocessing import preprocess
from baseline.structure import recover
from common.geometry import transform_box
from common.schema import validate


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