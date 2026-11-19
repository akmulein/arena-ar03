
import cv2

from solution.detection import detect
from solution.geometry import transform_box
from solution.ocr import recognize
from solution.preprocessing import preprocess
from solution.schema import validate
from solution.structure import recover


def solve(image, *, detections=None):
    """Recognize tables in a document image."""

    cv2.setNumThreads(1)

    deskewed, inverse = preprocess(image)
    boxes = detect(deskewed)

    tables = []
    height, width = image.shape[:2]

    # Collect raw detections independently of structure recognition.
    if detections is not None:
        detections.extend(
            {"bbox": transform_box(box, inverse, width, height)}
            for box in boxes
        )

    for box in boxes:
        table = recover(box, deskewed)

        if table is None:
            continue

        # Validate the logical grid before OCR.
        validate({"tables": [table]})

        recognize(deskewed, table)

        # Restore original table coordinates.
        table["bbox"] = transform_box(
            table["bbox"],
            inverse,
            width,
            height,
        )

        # Restore original cell coordinates.
        for cell in table["cells"]:
            cell["bbox"] = transform_box(
                cell["bbox"],
                inverse,
                width,
                height,
            )

        tables.append(table)

    return validate({"tables": tables})
