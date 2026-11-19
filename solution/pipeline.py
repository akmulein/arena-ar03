import cv2

from solution.detection import detect
from solution.geometry import transform_box
from solution.ocr import recognize
from solution.preprocessing import preprocess
from solution.schema import validate
from solution.structure import recover


def solve(image, *, detections=None):
    """Recognize tables; optionally collect raw boxes in input coordinates."""

    cv2.setNumThreads(1)

    # Preprocessing: deskew without binarization
    deskewed, inverse = preprocess(image)

    # Table detection: DocLayout-YOLO
    boxes = detect(deskewed)

    # Grayscale image for OCR
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    tables = []
    height, width = image.shape[:2]

    # Raw detections for benchmark
    if detections is not None:
        detections.extend(
            {"bbox": transform_box(box, inverse, width, height)}
            for box in boxes
        )

    for box in boxes:
        # Table structure recognition: SLANet_plus
        table = recover(box, gray=deskewed)

        if table is None:
            continue

        # Cell text recognition
        recognize(gray, table)

        # Restore table coordinates
        table["bbox"] = transform_box(
            table["bbox"],
            inverse,
            width,
            height,
        )

        # Restore cell coordinates
        for cell in table["cells"]:
            cell["bbox"] = transform_box(
                cell["bbox"],
                inverse,
                width,
                height,
            )

        tables.append(table)

    return validate({"tables": tables})