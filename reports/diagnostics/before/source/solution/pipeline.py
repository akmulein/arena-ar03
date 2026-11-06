import cv2

from solution.detection import detect
from solution.geometry import transform_box
from solution.ocr import recognize
from solution.preprocessing import preprocess
from solution.schema import validate
from solution.structure import recover


def solve(image):
    cv2.setNumThreads(1)
    gray, binary, inverse = preprocess(image)
    boxes, horizontal, vertical = detect(binary)
    tables = []
    height, width = image.shape[:2]
    for box in boxes:
        table = recover(box, horizontal, vertical)
        if table is None:
            continue
        recognize(gray, table)
        table["bbox"] = transform_box(table["bbox"], inverse, width, height)
        for cell in table["cells"]:
            cell["bbox"] = transform_box(cell["bbox"], inverse, width, height)
        tables.append(table)
    return validate({"tables": tables})
