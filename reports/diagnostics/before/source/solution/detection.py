import cv2


def detect(binary):
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 30), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 50))),
    )
    mask = cv2.morphologyEx(
        horizontal | vertical,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            w >= 60
            and h >= 30
            and cv2.countNonZero(horizontal[y : y + h, x : x + w]) > w
        ):
            boxes.append((x, y, x + w, y + h))
    return sorted(boxes, key=lambda b: (b[1], b[0])), horizontal, vertical
