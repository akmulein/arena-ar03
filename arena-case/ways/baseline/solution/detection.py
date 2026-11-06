import cv2
import numpy as np


def line_masks(binary):
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8)
    )
    horizontal = cv2.morphologyEx(
        horizontal,
        cv2.MORPH_OPEN,
        np.ones((1, max(25, width // 25)), np.uint8),
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)
    )
    vertical = cv2.morphologyEx(
        vertical, cv2.MORPH_OPEN, np.ones((max(15, height // 50), 1), np.uint8)
    )
    return horizontal, vertical


def text_like(binary, horizontal, vertical, box):
    x0, y0, x1, y1 = box
    crop = binary[y0:y1, x0:x1]
    rules = (horizontal | vertical)[y0:y1, x0:x1]
    text = cv2.bitwise_and(crop, cv2.bitwise_not(rules))
    ink = cv2.countNonZero(text)
    if ink < max(12, 0.1 * cv2.countNonZero(rules)):
        return True
    if cv2.countNonZero(crop) / crop.size > 0.30:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(text)
    components = stats[1:]
    usable = components[
        (components[:, 3] >= 3)
        & (components[:, 3] <= 25)
        & (components[:, 2] <= 60)
        & (components[:, 4] >= 3)
    ]
    return len(usable) >= 8 and usable[:, 4].sum() / ink >= 0.55


def detect(binary):
    height, width = binary.shape
    horizontal, vertical = line_masks(binary)
    contours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= max(50, width // 10) and h <= 8:
            segments.append((x, y, x + w, y + h))
    groups = []
    for segment in sorted(segments, key=lambda b: b[1]):
        x0, y0, x1, y1 = segment
        for group in groups:
            left, _, right, bottom = group[-1]
            tolerance = max(8, 0.06 * (x1 - x0))
            if (
                abs(x0 - left) <= tolerance
                and abs(x1 - right) <= tolerance
                and y0 - bottom <= height * 0.4
            ):
                group.append(segment)
                break
        else:
            groups.append([segment])
    boxes = []
    connected = cv2.morphologyEx(
        horizontal | vertical, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(
        connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            w >= 60
            and h >= 20
            and cv2.countNonZero(vertical[y : y + h, x : x + w]) > h
            and cv2.countNonZero(horizontal[y : y + h, x : x + w]) > w
        ):
            box = (x, y, x + w, y + h)
            if text_like(binary, horizontal, vertical, box):
                boxes.append(box)
    for group in groups:
        if len(group) < 2:
            continue
        box = (
            min(b[0] for b in group),
            min(b[1] for b in group),
            max(b[2] for b in group),
            max(b[3] for b in group),
        )
        if box[3] - box[1] >= 20 and text_like(
            binary, horizontal, vertical, box
        ):
            boxes.append(box)
    kept = []
    for box in sorted(
        boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
    ):
        area = (box[2] - box[0]) * (box[3] - box[1])
        if any(
            max(0, min(box[2], b[2]) - max(box[0], b[0]))
            * max(0, min(box[3], b[3]) - max(box[1], b[1]))
            > 0.8 * area
            for b in kept
        ):
            continue
        kept.append(box)
    return sorted(kept, key=lambda b: (b[1], b[0])), horizontal, vertical
