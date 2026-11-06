import cv2
import numpy as np


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1, 600 / max(gray.shape))
    thumbnail = cv2.resize(gray, None, fx=scale, fy=scale)
    binary = cv2.threshold(
        thumbnail, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 360,
        40,
        minLineLength=max(40, thumbnail.shape[1] // 10),
        maxLineGap=5,
    )
    angles = []
    if lines is not None:
        for x0, y0, x1, y1 in lines.reshape(-1, 4):
            angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
            if abs(angle) < 5:
                angles.append(angle)
    angle = float(np.median(angles)) if angles else 0.0
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    gray = cv2.warpAffine(gray, matrix, (width, height), borderValue=255)
    gray = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    return gray, binary, cv2.invertAffineTransform(matrix)
