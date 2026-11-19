import cv2
import numpy as np


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 180)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        60,
        minLineLength=max(60, gray.shape[1] // 8),
        maxLineGap=10,
    )

    angles, weights = [], []

    if lines is not None:
        for x0, y0, x1, y1 in lines.reshape(-1, 4):
            angle = np.degrees(
                np.arctan2(y1 - y0, x1 - x0)
            )

            if abs(angle) < 5:
                angles.append(angle)
                weights.append(abs(x1 - x0))

    angle = 0.0

    if angles:
        order = np.argsort(angles)
        cumulative = np.cumsum(np.array(weights)[order])

        angle = float(
            np.array(angles)[order][
                np.searchsorted(cumulative, cumulative[-1] / 2)
            ]
        )

    height, width = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        angle,
        1.0,
    )

    deskewed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    inverse = cv2.invertAffineTransform(matrix)

    return deskewed, inverse