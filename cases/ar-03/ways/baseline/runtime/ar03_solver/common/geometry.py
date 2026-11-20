import numpy as np


def transform_box(box, matrix, width, height):
    x0, y0, x1, y1 = box
    points = (
        np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]])
        @ matrix.T
    )
    return [
        float(np.clip(points[:, 0].min(), 0, width)),
        float(np.clip(points[:, 1].min(), 0, height)),
        float(np.clip(points[:, 0].max(), 0, width)),
        float(np.clip(points[:, 1].max(), 0, height)),
    ]
