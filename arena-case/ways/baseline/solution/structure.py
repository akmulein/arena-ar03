import cv2
import numpy as np


def positions(profile, threshold):
    indices = np.flatnonzero(profile > threshold)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1)
    return [int(round(group.mean())) for group in groups]


def text_separators(binary, horizontal, vertical):
    text = binary & ~(horizontal | vertical)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(text)
    keep = np.zeros(count, np.uint8)
    for index, (_, _, width, height, area) in enumerate(stats[1:], 1):
        if 2 <= height <= 30 and width <= 80 and area >= 3:
            keep[index] = 1
    text = keep[labels]
    heights = stats[1:][keep[1:] > 0, 3]
    char_height = float(np.median(heights)) if len(heights) else 6
    occupied = np.flatnonzero(text.sum(axis=1) >= 2)
    if not len(occupied):
        return [], []
    bands = np.split(occupied, np.flatnonzero(np.diff(occupied) > 2) + 1)
    ys = (
        [0]
        + [int((a[-1] + b[0]) / 2) for a, b in zip(bands, bands[1:])]
        + [text.shape[0] - 1]
    )
    # A spanning header must not fill the gutters of the data columns.
    body_start = bands[min(1, len(bands) - 1)][0]
    body = text[body_start:]
    whitespace = np.flatnonzero(body.sum(axis=0) <= max(1, len(body) * 0.02))
    gaps = np.split(whitespace, np.flatnonzero(np.diff(whitespace) > 1) + 1)
    xs = [0]
    for gap in gaps:
        if (
            len(gap) >= max(4, char_height * 0.6)
            and gap[0] > char_height
            and gap[-1] < text.shape[1] - char_height
        ):
            xs.append(int((gap[0] + gap[-1]) / 2))
    xs.append(text.shape[1] - 1)
    return xs, ys


def recover(box, horizontal, vertical, binary=None):
    x0, y0, x1, y1 = box
    h = horizontal[y0:y1, x0:x1] > 0
    v = vertical[y0:y1, x0:x1] > 0
    ys = positions(h.mean(axis=1), 0.3)
    xs = positions(v.mean(axis=0), 0.3)
    ruled_grid = len(xs) >= 3 and len(ys) >= 2
    if not ruled_grid and binary is not None:
        inferred_xs, inferred_ys = text_separators(
            binary[y0:y1, x0:x1],
            horizontal[y0:y1, x0:x1],
            vertical[y0:y1, x0:x1],
        )
        if len(inferred_xs) >= 3 and len(inferred_ys) >= 3:
            xs, ys = inferred_xs, inferred_ys
        else:
            return None
    if len(xs) < 2 or len(ys) < 2:
        return None
    rows, cols = len(ys) - 1, len(xs) - 1
    groups = [{(r, c)} for r in range(rows) for c in range(cols)]

    def merge(a, b):
        ga = next(g for g in groups if a in g)
        gb = next(g for g in groups if b in g)
        if ga is gb:
            return
        ga.update(gb)
        groups.remove(gb)

    for r in range(rows):
        for c in range(cols):
            if ruled_grid and c + 1 < cols:
                boundary = v[
                    ys[r] + 2 : ys[r + 1] - 1,
                    max(0, xs[c + 1] - 2) : xs[c + 1] + 3,
                ]
                if boundary.size and boundary.any(axis=1).mean() < 0.35:
                    merge((r, c), (r, c + 1))
            if ruled_grid and r + 1 < rows:
                boundary = h[
                    max(0, ys[r + 1] - 2) : ys[r + 1] + 3,
                    xs[c] + 2 : xs[c + 1] - 1,
                ]
                if boundary.size and boundary.any(axis=0).mean() < 0.35:
                    merge((r, c), (r + 1, c))
    rectangles = []
    for group in groups:
        rr, cc = zip(*group)
        if len(group) == (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1):
            rectangles.append(group)
        else:
            rectangles.extend({position} for position in sorted(group))
    cells = []
    for group in rectangles:
        rr, cc = zip(*group)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        cells.append(
            dict(
                r0=r0,
                r1=r1,
                c0=c0,
                c1=c1,
                text="",
                bbox=[
                    x0 + xs[c0],
                    y0 + ys[r0],
                    x0 + xs[c1 + 1],
                    y0 + ys[r1 + 1],
                ],
            )
        )
    return dict(
        bbox=list(box),
        rows=rows,
        cols=cols,
        cells=sorted(cells, key=lambda c: (c["r0"], c["c0"])),
    )
