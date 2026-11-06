import numpy as np


def positions(profile, threshold):
    indices = np.flatnonzero(profile > threshold)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1)
    return [int(round(group.mean())) for group in groups]


def recover(box, horizontal, vertical):
    x0, y0, x1, y1 = box
    h = horizontal[y0:y1, x0:x1] > 0
    v = vertical[y0:y1, x0:x1] > 0
    ys = positions(h.mean(axis=1), 0.3)
    xs = positions(v.mean(axis=0), 0.3)
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
            if c + 1 < cols:
                boundary = v[
                    ys[r] + 2 : ys[r + 1] - 1,
                    max(0, xs[c + 1] - 2) : xs[c + 1] + 3,
                ]
                if boundary.size and boundary.any(axis=1).mean() < 0.35:
                    merge((r, c), (r, c + 1))
            if r + 1 < rows:
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
