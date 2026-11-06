"""Page metrics; no dependency on the recognition pipeline."""

import re
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluator.teds import TEDS
from solution.postprocess import normalize, table_html
from solution.schema import validate


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - intersection
    )
    return intersection / union if union else 0


def match_tables(predicted, truth):
    if not predicted or not truth:
        return []
    overlap = np.array(
        [[iou(p["bbox"], t["bbox"]) for t in truth] for p in predicted]
    )
    # Maximize eligible match count first, then IoU.
    weights = np.where(overlap >= 0.5, overlap + min(overlap.shape) + 1, 0)
    rr, cc = linear_sum_assignment(weights, maximize=True)
    return [(int(r), int(c)) for r, c in zip(rr, cc) if overlap[r, c] >= 0.5]


def cell_key(cell, text=True):
    key = tuple(cell[k] for k in ("r0", "r1", "c0", "c1"))
    return key + (normalize(cell["text"]),) if text else key


def numeric(text):
    return bool(
        re.fullmatch(
            r"[+−-]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:\s*%)?", normalize(text)
        )
    )


def score(prediction, truth):
    validate(truth)
    valid = True
    try:
        validate(prediction)
    except (ValueError, KeyError, TypeError, OverflowError):
        prediction = {"tables": []}
        valid = False
    pp, tt = prediction["tables"], truth["tables"]
    matches = match_tables(pp, tt)
    count = len(pp) + len(tt) - len(matches)
    teds, structure, correct, num_correct = 0.0, 0.0, 0, 0
    for p, t in matches:
        ph = "<html><body>" + table_html(pp[p]) + "</body></html>"
        th = "<html><body>" + table_html(tt[t]) + "</body></html>"
        teds += TEDS().evaluate(ph, th)
        structure += TEDS(structure_only=True).evaluate(ph, th)
        pc = Counter(cell_key(c) for c in pp[p]["cells"])
        tc = Counter(cell_key(c) for c in tt[t]["cells"])
        correct += sum((pc & tc).values())
        num_correct += sum(
            numeric(c["text"]) and cell_key(c) in pc for c in tt[t]["cells"]
        )
    npred = sum(len(t["cells"]) for t in pp)
    ntrue = sum(len(t["cells"]) for t in tt)
    nnum = sum(numeric(c["text"]) for t in tt for c in t["cells"])
    return dict(
        teds=teds / count if count else float(valid),
        teds_struct=structure / count if count else float(valid),
        cell_f1=2 * correct / (npred + ntrue)
        if npred + ntrue
        else float(valid),
        numeric_accuracy=num_correct / nnum if nnum else None,
        cell_correct=correct,
        cell_predicted=npred,
        cell_true=ntrue,
        numeric_correct=num_correct,
        numeric_total=nnum,
        valid=valid,
    )
