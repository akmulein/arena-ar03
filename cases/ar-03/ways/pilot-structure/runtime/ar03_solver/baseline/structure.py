import logging
import os
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def structure_model():
    default_dir = (
        Path(__file__).resolve().parents[1]
        / "models/SLANet_plus"
    )

    model_dir = Path(
        os.environ.get(
            "AR03_STRUCTURE_MODEL_DIR",
            str(default_dir),
        )
    ).expanduser()

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"SLANet_plus weights not found: {model_dir}"
        )

    from paddlex import create_model

    return create_model(
        model_name="SLANet_plus",
        model_dir=str(model_dir),
        device="cpu",
    )


class TableParser(HTMLParser):
    """Convert HTML table structure into logical cells."""

    def __init__(self):
        super().__init__()
        self.row = -1
        self.col = 0
        self.cells = []
        self.occupied = set()

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row += 1
            self.col = 0
            return

        if tag not in ("td", "th") or self.row < 0:
            return

        attrs = dict(attrs)

        rowspan = max(1, int(attrs.get("rowspan", 1)))
        colspan = max(1, int(attrs.get("colspan", 1)))

        # Find an available position for the cell.
        while any(
            (r, c) in self.occupied
            for r in range(self.row, self.row + rowspan)
            for c in range(self.col, self.col + colspan)
        ):
            self.col += 1

        r0 = self.row
        r1 = self.row + rowspan - 1
        c0 = self.col
        c1 = self.col + colspan - 1

        self.cells.append(
            dict(
                r0=r0,
                r1=r1,
                c0=c0,
                c1=c1,
                text="",
            )
        )

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                self.occupied.add((r, c))

        self.col += colspan


def parse_structure(structure):
    """Parse predicted HTML tokens."""

    html = (
        structure
        if isinstance(structure, str)
        else "".join(structure)
    )

    parser = TableParser()
    parser.feed(html)
    parser.close()

    return parser.cells


def normalize_bbox(points, box):
    """Convert local cell polygon into page coordinates."""

    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0

    try:
        points = np.asarray(points, dtype=np.float32)
    except (ValueError, TypeError):
        return None

    if points.size not in (4, 8):
        return None

    points = points.reshape(-1, 2)

    if not np.isfinite(points).all():
        return None

    xmin = int(np.floor(points[:, 0].min()))
    ymin = int(np.floor(points[:, 1].min()))
    xmax = int(np.ceil(points[:, 0].max()))
    ymax = int(np.ceil(points[:, 1].max()))

    xmin = int(np.clip(xmin, 0, width))
    xmax = int(np.clip(xmax, 0, width))
    ymin = int(np.clip(ymin, 0, height))
    ymax = int(np.clip(ymax, 0, height))

    if xmin >= xmax or ymin >= ymax:
        return None

    return [
        x0 + xmin,
        y0 + ymin,
        x0 + xmax,
        y0 + ymax,
    ]


def axis_edges(cells, count, axis, lower, upper):
    """Estimate grid boundaries only for cells lacking usable geometry."""

    first_key, last_key = ("c0", "c1") if axis == 0 else ("r0", "r1")
    centers = [[] for _ in range(count)]
    spanning_centers = [[] for _ in range(count)]

    for cell in cells:
        bbox = cell.get("bbox")
        if bbox is None:
            continue
        first, last = cell[first_key], cell[last_key]
        start, end = bbox[axis], bbox[axis + 2]
        if first == last:
            centers[first].append((start + end) / 2)
        else:
            step = (end - start) / (last - first + 1)
            for index in range(first, last + 1):
                spanning_centers[index].append(start + step * (index - first + 0.5))

    known_indices = [-0.5]
    known_centers = [float(lower)]
    for index in range(count):
        values = centers[index] or spanning_centers[index]
        if values:
            known_indices.append(float(index))
            known_centers.append(float(np.median(values)))
    known_indices.append(count - 0.5)
    known_centers.append(float(upper))

    estimated = np.interp(np.arange(count), known_indices, known_centers)
    if np.any(np.diff(estimated) <= 0):
        logger.debug("Nonmonotone inferred grid axis; using uniform boundaries")
        return np.linspace(lower, upper, count + 1)

    return np.concatenate(([lower], (estimated[:-1] + estimated[1:]) / 2, [upper]))


def complete_grid(cells, box):
    """Complete a predicted logical grid while preserving valid cell boxes."""

    rows = max(cell["r1"] for cell in cells) + 1
    cols = max(cell["c1"] for cell in cells) + 1
    occupied = {
        (row, col)
        for cell in cells
        for row in range(cell["r0"], cell["r1"] + 1)
        for col in range(cell["c0"], cell["c1"] + 1)
    }
    missing = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if (row, col) not in occupied
    ]
    invalid = [cell for cell in cells if cell.get("bbox") is None]
    if not missing and not invalid:
        return cells
    if not any(cell.get("bbox") is not None for cell in cells):
        logger.debug("Cannot infer grid geometry without a valid cell box")
        return None

    x_edges = axis_edges(cells, cols, 0, box[0], box[2])
    y_edges = axis_edges(cells, rows, 1, box[1], box[3])

    for row, col in missing:
        cells.append(dict(r0=row, r1=row, c0=col, c1=col, text=""))
    for cell in cells:
        if cell.get("bbox") is not None:
            continue
        cell["bbox"] = [
            int(np.floor(x_edges[cell["c0"]])),
            int(np.floor(y_edges[cell["r0"]])),
            int(np.ceil(x_edges[cell["c1"] + 1])),
            int(np.ceil(y_edges[cell["r1"] + 1])),
        ]
    logger.debug("Repaired grid: %d missing cells, %d invalid boxes", len(missing), len(invalid))
    return cells


def recover(box, image):
    """Recover table structure using SLANet_plus."""

    x0, y0, x1, y1 = map(int, box)
    box = (x0, y0, x1, y1)

    crop = image[y0:y1, x0:x1]

    if crop.size == 0:
        logger.debug("Empty table crop: %s", box)
        return None

    crop = np.ascontiguousarray(crop)

    result = next(
        structure_model().predict(
            crop,
            batch_size=1,
        )
    )

    cells = parse_structure(result["structure"])
    bboxes = result["bbox"]

    if not cells:
        logger.debug(
            "SLANet_plus returned no cells: %s",
            box,
        )
        return None

    if len(cells) != len(bboxes):
        logger.debug(
            "Structure mismatch: %d cells, %d bboxes",
            len(cells),
            len(bboxes),
        )
        return None

    for cell, points in zip(cells, bboxes, strict=True):
        bbox = normalize_bbox(points, box)
        cell["bbox"] = bbox

    cells = complete_grid(cells, box)
    if cells is None:
        return None

    rows = max(cell["r1"] for cell in cells) + 1
    cols = max(cell["c1"] for cell in cells) + 1

    return dict(
        bbox=list(box),
        rows=rows,
        cols=cols,
        cells=sorted(
            cells,
            key=lambda c: (c["r0"], c["c0"]),
        ),
    )
