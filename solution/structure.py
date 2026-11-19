
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

    for cell, points in zip(cells, bboxes):
        bbox = normalize_bbox(points, box)

        if bbox is None:
            logger.debug(
                "Invalid bbox for cell (%d, %d)",
                cell["r0"],
                cell["c0"],
            )
            return None

        cell["bbox"] = bbox

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
