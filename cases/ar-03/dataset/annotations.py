"""Index source annotations and convert them to the shared table schema."""

import json
import sqlite3
import tarfile
from pathlib import Path

from ar03_solver.common.schema import validate
from dataset.config import ARCHIVES, ROOT
from dataset.storage import check_storage, verify_archive

CELL_FIELDS = ("row_nums", "column_nums", "pdf_bbox", "pdf_text_content")


def group_tables_by_page(tables: list[dict]) -> dict[str, list[dict]]:
    pages: dict[str, list[dict]] = {}
    for source in tables:
        table = source.copy()
        table["cells"] = [
            {field: cell[field] for field in CELL_FIELDS}
            for cell in source["cells"]
        ]
        page_id = f"{table['pmc_id']}_{table['pdf_page_index']}"
        pages.setdefault(page_id, []).append(table)
    return pages


def populate_index(db: sqlite3.Connection, root: Path) -> None:
    local, remote = ARCHIVES["pdf"]
    verify_archive(local, remote, root)

    with tarfile.open(root / "raw" / local, mode="r|gz") as archive:
        for number, member in enumerate(archive):
            if number % 10_000 == 0:
                check_storage(root)
            if not member.isfile() or not member.name.endswith(".json"):
                continue

            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read annotation: {member.name}")
            with stream:
                pages = group_tables_by_page(json.load(stream))

            db.executemany(
                "INSERT INTO pages (id, doc, split, tables) VALUES (?, ?, ?, ?)",
                (
                    (
                        page_id,
                        tables[0]["pmc_id"],
                        tables[0]["split"],
                        json.dumps(tables),
                    )
                    for page_id, tables in pages.items()
                ),
            )


def index_pdf(root: Path = ROOT) -> sqlite3.Connection:
    """Return the existing index, or build it in a single transaction."""
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "annotations.sqlite")
    try:
        with db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS pages "
                "(id TEXT PRIMARY KEY, doc TEXT, split TEXT, tables TEXT)"
            )
            if db.execute("SELECT count(*) FROM pages").fetchone()[0]:
                return db
            populate_index(db, root)

        count = db.execute("SELECT count(*) FROM pages").fetchone()[0]
        print(f"PDF index complete: {count}", flush=True)
        return db
    except BaseException:
        db.close()
        raise


def pixel_bbox(values, page_bbox, width: int, height: int) -> list[float]:
    """Map a PDF box to the original page image without changing rounding."""
    x0, y0, x1, y1 = page_bbox
    return [
        (values[0] - x0) * width / (x1 - x0),
        (values[1] - y0) * height / (y1 - y0),
        (values[2] - x0) * width / (x1 - x0),
        (values[3] - y0) * height / (y1 - y0),
    ]


def convert_cell(cell: dict, page_bbox, width: int, height: int) -> dict:
    rows = sorted(cell["row_nums"])
    cols = sorted(cell["column_nums"])
    if rows != list(range(rows[0], rows[-1] + 1)) or cols != list(
        range(cols[0], cols[-1] + 1)
    ):
        raise ValueError("nonrectangular source cell")

    return {
        "r0": rows[0],
        "r1": rows[-1],
        "c0": cols[0],
        "c1": cols[-1],
        "text": cell["pdf_text_content"],
        "bbox": pixel_bbox(cell["pdf_bbox"], page_bbox, width, height),
    }


def convert(tables: list[dict], width: int, height: int) -> dict:
    """Convert all tables on a page; reject pages with excluded annotations."""
    converted = []
    for table in tables:
        if table["exclude_for_detection"] or table["exclude_for_structure"]:
            raise ValueError("excluded source annotation")

        page_bbox = table["pdf_full_page_bbox"]
        cells = [
            convert_cell(cell, page_bbox, width, height)
            for cell in table["cells"]
        ]
        converted.append(
            {
                "bbox": pixel_bbox(
                    table["pdf_table_bbox"], page_bbox, width, height
                ),
                "rows": len(table["rows"]),
                "cols": len(table["columns"]),
                "cells": cells,
            }
        )

    return validate({"tables": converted})
