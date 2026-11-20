"""Shared, strict output contract; independent of recognition code."""

import math


def validate(result):
    if not isinstance(result, dict) or not isinstance(
        result.get("tables"), list
    ):
        raise ValueError("tables must be a list")
    for table in result["tables"]:
        rows, cols = table["rows"], table["cols"]
        if any(type(n) is not int or n <= 0 for n in (rows, cols)):
            raise ValueError("rows and cols must be positive integers")
        box = table["bbox"]
        if (
            len(box) != 4
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in box
            )
            or box[0] < 0
            or box[1] < 0
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise ValueError("invalid bbox")
        occupied = set()
        for cell in table["cells"]:
            r0, r1, c0, c1 = (cell[k] for k in ("r0", "r1", "c0", "c1"))
            if (
                any(type(v) is not int for v in (r0, r1, c0, c1))
                or not 0 <= r0 <= r1 < rows
                or not 0 <= c0 <= c1 < cols
                or not isinstance(cell["text"], str)
            ):
                raise ValueError("invalid cell")
            positions = {
                (r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            }
            if occupied & positions:
                raise ValueError("overlapping cells")
            occupied |= positions
        if len(occupied) != rows * cols:
            raise ValueError("grid must include empty cells")
    return result
