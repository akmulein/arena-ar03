
from html import escape

from solution.schema import validate


def normalize(text):
    """Normalize whitespace in recognized text."""

    return " ".join(text.split())


def table_html(table):
    """Convert a recognized table into HTML."""

    rows = [[] for _ in range(table["rows"])]

    for cell in sorted(
        table["cells"],
        key=lambda c: (c["r0"], c["c0"]),
    ):
        rs = cell["r1"] - cell["r0"] + 1
        cs = cell["c1"] - cell["c0"] + 1

        rows[cell["r0"]].append(
            f'<td rowspan="{rs}" colspan="{cs}">'
            f"{escape(normalize(cell['text']))}</td>"
        )

    body = "".join(
        "<tr>" + "".join(cells) + "</tr>"
        for cells in rows
    )

    return "<table><tbody>" + body + "</tbody></table>"


def to_html(result):
    """Convert recognition results into an HTML document."""

    validate(result)

    return (
        "<html><body>"
        + "".join(map(table_html, result["tables"]))
        + "</body></html>"
    )
