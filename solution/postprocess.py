from html import escape

from solution.schema import validate


def normalize(text):
    return " ".join(text.split())


def table_html(table):
    rows = []
    for r in range(table["rows"]):
        cells = []
        for cell in sorted(table["cells"], key=lambda c: (c["r0"], c["c0"])):
            if cell["r0"] == r:
                rs = cell["r1"] - r + 1
                cs = cell["c1"] - cell["c0"] + 1
                cells.append(
                    f'<td rowspan="{rs}" colspan="{cs}">'
                    f"{escape(normalize(cell['text']))}</td>"
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def to_html(result):
    validate(result)
    return (
        "<html><body>"
        + "".join(map(table_html, result["tables"]))
        + "</body></html>"
    )
