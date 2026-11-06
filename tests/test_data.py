from copy import deepcopy

import pytest

from scripts.prepare import convert


def source():
    return dict(
        exclude_for_detection=False,
        exclude_for_structure=False,
        pdf_full_page_bbox=[0, 0, 600, 800],
        pdf_table_bbox=[60, 80, 300, 400],
        rows=[{}],
        columns=[{}],
        cells=[
            dict(
                row_nums=[0],
                column_nums=[0],
                pdf_bbox=[60, 80, 300, 400],
                pdf_text_content="42",
            )
        ],
    )


def test_pdf_to_pixels():
    result = convert([source()], 1200, 1600)
    assert result["tables"][0]["bbox"] == [120, 160, 600, 800]
    assert result["tables"][0]["cells"][0]["text"] == "42"


def test_all_page_tables_required():
    second = deepcopy(source())
    second["exclude_for_structure"] = True
    with pytest.raises(ValueError):
        convert([source(), second], 1200, 1600)


def test_nonrectangular_source():
    table = source()
    table["cells"][0]["row_nums"] = [0, 2]
    with pytest.raises(ValueError):
        convert([table], 1200, 1600)


def test_subset_preserves_documents_and_seed():
    from scripts.subsets import choose

    records = [
        dict(id=f"d{d}_{p}", document=f"d{d}")
        for d in range(40)
        for p in range(2)
    ]
    features = {r["id"]: dict(tables=1, cells=30) for r in records}
    subset, _, targets = choose(records, features, 20, 42)
    again, _, _ = choose(records, features, 20, 42)
    assert subset == again
    assert len(subset) == sum(targets.values()) == 20
    documents = {r["document"] for r in subset}
    assert len(documents) == 10
    assert subset == [r for r in records if r["document"] in documents]
