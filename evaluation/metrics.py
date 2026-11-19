"""Page-level structure, text and numeric metrics."""

import re
from collections import Counter

from common.postprocess import normalize, table_html
from common.schema import validate
from evaluation.matching import iou, match_tables

# Preserve the public imports used by earlier evaluation code.
__all__ = ['iou', 'match_tables', 'cell_key', 'numeric', 'score']

NUMERIC_PATTERN = re.compile(r'[+−-]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:\s*%)?')
CELL_INDEX_KEYS = ('r0', 'r1', 'c0', 'c1')


def cell_key(cell: dict, text: bool = True) -> tuple:
    indices = tuple(cell[key] for key in CELL_INDEX_KEYS)
    return indices + (normalize(cell['text']),) if text else indices


def numeric(text: str) -> bool:
    """Recognize numeric strings using the original evaluation rule."""
    return NUMERIC_PATTERN.fullmatch(normalize(text)) is not None


def _table_document(table: dict) -> str:
    return '<html><body>' + table_html(table) + '</body></html>'


def _score_pair(predicted: dict, truth: dict) -> tuple[float, float, int, int]:
    # Load the tree-distance backend only when a matched table needs it.
    from evaluation.teds import TEDS

    pred_html = _table_document(predicted)
    true_html = _table_document(truth)
    teds = TEDS().evaluate(pred_html, true_html)
    structure = TEDS(structure_only=True).evaluate(pred_html, true_html)

    pred_cells = Counter(cell_key(cell) for cell in predicted['cells'])
    true_cells = Counter(cell_key(cell) for cell in truth['cells'])
    correct = sum((pred_cells & true_cells).values())
    numeric_correct = sum(
        numeric(cell['text']) and cell_key(cell) in pred_cells
        for cell in truth['cells']
    )
    return teds, structure, correct, numeric_correct


def score(prediction: dict, truth: dict) -> dict:
    """Evaluate one page; unmatched tables contribute zero to TEDS."""
    validate(truth)
    valid = True
    try:
        validate(prediction)
    except (ValueError, KeyError, TypeError, OverflowError):
        prediction = {'tables': []}
        valid = False

    predicted_tables = prediction['tables']
    true_tables = truth['tables']
    matches = match_tables(predicted_tables, true_tables)
    table_count = len(predicted_tables) + len(true_tables) - len(matches)

    teds = structure = 0.0
    correct = numeric_correct = 0
    for pred_index, true_index in matches:
        pair_teds, pair_structure, pair_correct, pair_numeric = _score_pair(
            predicted_tables[pred_index], true_tables[true_index]
        )
        teds += pair_teds
        structure += pair_structure
        correct += pair_correct
        numeric_correct += pair_numeric

    predicted_count = sum(len(table['cells']) for table in predicted_tables)
    true_count = sum(len(table['cells']) for table in true_tables)
    numeric_total = sum(
        numeric(cell['text']) for table in true_tables for cell in table['cells']
    )
    cell_count = predicted_count + true_count
    return {
        'teds': teds / table_count if table_count else float(valid),
        'teds_struct': structure / table_count if table_count else float(valid),
        'cell_f1': 2 * correct / cell_count if cell_count else float(valid),
        'numeric_accuracy': numeric_correct / numeric_total if numeric_total else None,
        'cell_correct': correct,
        'cell_predicted': predicted_count,
        'cell_true': true_count,
        'numeric_correct': numeric_correct,
        'numeric_total': numeric_total,
        'valid': valid,
    }
