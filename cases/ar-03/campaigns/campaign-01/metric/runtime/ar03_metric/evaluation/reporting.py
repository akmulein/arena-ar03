"""Benchmark aggregation and report serialization."""

import json
from pathlib import Path

import numpy as np

from ar03_metric.evaluation.detection import detection_summary


def summarize(
    results: list[dict], *, fingerprint: str, split: str,
    population_pages: int, sampling: str,
) -> dict:
    """Keep the original macro TEDS and micro cell/detection aggregation."""
    if not results:
        raise ValueError('No pages to evaluate')
    cell_correct = sum(row['cell_correct'] for row in results)
    cell_total = sum(row['cell_predicted'] + row['cell_true'] for row in results)
    numeric_correct = sum(row['numeric_correct'] for row in results)
    numeric_total = sum(row['numeric_total'] for row in results)
    seconds = [row['seconds'] for row in results]

    return {
        'fingerprint': fingerprint,
        'split': split,
        'pages': len(results),
        'population_pages': population_pages,
        'sampling': sampling,
        'teds': float(np.mean([row['teds'] for row in results])),
        'teds_struct': float(np.mean([row['teds_struct'] for row in results])),
        'cell_f1': 2 * cell_correct / max(1, cell_total),
        'numeric_accuracy': numeric_correct / max(1, numeric_total),
        'seconds_mean': float(np.mean(seconds)),
        'seconds_total': sum(seconds),
        'errors': sum(row['error'] is not None for row in results),
        **detection_summary(
            sum(row['detection_tp'] for row in results),
            sum(row['detection_fp'] for row in results),
            sum(row['detection_fn'] for row in results),
        ),
    }


def write_reports(directory: Path, name: str, summary: dict, results: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{name}.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    with (directory / f'{name}-pages.jsonl').open('w', encoding='utf-8') as stream:
        for row in results:
            stream.write(json.dumps(row) + '\n')
