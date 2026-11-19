"""Selection, cache and report regressions without training or inference."""

import json

import pytest

from evaluation import benchmark
from evaluation.cache import load_prediction, save_prediction
from evaluation.reporting import summarize


def write_manifest(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def test_manifest_order_precedes_limit(tmp_path, monkeypatch):
    rows = [{'id': f'p{i}', 'sha256': str(i)} for i in range(5)]
    write_manifest(tmp_path / 'manifest.jsonl', rows)
    subset = tmp_path / 'subset.jsonl'
    write_manifest(subset, [rows[4], rows[2], rows[0]])
    monkeypatch.setitem(benchmark.SPLITS, 'train', ('Train_Part1', 5))

    selected, population = benchmark.select_records(tmp_path, 'train', subset, 2)
    first, _ = benchmark.select_records(tmp_path, 'train', None, 2)

    assert population == 5
    assert selected == [rows[4], rows[2]]
    assert first == rows[:2]


def test_duplicate_manifest_is_rejected(tmp_path, monkeypatch):
    row = {'id': 'p0', 'sha256': 'test'}
    write_manifest(tmp_path / 'manifest.jsonl', [row])
    subset = tmp_path / 'subset.jsonl'
    write_manifest(subset, [row, row])
    monkeypatch.setitem(benchmark.SPLITS, 'train', ('Train_Part1', 1))
    with pytest.raises(ValueError, match='duplicate'):
        benchmark.select_records(tmp_path, 'train', subset, None)


def test_prediction_cache(tmp_path):
    path = tmp_path / 'predictions' / 'p0.json'
    run = {
        'fingerprint': 'baseline-v1', 'image_sha256': 'image-v1',
        'prediction': {'tables': []}, 'detections': [], 'error': None, 'seconds': 1.0,
    }
    save_prediction(path, run)
    assert load_prediction(path, 'baseline-v1', 'image-v1') == run
    assert load_prediction(path, 'baseline-v2', 'image-v1') is None
    assert load_prediction(path, 'baseline-v1', 'image-v2') is None
    path.write_text('{broken', encoding='utf-8')
    assert load_prediction(path, 'baseline-v1', 'image-v1') is None


def test_macro_and_micro_aggregation():
    first = dict(teds=1, teds_struct=1, cell_correct=1, cell_predicted=1,
                 cell_true=1, numeric_correct=1, numeric_total=1,
                 detection_tp=1, detection_fp=0, detection_fn=0,
                 seconds=2.0, error=None)
    second = dict(teds=0, teds_struct=0.5, cell_correct=0, cell_predicted=0,
                  cell_true=3, numeric_correct=0, numeric_total=3,
                  detection_tp=0, detection_fp=0, detection_fn=3,
                  seconds=4.0, error='example')
    report = summarize([first, second], fingerprint='test', split='train',
                       population_pages=5, sampling='test')
    assert report['teds'] == 0.5
    assert report['teds_struct'] == 0.75
    assert report['cell_f1'] == pytest.approx(0.4)
    assert report['numeric_accuracy'] == 0.25
    assert report['detection_f1'] == pytest.approx(0.4)
    assert report['errors'] == 1
    assert report['seconds_mean'] == 3.0
    assert report['seconds_total'] == 6.0
