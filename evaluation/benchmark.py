"""Run the baseline on a fixed dataset split and evaluate predictions."""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2

from dataset.config import REPORTS_DIR, ROOT, SPLITS
from dataset.storage import read_manifest
from evaluation.cache import baseline_fingerprint, load_prediction, save_prediction
from evaluation.detection import score_detection
from evaluation.metrics import score
from evaluation.reporting import summarize, write_reports

logger = logging.getLogger(__name__)


def select_records(
    split_root: Path, split: str, manifest: Path | None, limit: int | None,
) -> tuple[list[dict], int]:
    """Select pages without changing the source-manifest order or split size."""
    records = read_manifest(split_root / 'manifest.jsonl')
    population_pages = len(records)
    if population_pages != SPLITS[split][1]:
        raise ValueError('Full benchmark requires the complete split')

    if manifest is not None:
        selected = read_manifest(manifest)
        available = {row['id']: row for row in records}
        if len({row['id'] for row in selected}) != len(selected):
            raise ValueError('Subset contains duplicate pages')
        if any(available.get(row['id']) != row for row in selected):
            raise ValueError('Subset records must match the source manifest')
        records = selected

    if limit is not None:
        records = records[:limit]
    return records, population_pages


def predict_page(image_path: Path, fingerprint: str, image_sha256: str) -> dict:
    # Keep metric-only imports independent of the recognition pipeline.
    from baseline.pipeline import solve

    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f'Cannot read input image: {image_path}')

    detections = []
    start = time.perf_counter()
    try:
        prediction = solve(image, detections=detections)
        error = None
    except (RuntimeError, ValueError) as exc:
        # Retain the original benchmark policy for page-level recognition errors.
        prediction = {'tables': []}
        error = str(exc)
    seconds = time.perf_counter() - start
    if error is not None:
        logger.warning('Recognition failed on %s: %s', image_path.name, error)

    return {
        'fingerprint': fingerprint,
        'image_sha256': image_sha256,
        'prediction': prediction,
        'detections': detections,
        'seconds': seconds,
        'error': error,
    }


def evaluate_page(
    record: dict, split_root: Path, cache_dir: Path,
    fingerprint: str, *, cached_only: bool = False,
) -> dict:
    page_id = record['id']
    cache_path = cache_dir / f'{page_id}.json'
    run = load_prediction(cache_path, fingerprint, record['sha256'])
    if run is None:
        if cached_only:
            raise ValueError(f'Stale cache: {page_id}')
        run = predict_page(
            split_root / 'images' / f'{page_id}.jpg',
            fingerprint,
            record['sha256'],
        )
        save_prediction(cache_path, run)
    elif run['error'] is not None:
        logger.warning('Reusing cached recognition error on %s: %s', page_id, run['error'])

    truth_path = split_root / 'truth' / f'{page_id}.json'
    truth = json.loads(truth_path.read_text(encoding='utf-8'))
    metrics = score(run['prediction'], truth)
    metrics.update(score_detection(run['detections'], truth['tables']))
    return {
        'id': page_id,
        'seconds': run['seconds'],
        'error': run['error'],
        **metrics,
    }


def report_name(args: argparse.Namespace, page_count: int) -> str:
    name = args.manifest.stem if args.manifest else args.split
    if args.cached_only:
        name += f'-cached-{page_count}'
    elif args.limit:
        name += f'-first-{args.limit}'
    return name


def sampling_description(args: argparse.Namespace) -> str:
    if args.cached_only:
        return 'cached available pages; not a full-train estimate'
    if args.manifest:
        return str(args.manifest)
    if args.limit:
        return 'first pages'
    return 'complete split'


def run_benchmark(args: argparse.Namespace) -> dict:
    split_root = ROOT / args.split
    records, population_pages = select_records(
        split_root, args.split, args.manifest, args.limit
    )
    cache_dir = REPORTS_DIR / 'predictions' / args.split
    if args.cached_only:
        records = [row for row in records if (cache_dir / f"{row['id']}.json").is_file()]
    if not records:
        raise ValueError('No pages to evaluate')

    fingerprint = baseline_fingerprint()
    results = []
    for record in records:
        results.append(evaluate_page(
            record, split_root, cache_dir, fingerprint, cached_only=args.cached_only
        ))
        if len(results) % 100 == 0:
            print(args.split, len(results), flush=True)

    summary = summarize(
        results, fingerprint=fingerprint, split=args.split,
        population_pages=population_pages, sampling=sampling_description(args),
    )
    write_reports(REPORTS_DIR, report_name(args, len(results)), summary, results)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'public'], required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cached-only', action='store_true')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error('limit must be positive')
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    print(json.dumps(run_benchmark(args), indent=2))


if __name__ == '__main__':
    main()
