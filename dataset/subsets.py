"""Deterministic, diversity-balanced page subsets from existing official splits.

Documents may be sampled partially, but only within their original split.
Run public/private selection as the trusted operator, not the researcher.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataset.config import DEFAULT_SEED, EXCLUDED_PAGES, OFFICIAL_SPLITS, REVISION, ROOT
from dataset.storage import read_manifest, sha256_file

SIZES = {"train": 500, "public": 200, "private": 200}
METHOD = "ar03-marginal-stratified-v1"
# The joint table/cell-count stratum has priority. Other features are balanced
# marginally, avoiding the combinatorial explosion of full cross-stratification.
WEIGHTS = {"stratum": 4.0, "merged": 2.0, "top_span": 2.0}
NUMERIC = re.compile(r"[\s+\-−–.,%‰()<>≤≥=×*/^eE\d]+")


def band(value, low, high):
    return "small" if value <= low else "medium" if value <= high else "large"


def is_numeric(text):
    return any(c.isdigit() for c in text) and NUMERIC.fullmatch(text) is not None


def page_features(record, truth):
    tables = truth["tables"]
    cells = [cell for table in tables for cell in table["cells"]]
    nonempty = [cell["text"].strip() for cell in cells if cell["text"].strip()]
    numeric_fraction = sum(is_numeric(text) for text in nonempty) / max(1, len(nonempty))
    merged = any(c["r1"] > c["r0"] or c["c1"] > c["c0"] for c in cells)
    # Proxy only: the converted truth does not retain header-role annotations.
    top_span = any(c["r0"] <= 1 and c["c1"] > c["c0"] for c in cells)
    d = record["distortion"]
    table_band = "zero" if not tables else "single" if len(tables) == 1 else "multi"
    return {
        "stratum": f"{table_band}-{band(len(cells), 40, 100)}",
        "merged": "yes" if merged else "no",
        "top_span": "yes" if top_span else "no",
        "rows": band(max((t["rows"] for t in tables), default=0), 5, 15),
        "cols": band(max((t["cols"] for t in tables), default=0), 4, 8),
        "content": "empty" if not nonempty else (
            "text" if numeric_fraction < .25 else "numeric" if numeric_fraction > .75 else "mixed"
        ),
        "rotation": band(abs(d["angle"]), 2 / 3, 4 / 3),
        "scale": band(d["scale"], .65 + .35 / 3, .65 + 2 * .35 / 3),
        "blur": band(d["blur"], .8 / 3, 2 * .8 / 3),
        "noise": band(d["noise"], 7 / 3, 14 / 3),
        "jpeg": band(d["jpeg"], 68, 82),
    }


def load_population(root, split):
    source = root / split
    records = read_manifest(source / "manifest.jsonl")
    seen = set()
    for row in records:
        page_id = row["id"]
        if page_id in seen or page_id in EXCLUDED_PAGES:
            raise ValueError(f"Duplicate or excluded page in {split}: {page_id}")
        if row["source"] != OFFICIAL_SPLITS[split] or page_id.rsplit("_", 1)[0] != row["document"]:
            raise ValueError(f"Invalid source/document: {page_id}")
        seen.add(page_id)
    features = {
        r["id"]: page_features(r, json.loads((source / "truth" / f"{r['id']}.json").read_text(encoding="utf-8")))
        for r in records
    }
    return records, features


def distributions(records, features):
    counts = defaultdict(Counter)
    for row in records:
        for key, value in features[row["id"]].items():
            counts[key][value] += 1
    return {key: dict(sorted(values.items())) for key, values in sorted(counts.items())}


def choose(records, features, size, seed, max_pages_per_document=2):
    """Greedy marginal balancing under a hard per-document page cap.

    Targets use the page distribution of the eligible source population.
    Only whole pages are selected; source documents need not be included in full.
    """
    if size <= 0 or max_pages_per_document <= 0:
        raise ValueError("size and document cap must be positive")
    rows = sorted(records, key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source IDs")
    # Exact duplicates inside the source: retain the first deterministic ID.
    unique_by_hash = {}
    for row in rows:
        unique_by_hash.setdefault(row["sha256"], row)
    rows = list(unique_by_hash.values())
    documents = Counter(r["document"] for r in rows)
    capacity = sum(min(n, max_pages_per_document) for n in documents.values())
    if capacity < size:
        raise ValueError(f"Only {capacity} eligible pages under document cap; requested {size}")
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    keys = sorted(features[rows[0]["id"]])
    levels = sorted({(k, features[r["id"]][k]) for r in rows for k in keys})
    index = {level: i for i, level in enumerate(levels)}
    x = np.array([[index[(k, features[r["id"]][k])] for k in keys] for r in rows], dtype=np.int32)
    population = np.bincount(x.ravel(), minlength=len(levels))
    target = size * population / len(rows)
    weights = np.array([WEIGHTS.get(k, 1.0) for k, _ in levels]) / np.maximum(target, 1.0)
    selected = np.zeros(len(levels), dtype=np.int32)
    doc_indices = {d: i for i, d in enumerate(sorted(documents))}
    doc = np.array([doc_indices[r["document"]] for r in rows])
    doc_counts = np.zeros(len(documents), dtype=np.int32)
    unavailable = np.zeros(len(rows), dtype=bool)
    result = []
    for _ in range(size):
        # Incremental change of weighted squared error after adding a page.
        delta = weights * (2 * (selected - target) + 1)
        costs = delta[x].sum(axis=1)
        # Small preference for new documents, not a data-dependent quality score.
        costs += .1 * doc_counts[doc]
        costs[unavailable | (doc_counts[doc] >= max_pages_per_document)] = np.inf
        i = int(np.argmin(costs))
        if not np.isfinite(costs[i]):
            raise RuntimeError("Document capacity exhausted unexpectedly")
        result.append(rows[i])
        selected[x[i]] += 1
        doc_counts[doc[i]] += 1
        unavailable[i] = True
    ids = {r["id"] for r in result}
    subset = [r for r in records if r["id"] in ids]
    return subset


def profile(records, features):
    counts = distributions(records, features)
    n = len(records)
    docs = Counter(r["document"] for r in records)
    return {
        "pages": n,
        "documents": len(docs),
        "max_pages_per_document": max(docs.values(), default=0),
        "counts": counts,
        "fractions": {k: {v: c / n for v, c in vals.items()} for k, vals in counts.items()},
    }


def differences(population, subset):
    return {
        k: {v: round(subset["fractions"].get(k, {}).get(v, 0) - p, 6) for v, p in vals.items()}
        for k, vals in population["fractions"].items()
    }


def write_subset(root, split, size, seed, output_dir, cap, exclude_ids=(), overwrite=False):
    manifest = output_dir / f"{split}-{size}.jsonl"
    report_path = output_dir / f"{split}-{size}-selection.json"
    if not overwrite and (manifest.exists() or report_path.exists()):
        raise FileExistsError(f"Selection already exists: {manifest}; do not replace a frozen split")
    records, features = load_population(root, split)
    eligible = [r for r in records if r["id"] not in exclude_ids]
    subset = choose(eligible, features, size, seed, cap)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
    pop, selected = profile(records, features), profile(subset, features)
    report = {
        "method": METHOD, "dataset_revision": REVISION, "split": split,
        "seed": seed, "max_pages_per_document": cap,
        "source_manifest_sha256": sha256_file(root / split / "manifest.jsonl"),
        "manifest_sha256": sha256_file(manifest),
        "excluded_ids": sorted(set(exclude_ids) & {r["id"] for r in records}),
        "eligible_pages_before_exact_dedup": len(eligible),
        "population": pop, "subset": selected,
        "fraction_differences": differences(pop, selected),
        "notes": "top_span is a geometry proxy, not a semantic header label; content is heuristic",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{split}: {size} pages, {selected['documents']} documents; {manifest}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=list(SIZES), required=True)
    parser.add_argument("--pages", type=int, help="Default: 500/200/200 by split")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-pages-per-document", type=int, default=2)
    parser.add_argument("--exclude-ids-file", type=Path, help="One reviewed duplicate ID per line, before freezing")
    parser.add_argument("--overwrite", action="store_true", help="Only before the campaign is frozen")
    args = parser.parse_args()
    excluded = set(args.exclude_ids_file.read_text(encoding="utf-8").split()) if args.exclude_ids_file else set()
    write_subset(args.root, args.split, SIZES[args.split] if args.pages is None else args.pages,
                 args.seed, args.output_dir or args.root / "subsets", args.max_pages_per_document,
                 excluded, args.overwrite)


if __name__ == "__main__":
    main()
