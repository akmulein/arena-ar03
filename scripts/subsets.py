"""Fixed document-grouped subsets, stratified by table count and cell count."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def choose(records, features, size, seed):
    documents = defaultdict(list)
    for record in records:
        documents[record["document"]].append(record)
    strata = defaultdict(list)
    for document, pages in sorted(documents.items()):
        cells = np.mean([features[p["id"]]["cells"] for p in pages])
        multiple = any(features[p["id"]]["tables"] > 1 for p in pages)
        band = (
            "small" if cells <= 40 else "medium" if cells <= 100 else "large"
        )
        strata[f"{band}-{'multi' if multiple else 'single'}"].append(document)
    weights = {
        key: sum(len(documents[d]) for d in docs)
        for key, docs in strata.items()
    }
    ideal = {
        key: size * weight / len(records) for key, weight in weights.items()
    }
    targets = {key: int(value) for key, value in ideal.items()}
    remainder = size - sum(targets.values())
    for key in sorted(ideal, key=lambda k: (-(ideal[k] - targets[k]), k))[
        :remainder
    ]:
        targets[key] += 1
    rng = random.Random(seed)
    selected = set()
    for key in sorted(strata):
        docs = strata[key].copy()
        rng.shuffle(docs)
        remaining = targets[key]
        for document in docs:
            if len(documents[document]) <= remaining:
                selected.add(document)
                remaining -= len(documents[document])
            if remaining == 0:
                break
        if remaining:
            raise ValueError(f"Cannot fill {key} without splitting documents")
    subset = [r for r in records if r["document"] in selected]
    assert len(subset) == size
    return subset, weights, targets


def profile(records, features):
    cells = [features[r["id"]]["cells"] for r in records]
    tables = [features[r["id"]]["tables"] for r in records]
    return dict(
        pages=len(records),
        documents=len({r["document"] for r in records}),
        cells_mean=float(np.mean(cells)),
        cells_median=float(np.median(cells)),
        cells_p90=float(np.percentile(cells, 90)),
        tables_mean=float(np.mean(tables)),
        multi_table_fraction=float(np.mean(np.array(tables) > 1)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "public"], required=True)
    parser.add_argument("--pages", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path("data") / args.split
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    if not 0 < args.pages <= len(records):
        parser.error("pages must be within the source split size")
    features = {}
    for record in records:
        truth = json.loads(
            (root / "truth" / f"{record['id']}.json").read_text()
        )
        features[record["id"]] = dict(
            tables=len(truth["tables"]),
            cells=sum(len(t["cells"]) for t in truth["tables"]),
        )
    subset, weights, targets = choose(records, features, args.pages, args.seed)
    out = Path("data/subsets")
    out.mkdir(exist_ok=True)
    name = f"{args.split}-{args.pages}"
    (out / f"{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in subset)
    )
    report = dict(
        split=args.split,
        seed=args.seed,
        selection="whole documents; proportional table/cell-count strata",
        population=profile(records, features),
        subset=profile(subset, features),
        population_strata=weights,
        subset_strata=targets,
    )
    Path(f"reports/{name}-selection.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
