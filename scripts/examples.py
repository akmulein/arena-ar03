"""Select honest train examples using already computed page scores."""

import argparse
import json
from pathlib import Path

import cv2

from solution.cli import visualize
from solution.postprocess import table_html, to_html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    args = parser.parse_args()
    pages = [json.loads(line) for line in args.scores.read_text().splitlines()]
    detected = [p for p in pages if p["cell_predicted"] > 0]
    ranked = sorted(detected, key=lambda p: p["teds"], reverse=True)
    missed = [p for p in pages if p["cell_predicted"] == 0]
    selected = [("best", p) for p in ranked[:2]]
    selected += [
        ("structure-success", p) for p in ranked if p["teds_struct"] == 1
    ][:1]
    selected += [("error", p) for p in ranked[-2:]]
    selected += [("missed", p) for p in missed[:1]]
    out = Path("reports/examples")
    out.mkdir(exist_ok=True)
    sections = []
    for label, page in selected:
        page_id = page["id"]
        image = cv2.imread(f"data/train/images/{page_id}.jpg")
        run = json.loads(
            Path(f"reports/predictions/train/{page_id}.json").read_text()
        )
        truth = json.loads(
            Path(f"data/train/truth/{page_id}.json").read_text()
        )
        before = visualize(image, truth)
        after = visualize(image, run["prediction"])
        preview = cv2.hconcat([before, after])
        cv2.imwrite(str(out / f"{label}-{page_id}.jpg"), preview)
        (out / f"{label}-{page_id}.html").write_text(
            to_html(run["prediction"])
        )
        comparison = (
            '<!doctype html><meta charset="utf-8">'
            "<style>body{font:14px sans-serif}main{display:flex;gap:24px}"
            "section{width:50%;overflow:auto}table{border-collapse:collapse}"
            "td{border:1px solid #aaa;padding:5px}</style>"
            "<main><section><h2>Эталон</h2>"
            + "".join(table_html(t) for t in truth["tables"])
            + "</section><section><h2>Baseline</h2>"
            + "".join(table_html(t) for t in run["prediction"]["tables"])
            + "</section></main>"
        )
        (out / f"{label}-{page_id}-compare.html").write_text(comparison)
        sections.append(
            f"## {label}: {page_id}\n\n"
            f"TEDS={page['teds']:.6f}; TEDS-struct={page['teds_struct']:.6f}; "
            f"F1={page['cell_f1']:.6f}. "
            "Слева эталон, справа baseline.\n\n"
            f"![{label}]({label}-{page_id}.jpg)\n\n"
            f"[Сравнение текста]({label}-{page_id}-compare.html)\n"
        )
    (out / "README.md").write_text(
        f"# Примеры из оценённых {len(pages)} train-страниц\n\n"
        "Лучшие доступные результаты не обязательно "
        "полностью правильны; фактические метрики указаны "
        "под каждым примером.\n\n" + "\n".join(sections)
    )


if __name__ == "__main__":
    main()
