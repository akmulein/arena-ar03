"""Render before/after train diagnostics from saved predictions only."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.diagnose import ROOT, draw_boxes
from dataset.config import ROOT as DATA_ROOT
from common.postprocess import table_html


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="before")
    parser.add_argument("--after", default="after")
    args = parser.parse_args()
    if args.before == args.after or any(Path(name).name != name or name in {"", ".", ".."} for name in (args.before, args.after)):
        parser.error("Provide two distinct phase directory names")
    phases = {"before": args.before, "after": args.after}
    summaries = {
        phase: load(ROOT / phases[phase] / "summary.json")
        for phase in ["before", "after"]
    }
    rows = {
        phase: {
            r["id"]: r
            for r in map(
                json.loads,
                (ROOT / phases[phase] / "pages.jsonl").read_text().splitlines(),
            )
        }
        for phase in summaries
    }
    ids = list(rows["before"])
    if not ids or set(ids) != set(rows["after"]):
        raise ValueError("Both phases must contain the same nonempty set of pages")
    gain = lambda key: (
        rows["after"][key]["normal"]["teds"]
        - rows["before"][key]["normal"]["teds"]
    )
    selected = [
        (key, "Largest TEDS improvement")
        for key in sorted(ids, key=gain, reverse=True)[:4]
    ]
    used = {key for key, _ in selected}
    false_positives = lambda phase, key: (
        rows[phase][key]["raw_detection"]["pred_tables"]
        - rows[phase][key]["raw_detection"]["matched_tables"]
    )
    removed = sorted(
        (key for key in ids if key not in used),
        key=lambda key: (
            false_positives("before", key) - false_positives("after", key)
        ),
        reverse=True,
    )
    selected += [
        (key, "Largest reduction in unmatched detections")
        for key in removed[:3]
    ]
    used = {key for key, _ in selected}
    failures = sorted(
        (key for key in ids if key not in used),
        key=lambda key: (rows["after"][key]["normal"]["teds"], key),
    )
    selected += [
        (key, "Remaining failure (lowest TEDS, ID tie-break)")
        for key in failures[:3]
    ]
    gallery = ROOT / "examples"
    gallery.mkdir(exist_ok=True)
    sections = []
    thumbnails = []
    for number, (key, reason) in enumerate(selected, 1):
        truth = load((DATA_ROOT / f"train/truth/{key}.json"))
        image = cv2.imread(str(DATA_ROOT / f"train/images/{key}.jpg"))
        panels = []
        content = []
        for phase in summaries:
            result = load(ROOT / phases[phase] / f"{key}.json")
            pred = result["predictions"]["normal"]
            view = draw_boxes(image, truth["tables"], pred["tables"])
            for table in pred["tables"]:
                for cell in table["cells"]:
                    x0, y0, x1, y1 = map(round, cell["bbox"])
                    cv2.rectangle(view, (x0, y0), (x1, y1), (220, 130, 0), 1)
            cv2.putText(
                view,
                phase,
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            panels.append(view)
            metric = rows[phase][key]["normal"]
            stage = f"../{phases[phase]}/stages/{key}"
            links = " | ".join(
                f'<a href="{stage}/{file}">{label}</a>'
                for file, label in [
                    ("original.jpg", "Original"),
                    ("deskew.jpg", "Deskew"),
                    ("boxes.jpg", "Raw detector boxes"),
                ]
                if (ROOT / phases[phase] / "stages" / key / file).exists()
            )
            tables = "".join(table_html(t) for t in pred["tables"])
            content.append(
                f"<h3>{phase}: TEDS {metric['teds']:.4f}, struct {metric['teds_struct']:.4f}</h3><p>{links}</p><details><summary>Recognized tables</summary>{tables or 'Empty'}</details>"
            )
        joined = np.concatenate(panels, axis=1)
        cv2.imwrite(str(gallery / f"{key}.jpg"), joined)
        thumb = cv2.resize(
            joined, (600, round(joined.shape[0] * 600 / joined.shape[1]))
        )
        thumbnails.append(thumb)
        sections.append(
            f'<section><h2>{number}. {key}</h2><p>{reason}</p><img src="{key}.jpg">{"".join(content)}<details><summary>GT tables</summary>{"".join(table_html(t) for t in truth["tables"])}</details></section>'
        )
    (gallery / "selection.json").write_text(json.dumps(selected, indent=2))
    (gallery / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>AR-03 train diagnosis</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}td,th{border:1px solid #aaa;padding:4px}table{border-collapse:collapse}section{border-top:2px solid #555;margin-top:40px}details{overflow:auto}</style><h1>10 train examples: before / after</h1><p>Green: GT; red: final table; blue: recovered cells. Selection is diagnostic, not representative. Raw detector overlays and intermediate stages are linked separately.</p>'
        + "".join(sections)
    )
    height = max(t.shape[0] for t in thumbnails)
    sheet = np.full((height * ((len(thumbnails) + 1) // 2), 1200, 3), 255, np.uint8)
    for index, thumb in enumerate(thumbnails):
        y, x = (index // 2) * height, (index % 2) * 600
        sheet[y : y + thumb.shape[0], x : x + 600] = thumb
    cv2.imwrite(str(gallery / "contact.jpg"), sheet)
    lines = [
        "# Диагностика и исправление AR-03 — только train",
        "",
        f"Диагностическая выборка: {len(ids)} train-страниц, фазы `{args.before}` и `{args.after}`. Это не независимая оценка и не результат по всему train.",
        "",
        "## Метрики до и после",
        "",
        "Детекция: взаимно-однозначное сопоставление при IoU ≥ 0.5. Recall/precision и Cell F1 агрегируются по объектам; TEDS/структурный TEDS — среднее по страницам. Пусто — доля страниц без предсказанных таблиц.",
        "",
        "| Режим | Версия | Recall | Precision | Пусто | TEDS | TEDS-struct | Cell F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in [
        "raw_detection",
        "normal",
        "oracle_detection",
        "oracle_structure",
        "identity",
    ]:
        for phase, summary in summaries.items():
            if mode not in summary:
                continue
            r = summary[mode]
            values = [
                f"{r[k]:.6f}" if k in r else "—"
                for k in [
                    "recall",
                    "precision",
                    "empty_fraction",
                    "teds",
                    "teds_struct",
                    "cell_f1",
                ]
            ]
            lines.append(f"| {mode} | {phase} | " + " | ".join(values) + " |")
    lines += [
        "",
        "`raw_detection` — рамки до восстановления структуры; `normal` — конечный выход baseline. Oracle detection использует GT-рамки, но структура может отвергнуть регион, поэтому конечный recall может быть меньше 1. Oracle structure передаёт GT-сетку/ячейки без текста в OCR; оценка текста использует восстановленные GT-координаты/спаны. Identity передаёт полный GT в evaluator.",
        "",
        "## Ограничения",
        "",
        f"Средняя ошибка deskew: {summaries['before']['angle_mae']:.4f}° → {summaries['after']['angle_mae']:.4f}°. Результаты относятся только к выбранной train-выборке.",
        "",
        "## Визуализации",
        "",
        "[Диагностические сравнения](examples/index.html): отобранные улучшения и оставшиеся ошибки, не репрезентативная выборка.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "python -m evaluation.diagnose --phase new-run",
        "python -m evaluation.check_diagnostic_geometry",
        "python -m evaluation.diagnostic_report",
        "python -m pytest -q",
        "```",
        "",
        "Для нового кода используйте новое имя phase: кеш фазы относится к конкретной версии.",
    ]
    for phase in summaries:
        timing = ROOT / phases[phase] / "timing-20.json"
        if timing.exists():
            values = load(timing)
            mean = np.mean([r["seconds"] for r in values])
            lines += [
                "",
                f"Время обычного inference {phase}: {mean:.4f} с/страницу на первых 20 фиксированных страницах (чтение JPEG и evaluator исключены; время всех oracle-режимов не используется как latency baseline).",
            ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
