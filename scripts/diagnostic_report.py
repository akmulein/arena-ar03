"""Render before/after train diagnostics from saved predictions only."""

import html
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.diagnose import ROOT, draw_boxes
from solution.postprocess import table_html


def load(path):
    return json.loads(path.read_text())


def main():
    summaries = {
        phase: load(ROOT / phase / "summary.json")
        for phase in ["before", "after"]
    }
    rows = {
        phase: {
            r["id"]: r
            for r in map(
                json.loads,
                (ROOT / phase / "pages.jsonl").read_text().splitlines(),
            )
        }
        for phase in summaries
    }
    ids = list(rows["before"])
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
        truth = load(Path(f"data/train/truth/{key}.json"))
        image = cv2.imread(f"data/train/images/{key}.jpg")
        panels = []
        content = []
        for phase in summaries:
            result = load(ROOT / phase / f"{key}.json")
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
            stage = f"../{phase}/stages/{key}"
            links = " | ".join(
                f'<a href="{stage}/{file}">{label}</a>'
                for file, label in [
                    ("original.jpg", "Original"),
                    ("deskew.jpg", "Deskew"),
                    ("binary.png", "Binary"),
                    ("horizontal.png", "Horizontal"),
                    ("vertical.png", "Vertical"),
                    ("boxes.jpg", "Raw detector boxes"),
                ]
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
    sheet = np.full((height * 5, 1200, 3), 255, np.uint8)
    for index, thumb in enumerate(thumbnails):
        y, x = (index // 2) * height, (index % 2) * 600
        sheet[y : y + thumb.shape[0], x : x + 600] = thumb
    cv2.imwrite(str(gallery / "contact.jpg"), sheet)
    lines = [
        "# Диагностика и исправление AR-03 — только train",
        "",
        "Фиксированная случайная выборка: 200 страниц, 199 исходных документов, 231 таблица; seed=42. Манифест: `train-200.jsonl`. Это диагностическая выборка для разработки, не независимая оценка и не результат по всему train. Public/private в этой работе не использовались.",
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
        "## Найденные ошибки и изменения",
        "",
        "- Детектор требовал связного контура: раздельные горизонтальные границы пропускались. Добавлена группировка согласованных горизонтальных отрезков; сохранена ветка связной сетки.",
        "- Картинки с рамками проходили фильтр линий. Добавлены проверки плотности и текстоподобных компонент; это эвристика, диаграммы могут оставаться ложными детекциями.",
        "- Восстановление структуры требовало вертикальных границ и отбрасывало почти все GT-регионы. Для неполной разлиновки добавлено разделение по строкам текста и устойчивым межколоночным пробелам.",
        "- Median blur и внутренние отступы обрезали мелкие символы. OCR получает исходный deskew grayscale, полную ячейку, увеличение 3× и белое поле.",
        "- Угол оценивался на уменьшенном изображении с грубым шагом. Теперь используются полноразмерные линии и взвешенная медиана углов.",
        "",
        "## Проверки и ограничения",
        "",
        "Геометрия: `geometry.json`. Проверены 16 572 рамки, совпадение реализации с cv2.transform, обратное преобразование точек и 30 пиксельных маркеров через реальную аугментацию (максимальная ошибка границы < 0.72 px). Исходные PDF-аннотации уже удалены: это не повторный аудит переноса из PDF. Обратное вращение AABB расширяет рамку, поэтому проверяются точки, а не равенство AABB после round trip.",
        "",
        f"Средняя ошибка deskew: {summaries['before']['angle_mae']:.4f}° → {summaries['after']['angle_mae']:.4f}°. Хеши evaluator, schema/postprocess, train manifest и выбранных GT сохранены в `frozen.json` и проверены без изменений.",
        "",
        "Сложные многострочные ячейки, объединённые заголовки без разделителей, большие пробелы между горизонтальными линиями и очень мелкий текст остаются ограничениями. Улучшение на этой выборке не доказывает обобщение; дальнейшую разработку вести на train.",
        "",
        "## Визуализации",
        "",
        "[10 сравнений до/после](examples/index.html): четыре наибольших улучшения TEDS, три наибольших уменьшения числа несопоставленных детекций, три оставшиеся ошибки. Это явно отобранные диагностические примеры. Для всех 200 страниц сохранены original, deskew, binary, horizontal, vertical и raw boxes в `before/stages` и `after/stages`.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        "python -m scripts.diagnose --phase new-run",
        "python -m scripts.check_diagnostic_geometry",
        "python -m scripts.diagnostic_report",
        "python -m pytest -q",
        "```",
        "",
        "Исходный baseline сохранён в `before/source/solution`; итоговый — в `after/source/solution`. Для нового кода используйте новое имя phase: кеш фазы относится к конкретной версии. Промежуточная неокончательная попытка сохранена отдельно в `trial-1`.",
    ]
    for phase in summaries:
        timing = ROOT / phase / "timing-20.json"
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
