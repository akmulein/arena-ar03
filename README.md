# AR-03 — распознавание сканов таблиц

CPU baseline OpenCV + Tesseract и независимый оценщик для ИТМО × DeepPavlov.
Основа структуры — официальный track-b-arena-skill; исходный README сохранён
в docs/upstream. Кампании Arena, обучение и упаковка навыка не реализуются.

## Установка (macOS Apple Silicon)

```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

```bash
mkdir -p data/raw
python -m scripts.download
python -m scripts.prepare --seed 42
python -m scripts.audit
```

Подробности сопоставления PDF и страниц, отбора и искажений — case/dataset/README.md.
Данные и private GT не включать в Git. Отчёт готовности создаётся только полным
аудитом, не самим фактом скачивания файлов.

## Запуск

```bash
python -m solution.cli --input page.jpg --output result.json --html result.html --visualization result.jpg
python -m pytest -q
python -m scripts.benchmark --split train --manifest data/subsets/train-2000.jsonl
python -m scripts.benchmark --split public
python -m case.metric.score result.json truth.json
```

## Структура

- solution/: preprocessing, detection, structure, ocr, postprocess, schema, cli.
- evaluator/: исходный TEDS с минимальными адаптациями и метрики страницы.
- scripts/: подготовка, искажения, аудит, последовательный benchmark.
- tests/: контракт, метрики, геометрия, сетка, объединения, несколько таблиц.
- case/: формат задачи, описание данных, metric.json, вход baseline.
- docs/: алгоритм, решения по метрикам, официальные требования.
- data/: локальные изображения, эталоны, manifests (вне Git).
- reports/: реальные результаты запусков и аудит.

Ограничения описаны в docs/solution.md; формулы агрегации — docs/adr/0001-metrics.md.

## Подготовленные данные

Полный аудит завершён: train 16000 страниц / 15101 документ,
public 2000 / 1936, private 2000 / 1928. Пересечений документов: 0.
Проверены декодирование каждого JPEG, SHA256, официальные filelist,
полнота сеток и координаты каждой ячейки.

Размер после удаления временных PDF-архива и SQLite-индекса — 2 249 003 566 байт
(2,25 GB). До очистки данные и окружение занимали около 20,2 GB, ниже 120 GB.
Отчёт: `reports/dataset-audit.json`. Private baseline не запускается.
Повторный `scripts.prepare` при готовых manifests выполняет аудит; для новой
выборки нужны исходные архивы по командам выше.

Проверено на macOS 15.5 ARM64, Python 3.13.7, Tesseract 5.5.3.

## Итоговые измерения

[Таблица результатов и ограничения](reports/summary.md). Train оценён только
на сохранённом префиксе 8476 страниц; это не результат по всему train.
Public оценён полностью (2000 страниц). Private не запускался.

Фиксированные subsets создаются так:

```bash
python -m scripts.subsets --split train --pages 2000 --seed 42
python -m scripts.subsets --split public --pages 500 --seed 42
```

Train-2000 предназначен для дальнейших экспериментов и сейчас отдельно не прогонялся.
Public-500 — предложение для быстрых циклов, основной public остаётся 2000.

Сохранённые результаты можно переагрегировать без нового OCR:

```bash
python -m scripts.benchmark --split train --cached-only
python -m scripts.examples --scores reports/train-cached-8476-pages.jsonl
```

Проверки: 26 pytest-тестов проходят, Ruff — без замечаний.
