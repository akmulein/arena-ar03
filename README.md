# AR-03 — распознавание сканов таблиц

Учебный проект ИТМО × DeepPavlov (задание B): изображение страницы → структурированная
таблица (JSON/HTML) со строками, столбцами, объединёнными ячейками и текстом.

Обзор подходов к распознаванию таблиц — [docs/domain.md](docs/domain.md).

## Baseline и Arena

- `solution/` — baseline: OpenCV (предобработка, детекция, структура) + Tesseract (OCR), CPU.
- `evaluator/` — TEDS, TEDS-struct, Cell F1, Numeric accuracy.
- `arena-case/` — адаптация под DeepPavlov Arena: stdin/stdout JSON, контейнеры,
  замороженная метрика. Эволюционная кампания ещё не запускалась.

## Данные и метрики

PubTables-1M: 19 998 страниц — train 15 998, public 2 000, private 2 000. Разбиение
по исходным документам, предусмотрены синтетические искажения. Основная метрика —
TEDS; также TEDS-struct, Cell F1 и Numeric accuracy. `data/` в Git не входит.

## Структура

| Каталог | Назначение |
|---|---|
| `solution/` | baseline |
| `evaluator/` | реализация метрик |
| `case/` | исходный кейс и CLI |
| `arena-case/` | интеграция с Arena |
| `scripts/` | подготовка данных |
| `tests/` | тесты |
| `docs/` | документация |
| `results/` | итоговые результаты и проверки |
| `reports/` | локальные диагностические материалы, исключённые из Git |
| `data/` | локальный датасет, вне Git |

## Установка и запуск

Локальный baseline проверен на macOS Apple Silicon, Python 3.13.7 и Tesseract 5.5.3.
Docker-интеграция Arena проверена с Tesseract 5.3.0.

```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
mkdir -p data/raw
python -m scripts.download
python -m scripts.prepare --seed 42
python -m scripts.audit
```

```bash
python -m pytest -q
python -m solution.cli --input page.jpg --output result.json --html result.html --visualization result.jpg
python -m case.metric.score result.json truth.json
```

### Arena

Разместите `arena-case/` как `cases/ar-03/` в отдельном клоне DeepPavlov Arena.
Подробные инструкции — [arena-case/README.md](arena-case/README.md).

## Результаты baseline

На 200 train-страницах, использованных для диагностики и исправлений:

| Метрика | Значение |
|---|---|
| TEDS | 0.122096 |
| TEDS-struct | 0.255211 |
| Cell F1 | 0.029061 |
| Numeric accuracy | 0.007989 |

Это диагностическая выборка; не независимая оценка и не результат public/private.

## Статус

### ЧП1

Готовы baseline, метрики, данные, Docker-интеграция, smoke-тесты и `freeze/verify`
(10 файлов). Не завершены API submit → score и подготовка реальной кампании.
Подробности — [отчёт ЧП1](results/cp1/README.md), роли и устройство Arena —
[docs/arena.md](docs/arena.md).

### ЧП2

Запуск эволюции, журнал кандидатов, кривая качества и родословная.
