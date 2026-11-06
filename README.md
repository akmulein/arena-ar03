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
| `scripts/` | подготовка данных и setup Arena |
| `tests/` | тесты |
| `docs/` | документация |
| `results/` | итоговые результаты и проверки |
| `reports/` | локальные диагностические материалы, исключённые из Git |
| `data/` | локальный датасет, вне Git |

## A. Baseline без Arena


```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запуск и проверка:

```bash
python -m solution.cli --input page.jpg --output result.json --html result.html --visualization result.jpg
python -m pytest -q
```

## B. Интеграция с Arena

Arena скачивается скриптом как внешняя зависимость; `arena-case/` подключается
автоматически, вручную переносить не нужно. Docker-интеграция проверена с Tesseract 5.3.0.
Для сценария B нужны `git`, `uv` и запущенный Docker.

```bash
git clone https://github.com/akmulein/arena-ar03.git
cd arena-ar03
AR3_ROOT=$PWD
bash scripts/setup_arena.sh
cd "$HOME/.cache/ar03-arena"
uv run python -m evaluator.smoke
uv run python -m cases.ar-03.container_check --train "$AR3_ROOT/data/train"
```

Скрипт [`scripts/setup_arena.sh`](scripts/setup_arena.sh) по умолчанию клонирует
Arena в `$HOME/.cache/ar03-arena`, закрепляет commit
`669d9ed31840bfc46be10c8c63ccd8c3cd1f76ab`, выполняет `uv sync --frozen` и собирает
образы `ar03-runner:local` и `ar03-metric:local`, если их нет. Для контейнерной
проверки нужен локальный train-датасет в `data/train`.

API submit → score и эволюция ещё не проверены.

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
