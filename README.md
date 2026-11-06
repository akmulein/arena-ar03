# AR-03 — распознавание сканов таблиц

Учебный проект ИТМО × DeepPavlov, задание B. Цель — построить систему извлечения таблиц из изображений документов и улучшить её с помощью эволюционного поиска в DeepPavlov Arena.

**Вход:** изображение страницы с таблицами.  
**Выход:** структурированный JSON/HTML со строками, столбцами, объединёнными ячейками и текстом.

## Подход

Baseline — классический CPU-пайплайн: OpenCV для предобработки, детекции и восстановления структуры; Tesseract для OCR.

Интеграция с [DeepPavlov Arena](https://gitlab.deeppavlov.ai/deeppavlov/arena) позволяет запускать кандидатов в контейнерах и оценивать их программной метрикой. Эволюционная кампания пока не проводилась.

Подробнее: [обзор подходов](docs/domain.md), [baseline](docs/baseline.md), [устройство Arena](docs/arena.md).

## Данные и метрики

Источник — PubTables-1M. Используются 19 998 полных страниц с синтетическими искажениями сканирования.

| Раздел | Страниц |
|---|---:|
| Train | 15 998 |
| Public | 2 000 |
| Private | 2 000 |

Разбиение — по исходным документам. Данные и GT не публикуются в Git. Метод подготовки описан в [документации датасета](case/dataset/README.md).

**Основная метрика:** TEDS. Дополнительные: TEDS-struct, Cell F1 (позиция, объединения и текст), Numeric accuracy.

## Структура

| Каталог | Назначение |
|---|---|
| `solution/` | baseline OpenCV + Tesseract |
| `evaluator/` | вычисление метрик |
| `case/` | исходный формат задачи и CLI |
| `arena-case/` | адаптеры, контейнеры и метрика Arena |
| `scripts/` | подготовка данных и установка Arena |
| `tests/` | автоматические тесты |
| `docs/` | домен, алгоритм, Arena и принятые решения |
| `results/` | итоговые результаты и проверки |

`data/` и `reports/` — локальные данные и диагностические артефакты, исключённые из Git.

## Запуск baseline

```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m solution.cli --input page.jpg --output result.json --html result.html --visualization result.jpg
```

Подготовка датасета из открытого источника (требует загрузки больших архивов):

```bash
python -m scripts.download
python -m scripts.prepare --seed 42
```

## Проверка интеграции с Arena

Требуются Git, uv и запущенный Docker. Arena устанавливается как внешняя зависимость автоматически; вручную копировать кейс не нужно.

```bash
git clone https://github.com/akmulein/arena-ar03.git
cd arena-ar03
AR3_ROOT="$PWD"
bash scripts/setup_arena.sh
cd "$HOME/.cache/ar03-arena"
uv run python -m evaluator.smoke
```

Контейнерная проверка AR-03 при наличии подготовленного `data/train`:

```bash
uv run python -m cases.ar-03.container_check --train "$AR3_ROOT/data/train"
```

Скрипт установки закрепляет версию Arena, подключает кейс и собирает Docker-образы. Контейнерная версия Tesseract — 5.3.0. Подробности — в [документации кейса](arena-case/README.md).

## Результаты

Исправленный baseline, диагностическая выборка из 200 train-страниц:

| Метрика | Значение |
|---|---:|
| TEDS | 0.122096 |
| TEDS-struct | 0.255211 |
| Cell F1 | 0.029061 |
| Numeric accuracy | 0.007989 |

Выборка использовалась при исправлении baseline; это не независимая оценка на public/private.

## Статус

**ЧП1:** baseline, evaluator, датасет, локальные и контейнерные smoke-тесты, замороженный пакет метрики. 

API submit → score и полная подготовка кампании ещё не завершены.


Результаты проверок — [results/cp1](results/cp1/README.md).
