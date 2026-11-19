# AR-03 — распознавание сканов таблиц

Учебный проект ИТМО × DeepPavlov, задание B. Цель — построить систему извлечения таблиц из изображений документов и улучшить её с помощью эволюционного поиска в DeepPavlov Arena.

**Вход:** изображение страницы с таблицами.  
**Выход:** структурированный JSON/HTML со строками, столбцами, объединёнными ячейками и текстом.

## Подход

Текущий CPU-пайплайн: OpenCV для предобработки и восстановления структуры, `PicoDet_layout_1x_table` через PaddleX для поиска таблиц, Tesseract для OCR. Описание прежнего классического baseline и его результаты сохранены отдельно.

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

**Основная метрика:** TEDS. Дополнительные: TEDS-struct, Cell F1 (позиция, объединения и текст), Numeric accuracy. Качество самого детектора измеряется отдельно: `detection_precision`, `detection_recall`, `detection_f1` при IoU ≥ 0.5, без учёта ячеек и текста.

`scripts.benchmark` сохраняет эти метрики и счётчики `detection_tp/fp/fn` в общую сводку и постраничный JSONL. Оцениваются рамки до восстановления сетки и OCR, возвращённые в координаты исходной страницы; отброшенная на этапе структуры таблица остаётся в оценке детектора. Итоговые показатели вычисляются из суммарных TP/FP/FN по выборке. Для одновременно пустых prediction и GT принята оценка 1; при отсутствии только одной стороны — 0. Старые кеши без рамок детектора пересчитываются обычным запуском benchmark; `--cached-only` требует актуального кеша. Замороженная метрика Arena не меняется.

## Структура

| Каталог | Назначение |
|---|---|
| `solution/` | OpenCV + PicoDet для поиска таблиц + Tesseract |
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
pip install -r requirements-picodet.txt
python -m pytest -q
python -m solution.cli --input page.jpg --output result.json --html result.html --visualization result.jpg
```

PicoDet получает выровненную grayscale-страницу в трёх каналах, использует порог уверенности 0.5 и NMS. Модель загружается один раз на процесс из `models/PicoDet_layout_1x_table/`; каталог весов исключён из Git. Другой путь задаётся через `AR03_TABLE_MODEL_DIR=/absolute/path/to/PicoDet_layout_1x_table`. Локальная загрузка не обращается к механизму поиска хостингов PaddleX. Восстановление сетки, OCR и метрики не менялись; инференс новой модели здесь не проверялся. API модели описан в [документации PaddleX](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html).

Для новой копии проекта скачайте файлы `config.json`, `inference.json`, `inference.pdiparams` и `inference.yml` из [официального репозитория модели](https://huggingface.co/PaddlePaddle/PicoDet_layout_1x_table/tree/a85fc8c201fff0db451cde0d6664976fcc05f8ac) в указанный каталог. Эта ревизия использована для локальных весов. Нужен именно каталог с файлами модели, а не путь к архиву.

Копия baseline в `arena-case/ways/baseline/` и ранее собранные образы остаются прежними; локальная оценка через `scripts.benchmark` использует обновлённый `solution/`.

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

Прежний baseline с OpenCV-детекцией, диагностическая выборка из 200 train-страниц (не результаты PicoDet):

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
