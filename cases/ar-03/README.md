# AR-03: воспроизведение и сдача проекта

ДЗ по курсу Agentic LLM Sys. Код арены взят здесь: https://gitlab.deeppavlov.ai/deeppavlov/arena/-/commits/main

Мое решение в cases/ar-03

Задача: взять скан таблицы и получить ее структуру в JSON/HTML. Запрещено использовать тяжелые ГПУ модели, инференс на ЦПУ.

Baseline:
1. Детекция таблицы с DocLayout-YOLO
2. Определение структуры с SLANet_plus 
3. Выделение текста с PP-OCRv6_small_rec 

Метрики:
1. TEDS (Tree-Edit-Distance-based Similarity) - основная
2. TEDS struct (учитывает только структуру, без текста)
3. Detection precision, recall, f1 (детекция целой таблицы)
4. Numeric accuracy (точность распознавания числовых значений в таблице)
5. Cell F1 (насколько хорошо алгоритм восстанавливает ячейки таблицы)

Реализация: на основе бейзлайна запускается эволюция решения на арене. Дообучать модели запрещено. 

**Структура репозитория**

| Каталог | Содержимое |
|---|---|
| `common/`, `prompts/` | Протоколы Arena и инструкции оркестратора и исследователей |
| `evaluator/`, `runner/` | Штатное оценивание, журнал решений и исполнение кандидатов |
| `cases/ar-03/ways/` | Baseline и варианты решения |
| `cases/ar-03/metric/` | Программные метрики и агрегация |
| `cases/ar-03/dataset/` | Загрузка данных, имитация сканов и восстановление выборок |
| `cases/ar-03/harness/` | Dockerfile и зависимости CPU-решателя |
| `cases/ar-03/deploy/` | Compose, операторский API, проверки безопасности и `operatorctl.py` |
| `cases/ar-03/campaigns/` | Условия кампаний, свидетельства подготовки, референс baseline и экспортированные результаты |
| `cases/ar-03/tests/` | Тесты интеграции и компонентов |

`client.py` отправляет кандидатов через API, `monitor.py` строит графики прогресса, `operator_artifacts.py` сохраняет визуализации из готовых предсказаний.

**Датасет**

Используется PubTables-1M. Подготовка преобразует разметку и имитирует сканирование: поворот, масштабирование, размытие, шум и JPEG-сжатие.

Зафиксированы выборки: train — 500, public — 200, private — 200 страниц. 

Train открыт исследователям. Public возвращает только общий TEDS, private предназначен для финального оценивания и не раскрывает результаты агентам. 

**Docker-изоляция**

- **Оператор на хосте:** доверенные оценщики, замороженная метрика, данные, журнал и результаты.
- **Researcher:** Codex, оркестратор и исследователи. Доступны разрешённый код, train и API Arena. Закрытые данные, реализация метрики и Docker daemon недоступны.
- **DockerRunner:** отдельный временный контейнер для исполнения кандидата на CPU. Получает текущую страницу, код и разрешённые веса. 

Сеть researcher ограничена API Arena и OpenAI для Codex. Перед запуском автоматически проверяется изоляция. Нарушение условий отзывает разрешение на оценивание — **fail-closed**.

Соседний каталог `AR-3-operator/` создаётся автоматически и хранит данные, веса, токены, рабочее состояние и закрытые артефакты. 

**Запуск**

```sh
export AR03_RESOURCES="$(pwd)-operator"

uv sync --frozen --python 3.13
uv pip install --python .venv/bin/python -r cases/ar-03/dataset/requirements.txt -r cases/ar-03/metric/requirements.txt

.venv/bin/python -B cases/ar-03/deploy/operatorctl.py bootstrap --resources "$AR03_RESOURCES"
.venv/bin/python -B cases/ar-03/deploy/operatorctl.py prepare --resources "$AR03_RESOURCES" --mode pilot
```

Запуск сервисов и первая авторизация Codex через подписку ChatGPT:

```sh
.venv/bin/python -B cases/ar-03/deploy/operatorctl.py start --resources "$AR03_RESOURCES"
.venv/bin/python -B cases/ar-03/deploy/operatorctl.py containers-up --resources "$AR03_RESOURCES"
.venv/bin/python -B cases/ar-03/deploy/operatorctl.py verify --resources "$AR03_RESOURCES"
.venv/bin/python -B cases/ar-03/deploy/operatorctl.py login --resources "$AR03_RESOURCES"
```

Единственная команда запуска или продолжения подготовленного пилота:

```sh
.venv/bin/python -B cases/ar-03/deploy/operatorctl.py run --resources "$AR03_RESOURCES" --source-ref 'Запуск или продолжение пилота владельцем проекта'
```

Повторная подготовка при продолжении не нужна: журнал, результаты и оставшийся бюджет сохраняются.

Прогресс и экспорт завершённой кампании:

```sh
.venv/bin/python -B cases/ar-03/deploy/operatorctl.py progress --resources "$AR03_RESOURCES"
.venv/bin/python -B cases/ar-03/deploy/operatorctl.py export --resources "$AR03_RESOURCES"
```

Экспорт сохраняет разрешённую историю и снимки решений в `cases/ar-03/campaigns/pilot-01/publication/`, закрытые визуализации остаются у оператора.