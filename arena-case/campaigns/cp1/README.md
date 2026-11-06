# AR-03 CP1: окружение и замороженная метрика

Выполнена только фиксация существующего окружения и штатная заморозка метрики; это не свидетельство полной готовности кампании. Baseline, исходный evaluator, формулы, данные и splits не изменялись. Сборки, inference, smoke-тесты и проверки размеров не повторялись.

## Окружение

Полные неизменяемые ссылки образов, их идентификаторы, Python-пакеты и OCR-пакеты записаны в [environment.json](environment.json). Для воспроизведения используйте reference из этого файла, а не изменяемые теги :local. Образы доступны в текущем Docker-хранилище; публикация или экспорт на другие узлы не выполнялись. Пересборка по прежнему Dockerfile с плавающими тегами не заменяет зафиксированный образ.

Оба образа: linux/arm64, Python 3.13.18, pip 26.2.2. Базовый образ python:3.13-slim-bookworm закреплён ссылкой из фактического предыдущего build output. Runner: Tesseract/libtesseract5 5.3.0-2, английские OCR-данные tesseract-ocr-eng 1:4.1.0-2 (eng.traineddata), OSD 1:4.1.0-2; numpy 2.5.4, opencv-python-headless 5.0.0.93, pytesseract 0.3.13, Pillow 13.0.0, packaging 26.3. Метрика: numpy 2.5.4, scipy 1.18.1, apted 1.0.3, lxml 6.1.3, RapidFuzz 3.14.6. Версии прочитаны из существующих образов, зависимости не обновлялись. Использование контейнерного Tesseract 5.3.0 зафиксировано по указанию пользователя.

## Замороженный пакет

Источник: cases/ar-03/metric. Результат: cases/ar-03/campaigns/cp1/metric, включая metric.lock.json; версия метрики ar03-existing-teds-v1. Замороженный каталог не редактировать.

Команды из /Users/akmulein/arena:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m evaluator.prepare freeze --metric cases/ar-03/metric --output cases/ar-03/campaigns/cp1/metric
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m evaluator.prepare verify --metric cases/ar-03/campaigns/cp1/metric
```

Обе команды завершились с кодом 0:

```json
{"action": "freeze", "files": 10, "semantic_tests_run": false}
{"action": "verify", "files": 10, "semantic_tests_run": false}
```

Препятствий для заморозки не обнаружено. Freeze/verify подтверждают пакет, но не запускают его смысловые тесты. Установка замороженного пакета в доверенный оценщик (ARENA_METRIC_PATH), подключение закреплённых образов и полная подготовка кампании остаются отдельным этапом; сервисы не запускались. В существующем образе метрики содержится прежняя незамороженная копия исходников; при установке следует использовать именно новый каталог metric с lock-файлом, не менять его содержимое.
