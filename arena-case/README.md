# AR-03: локальная интеграция Arena

Структура кейса основана на cases/template. Алгоритм OpenCV + Tesseract и evaluator скопированы из /Users/akmulein/AR-3 без изменений; датасет не копируется. Это первый этап интеграции, не подготовленная кампания.

## Интерфейс

ways/baseline/solve.py получает через stdin один JSON с input (строка JSON вида {"image_base64":"..."}), assets_dir и необязательным parameters. Изображение декодируется без изменения размера; вызывается solution.pipeline.solve. Stdout — один JSON с answer, содержащим строку JSON tables. assets_dir и parameters сейчас не используются. Контракт: input ≤ 500000 символов, ответ ≤ 1 MiB. Неверный вход завершает процесс ошибкой; пустое распознавание — штатный {"tables":[]}.

metric/score.py получает {"example":{"id":"...","input":"...","reference":"JSON GT"},"answer":"JSON prediction"}. Вызывает неизменённый evaluator.metrics.score из metric/vendor и возвращает {"score":TEDS,"statistics":...}. Невалидный JSON prediction передаётся существующему evaluator как None; ошибочный reference остаётся ошибкой метрики. Среднее score по страницам — штатная агрегация Arena. TEDS-struct и счётчики Cell F1/Numeric accuracy сохраняются в statistics; вторичные F1/accuracy по набору вычисляются из сумм счётчиков, не средних постраничных долей.

## Локальное воспроизведение

Используются уже установленные окружения: Arena для её API и Python AR-3 для зависимостей решения/метрики. Установка и скачивание не выполняются. Команда проверяет два синтетических случая адаптера и выполняет ровно один inference на первой train-странице, затем вызывает Arena evaluate_answer и aggregate_results. Ответ и GT остаются в памяти; результат печатается в терминал. Это проверка протокола, не контейнерной изоляции и не качества набора.

```bash
cd /Users/akmulein/arena
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m cases.ar-03.local_check --ar3 /Users/akmulein/AR-3 --python /Users/akmulein/AR-3/.venv/bin/python
```

metric.json остаётся переносимым: python score.py. Локальная проверка разрешает эти пути в памяти в существующий интерпретатор AR-3 и абсолютный путь адаптера. Она не меняет метрику или её файл. На этапе подготовки оператор устанавливает зависимости метрики в окружение оценщика; solve.py требует cv2, numpy, pytesseract, Pillow и системный tesseract с eng. Метрике нужны numpy, scipy, apted, lxml, RapidFuzz; версии исходного окружения перечислены в AR-3/requirements.txt.

## Что ещё не готово

Образы runner и метрики собраны и проверены локально (см. ниже). Ещё нет регистрации немодельного harness (uses_model=false), замороженного пакета метрики, сервисов и условий кампании. Размер base64 проверен на пяти train-страницах (67303–255595 символов с компактным JSON), не на всём датасете; oversized input должен быть явной ошибкой, а не изменением изображения. Доступ исследователей к public/private и прежние подробные public-результаты требуют отдельного решения оператора. AR-3 не следует целиком монтировать в runner или предоставлять исследователям.

## Выполненная проверка

Локальный прогон: PMC4683861_3, input 138880 символов, валидный ответ, TEDS = 0.0. Время subprocess baseline около 0.499 с, вызова метрики и агрегации около 0.289 с; токены равны нулю. Два синтетических случая (пустой правильный ответ и невалидный JSON) прошли. Нулевой TEDS не является ошибкой интеграции; этот одиночный пример не оценивает качество всего train.

## Контейнерная проверка без исходного venv AR-3

Dockerfile имеет два target: runner с неизменённым runner/main.py Arena, OpenCV, Tesseract и eng; metric с неизменённым пакетом метрики и собственными Python-зависимостями. Build context ограничен Dockerfile.dockerignore: ни данные, ни GT не попадают в образы. Python-зависимости закреплены в container/*-requirements.txt. Docker Desktop должен быть запущен.

```bash
cd /Users/akmulein/arena
docker build --build-context arena_runner=runner --target runner -f cases/ar-03/container/Dockerfile -t ar03-runner:local cases/ar-03
docker build --target metric -f cases/ar-03/container/Dockerfile -t ar03-metric:local cases/ar-03
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m cases.ar-03.container_check --train /Users/akmulein/AR-3/data/train
```

container_check.py использует штатный Arena DockerRunner (uses_model=false) и evaluate_answer/aggregate_results, направляя команду метрики в отдельный одноразовый контейнер. Контейнеры работают без сети, с read-only rootfs, UID 10001, ограниченными ресурсами и временным /tmp. Только папка кандидата монтируется в runner read-only; изображение приходит через stdin, GT — исключительно в команду метрики. В метрику не монтируются каталоги хоста. Окружение AR-3 не используется; путь --train служит только для чтения одной страницы и её GT на стороне проверяющего процесса.

Проверяются один синтетический grid и первая train-страница, а также identity и невалидный ответ для синтетического GT. Невалидный ответ по существующей метрике получает score=0 и valid=false; это не инфраструктурная ошибка. Скрипт останавливается при провале утверждений. Он не замораживает пакет и не создаёт кампанию. Результаты текущего запуска: [отчёт](../../archive/development/ar03-container-check.md).

Перед заморозкой нужно закрепить операторские образы и решить вопрос версии OCR: контейнер использует Debian Tesseract 5.3.0-2, исходное macOS-окружение использовало 5.5.3. Поэтому подтверждена автономная исполнимость, но не побитовая идентичность локального OCR. Базовый образ и apt-репозиторий пока выбираются по тегу/дистрибутиву, а не являются полностью замороженной средой.
