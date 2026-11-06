# CP1 — замороженная метрика

Пакет метрики AR-03 зафиксирован до запуска эволюции.

- Источник: `arena-case/metric/`
- Замороженный пакет: `arena-case/campaigns/cp1/metric/`
- Версия: `ar03-existing-teds-v1`
- Направление: `maximize`
- Проверка: `freeze/verify` успешно, 10 файлов
- Семантические тесты командами freeze/verify не запускались

Версии окружения и идентификаторы локальных Docker-образов сохранены в [environment.json](environment.json). Контейнерное окружение — Linux ARM64, Python 3.13.18, Tesseract 5.3.0.

Проверка пакета после установки Arena:

```bash
cd "$HOME/.cache/ar03-arena"
uv run python -m evaluator.prepare verify --metric cases/ar-03/campaigns/cp1/metric
```

Заморозка пакета не означает готовности всей кампании. Установка метрики в закрытый сервис, проверка API, изоляция public/private и запуск исследователей ещё не завершены.
