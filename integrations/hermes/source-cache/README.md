# Необязательный кеш исходников Hermes

Для сборки без загрузки Git-истории положите сюда `source.tar`, полученный из закреплённого upstream. Он используется только во время сборки и не сохраняется в итоговом образе или Git.

```bash
git -C /path/to/hermes-agent archive --format=tar 86fa9e6ca1de7ee71132d2be0103f3e37d1fa872 --output=/path/to/arena/integrations/hermes/source-cache/source.tar
```

Без архива общий Dockerfile загружает тот же коммит. Подключения кейсов могут передать свой прежний кеш тому же установщику; код установки принадлежит общей интеграции.
