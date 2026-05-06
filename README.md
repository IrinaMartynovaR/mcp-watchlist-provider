# WatchQuest MCP

MVP MCP-сервера для персонального агента по фильмам, сериалам и играм.

## Что умеет

- читать RSS-источники из `data/sources.json`;
- опционально читать Feedly streams через API;
- хранить профиль вкусов в `data/profile.json`;
- хранить watchlist в `data/watchlist.json`;
- отдавать tools через MCP;
- опционально писать traces в Langfuse.

## Быстрый запуск локально

```bash
cp .env.example .env
uv sync
uv run watchquest-mcp
```

## Запуск в Docker

```bash
cp .env.example .env
docker compose up -d --build
```

## Langfuse

Langfuse лучше держать отдельным compose, потому что официальный v3 stack включает несколько сервисов.

```bash
./scripts/setup-langfuse-compose.sh
docker compose -f vendor/langfuse/docker-compose.yml up -d
```

Потом открой `http://localhost:3000`, создай проект и добавь ключи в `.env`.

## Подключение к MCP-клиенту

Пример локального stdio-конфига:

```json
{
  "mcpServers": {
    "watchquest": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/watchquest-mcp", "run", "watchquest-mcp"]
    }
  }
}
```

## Основные tools

- `get_profile`
- `update_profile`
- `list_sources`
- `fetch_latest_items`
- `search_cached_items`
- `add_to_watchlist`
- `list_watchlist`
- `rate_watchlist_item`

## Как пользоваться агенту

1. Сначала вызвать `fetch_latest_items`.
2. Потом искать через `search_cached_items`.
3. Для рекомендаций учитывать `get_profile` и `list_watchlist`.
4. Интересное сохранять через `add_to_watchlist`.
