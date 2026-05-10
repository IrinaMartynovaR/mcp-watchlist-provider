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
docker compose -f compose/docker-compose.watchquest.yml up -d --build
```

## Langfuse

Для локальной разработки используется официальный developer docker compose stack Langfuse с зафиксированной версией `4.5.1`.

Подготовка и запуск:

```bash
docker compose -f compose/docker-compose.langfuse.yml up -d
```

После запуска:

1. Открыть `http://localhost:3000`
2. Создать проект
3. Скопировать API keys
4. Добавить ключи в `.env`

```env
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

## Z.AI LLM

WatchQuest can use Z.AI GLM for direct answers and recommendations.

```env
ZAI_API_KEY=...
ZAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZAI_MODEL=glm-4.7-flash
ZAI_TIMEOUT_SECONDS=90
```

Tools:

- `ask_llm`
- `recommend_with_llm`

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
- `ask_llm`
- `recommend_with_llm`

## Как пользоваться агенту

1. Сначала вызвать `fetch_latest_items`.
2. Потом искать через `search_cached_items`.
3. Для рекомендаций учитывать `get_profile` и `list_watchlist`.
4. Интересное сохранять через `add_to_watchlist`.
