# WatchQuest MCP

MVP MCP-сервиса для персонального агента по играм, фильмам и сериалам. Основной источник свежего контекста - RSS-фиды из `data/sources.json`; RSS-Bridge может использоваться как отдельный инфраструктурный сервис для источников без удобного RSS.

## Архитектура

Код сервиса лежит в `src`:

- `app` - backend/runtime: Telegram bot, logging, observability и backend settings.
- `domain` - доменные модели и локальное JSON-хранилище.
- `llm_core` - нейтральный LLM-слой: settings, client factory, provider adapters и prompts.
- `mcp_server` - MCP server entrypoint.
- `mcp_tools` - только runtime MCP tools.
- `rss_feeds` - RSS fetch слой и RSS settings.

LLM-слой не завязан на конкретного провайдера. Сейчас реализован provider adapter для Z.AI GLM-compatible API, но наружу код работает через `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.

## Быстрый запуск

```bash
cp .env.example .env
uv sync
uv run watchquest-mcp
```

## Docker

```bash
cp .env.example .env
docker compose -f compose/docker-compose.watchquest.yml up -d --build
```

Этот compose поднимает:

- `rss-bridge` на `http://localhost:3001`
- `watchquest-mcp`
- `watchquest-telegram-bot`

Для Telegram в `.env` должен быть заполнен `TELEGRAM_BOT_TOKEN`.

## LLM

LLM настраивается через `.env`:

```env
LLM_PROVIDER=zai_glm
LLM_API_KEY=...
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4.7-flash
LLM_TIMEOUT_SECONDS=90
LLM_TEMPERATURE=0.7
```

Старые `ZAI_*` env-переменные пока поддерживаются как fallback, чтобы существующие локальные `.env` не ломались сразу.

## Telegram Bot

Бот работает через long polling на актуальном `aiogram`.

```bash
uv run watchquest-telegram-bot
```

Можно писать обычным текстом или через команду:

```text
/recommend посоветуй вайбовую игру
```

## RSS

Основной рабочий поток:

```text
validate_sources(category="all", limit_per_source=3)
refresh_feeds(category="all", limit_per_source=20)
search_cached_items(query="cozy RPG story", category="games", days=30, limit=10)
recommend_media(query="посоветуй вайбовую игру", category="games")
```

RSS-Bridge запускается как отдельная инфраструктура. Сервис не содержит admin tools для генерации RSS-Bridge URL; готовые feed URL должны лежать в `data/sources.json`.

```bash
docker compose -f compose/docker-compose.rss-bridge.yml up -d
```

## Langfuse

Локальный Langfuse запускается отдельным compose-файлом:

```bash
docker compose -f compose/docker-compose.langfuse.yml up -d
```

После запуска открыть `http://localhost:3000`, создать проект, скопировать API keys и добавить их в `.env`:

```env
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

## MCP client

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

## Public MCP Tools

- `get_profile`
- `update_profile`
- `list_sources`
- `validate_sources`
- `refresh_feeds`
- `search_cached_items`
- `add_to_watchlist`
- `list_watchlist`
- `rate_watchlist_item`
- `recommend_media`
