# WatchQuest MCP

MVP MCP-сервера для персонального агента по играм, фильмам и сериалам. Главный источник свежего контекста - RSS-фиды из `data/sources.json`, опционально расширяемые через RSS-Bridge.

## Что умеет

- проверять RSS-источники и показывать диагностику по каждому фиду;
- обновлять локальный кеш новостей из RSS;
- искать по кешу свежие материалы для рекомендаций;
- добавлять новые RSS-источники через RSS-Bridge;
- хранить профиль вкусов в `data/profile.json`;
- хранить watchlist в `data/watchlist.json`;
- генерировать рекомендации через Z.AI GLM;
- опционально писать traces в Langfuse.

## Быстрый запуск локально

```bash
cp .env.example .env
uv sync
uv run watchquest-mcp
```

## Telegram Bot

Бот работает через long polling на актуальном `aiogram`.

```env
TELEGRAM_BOT_TOKEN=...
```

```bash
uv run watchquest-telegram-bot
```

В Telegram можно писать обычным текстом:

```text
посоветуй вайбовую игру
```

Команды:

- `/start`
- `/help`
- `/recommend посоветуй атмосферный сериал`

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

## RSS: основной рабочий поток

Сначала агент проверяет источники:

```text
validate_sources(category="all", limit_per_source=3)
```

Потом обновляет кеш:

```text
refresh_feeds(category="all", limit_per_source=20)
```

После этого можно искать по свежему кешу:

```text
search_cached_items(query="cozy RPG story", category="games", days=30, limit=10)
```

Для совместимости остался старый tool `fetch_latest_items`; внутри он теперь использует `refresh_feeds`.

## RSS-Bridge

RSS-Bridge нужен для сайтов, у которых нет удобного RSS.

```bash
docker compose -f compose/docker-compose.rss-bridge.yml up -d
```

Tools:

- `list_rss_bridges`
- `build_rss_bridge_feed_url`
- `add_rss_bridge_source`

## Z.AI LLM

```env
ZAI_API_KEY=...
ZAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZAI_MODEL=glm-4.7-flash
ZAI_TIMEOUT_SECONDS=90
```

Tools:

- `ask_llm`
- `recommend_with_llm`
- `recommend_media`

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

## Подключение к MCP-клиенту

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

- `validate_sources`
- `refresh_feeds`
- `list_sources`
- `search_cached_items`
- `get_profile`
- `update_profile`
- `add_to_watchlist`
- `list_watchlist`
- `rate_watchlist_item`
- `ask_llm`
- `recommend_with_llm`
- `recommend_media`
