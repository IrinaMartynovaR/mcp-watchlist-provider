# WatchQuest MCP

MVP-сервис рекомендаций по играм, фильмам и сериалам на базе RSS-контекста, LLM и MCP-интерфейса.

Основной пользовательский поток:

1. RSS-ленты обновляются и кешируются локально.
2. По пользовательскому запросу выбираются релевантные кандидаты.
3. Профиль вкусов и watchlist добавляются в prompt.
4. LLM формирует итоговую рекомендацию.
5. Через Langfuse можно посмотреть весь recommendation flow: какие tool-шаги выполнились, какой prompt ушёл в модель и какой ответ пришёл.

## Архитектура

Код сервиса лежит в `src`:

- `app` - backend/runtime: Telegram bot, logging и backend settings.
- `domain` - доменные модели и локальное JSON-хранилище.
- `llm_core` - нейтральный LLM-слой: settings, client factory, provider adapters и prompts.
- `mcp_server` - MCP server entrypoint.
- `mcp_tools` - runtime MCP tools и recommendation pipeline.
- `rss_feeds` - RSS fetch слой и RSS settings.

Текущая observability-модель:

- `recommend_media` - `CHAIN`
- RSS/profile/watchlist шаги - `TOOL`
- `ask_llm` - `GENERATION`


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

Проверить состояние:

```bash
docker compose -f compose/docker-compose.watchquest.yml ps
docker compose -f compose/docker-compose.watchquest.yml logs --tail 100 watchquest-mcp watchquest-telegram-bot
```

## LLM

LLM настраивается через `.env`:

```env
LLM_PROVIDER=
LLM_API_KEY=...
LLM_BASE_URL=
LLM_MODEL=
LLM_TIMEOUT_SECONDS=90
LLM_TEMPERATURE=0.3
```


## Telegram Bot

Бот работает через long polling на актуальном `aiogram`.
Он вызывает внутренний recommendation flow напрямую; внешний MCP roundtrip для Telegram-сценария не нужен.

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

Для Docker-сценария WatchQuest сам использует container-friendly endpoint:

```env
LANGFUSE_HOST=http://localhost:3000
```

А compose контейнеров прокидывает:

```env
LANGFUSE_BASE_URL=http://host.docker.internal:3000
```

После этого в Langfuse должны появляться trace-деревья примерно такого вида:

```text
recommend_media (CHAIN)
├── fetch_latest_items (TOOL)
│   └── refresh_feeds (TOOL)
├── search_cached_items (TOOL)
├── get_profile (TOOL)
├── list_watchlist (TOOL)
└── ask_llm (GENERATION)
```

В output корневого `recommend_media` дополнительно пишутся:

- `tool_usage` - краткая сводка по фактически использованным шагам pipeline;
- `mcp_tool_usage` - полная карта публичных MCP-инструментов с `true/false` для текущего сценария.

Это удобно, когда нужно увидеть не только то, что выполнилось, но и какие MCP-возможности в данном запросе не участвовали.

## Graph Memory (Mem0 + Memgraph)

Опциональная графовая память предпочтений пользователя: feedback-события (`like` / `dislike` / `watchlist` / `block_similar`) записываются в [Mem0](https://github.com/mem0ai/mem0) (embedded-библиотека внутри процесса, без отдельного Mem0-сервера) с Memgraph как graph store и локальным Chroma (`data/mem0_chroma/`) как vector store. При `recommend_media` кандидаты получают знаковый `preference_memory_score` по релевантным воспоминаниям.

Memgraph запускается отдельным compose-файлом:

```bash
docker compose -f compose/docker-compose.memgraph.yml up -d
```

Затем включить фичу в `.env`:

```env
TOOLS_MEMORY_ENABLED=true
MEMGRAPH_URL=bolt://localhost:7687
MEMGRAPH_USERNAME=memgraph
MEMGRAPH_PASSWORD=
```

Важно:

- фича выключена по умолчанию (`TOOLS_MEMORY_ENABLED=false`); при выключенной фиче memory-score всегда пустой и никаких подключений к Memgraph не происходит;
- требуется поднятый Memgraph **и** рабочий `LLM_API_KEY` — Mem0 сам делает LLM-вызовы (entity/fact extraction) и embedding-вызовы на каждую запись feedback через тот же OpenAI-совместимый endpoint, что и остальной проект;
- Mem0 требует непустые `username`/`password` для Memgraph; контейнер без включённой авторизации игнорирует их, поэтому при пустых значениях подставляется безопасный fallback `memgraph`;
- любой сбой Mem0/Memgraph/Chroma обрабатывается fail-soft: feedback и рекомендации продолжают работать, память просто не участвует;
- флаг независим от `TOOLS_RAG_ENABLED` (семантический RSS-retrieval) — это разные фичи.

Для Docker-сценария compose контейнеров сам прокидывает container-friendly endpoint:

```env
MEMGRAPH_URL=bolt://host.docker.internal:7687
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
- `import_myshows_history`

## Проверки

```bash
uv run ruff check .
uv run mypy .
uv run pytest
```
