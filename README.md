# WatchQuest MCP

MVP MCP-сервер для персонального агента по фильмам, сериалам и играм.

## Что умеет

- читать RSS-источники;
- хранить профиль пользователя;
- хранить watchlist;
- отдавать MCP tools;
- отправлять traces в Langfuse.

## Быстрый старт

```bash
cp .env.example .env
uv sync
uv run watchquest-mcp
```

## Docker

```bash
docker compose up -d --build
```

## Langfuse

В проекте используется минимальный локальный Langfuse v2 stack:

- без ClickHouse;
- без Redis;
- без S3/MinIO;
- только UI и traces.

Запуск:

```bash
docker compose -f vendor/langfuse/docker-compose.yml up -d
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
