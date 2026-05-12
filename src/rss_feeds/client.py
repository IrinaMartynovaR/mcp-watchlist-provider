import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

from domain.models import FeedItem, Source
from rss_feeds.settings import RSS_FETCH_ITEM_LIMIT, RSS_FETCH_TIMEOUT_SECONDS, RSS_USER_AGENT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RSSFetchResult:
    """Хранит результат чтения одного RSS-источника."""
    source: Source
    url: str
    ok: bool
    items: list[FeedItem]
    error: str | None = None
    status_code: int | None = None


def _parse_date(entry: Any) -> datetime:
    """Извлекает дату публикации RSS-entry.

    Args:
        entry: Сырые данные feedparser для одной записи.

    Returns:
        Дату публикации в UTC либо текущий момент при отсутствии даты.
    """
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not raw:
        return datetime.now(UTC)

    try:
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    except Exception:
        return datetime.now(UTC)


def _entry_tags(entry: Any) -> list[str]:
    """Извлекает непустые теги RSS-entry.

    Args:
        entry: Сырые данные feedparser для одной записи.

    Returns:
        Список найденных тегов.
    """
    tags: list[str] = []
    for tag in getattr(entry, "tags", []) or []:
        term = getattr(tag, "term", "")
        if term:
            tags.append(term)
    return tags


def _parse_items(parsed: Any, source: Source, limit: int) -> list[FeedItem]:
    """Нормализует записи feedparser в доменные RSS-элементы.

    Args:
        parsed: Результат `feedparser.parse`.
        source: Исходный RSS-источник.
        limit: Максимальное число записей для разбора.

    Returns:
        Валидированные элементы RSS-кеша.
    """
    items: list[FeedItem] = []

    for entry in parsed.entries[:limit]:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""

        if not title or not link:
            continue

        items.append(
            FeedItem(
                title=title,
                url=link,
                source=source.name,
                source_language=source.language,
                category=source.category,
                summary=summary,
                published_at=_parse_date(entry),
                tags=_entry_tags(entry),
            )
        )

    return items


def fetch_rss_source_result(
    source: Source,
    limit: int = RSS_FETCH_ITEM_LIMIT,
    timeout: float = RSS_FETCH_TIMEOUT_SECONDS,
) -> RSSFetchResult:
    """Загружает RSS-источник и возвращает диагностику результата.

    Args:
        source: Описанный в конфиге RSS-источник.
        limit: Максимальное число записей для разбора.
        timeout: Таймаут HTTP-запроса в секундах.

    Returns:
        Структуру с элементами, статусом, ошибкой и HTTP-кодом.
    """
    url = str(source.url)
    logger.info(
        "Fetching RSS source",
        extra={"source": source.name, "category": source.category, "url": url, "limit": limit},
    )

    try:
        with httpx.Client(
            follow_redirects=True,
            headers={"User-Agent": RSS_USER_AGENT},
            timeout=timeout,
        ) as client:
            response = client.get(url)
            response.raise_for_status()

        parsed = feedparser.parse(response.content)
        items = _parse_items(parsed, source=source, limit=limit)
        bozo_exception = getattr(parsed, "bozo_exception", None)
        error = str(bozo_exception) if bozo_exception else None

        if not items and error is None:
            error = "Feed has no usable entries"

        logger.info(
            "RSS source fetched",
            extra={
                "source": source.name,
                "status_code": response.status_code,
                "item_count": len(items),
                "ok": bool(items),
            },
        )
        return RSSFetchResult(
            source=source,
            url=url,
            ok=bool(items),
            items=items,
            error=error,
            status_code=response.status_code,
        )
    except Exception as exc:
        logger.warning(
            "RSS source fetch failed",
            extra={"source": source.name, "url": url, "error": str(exc)},
        )
        return RSSFetchResult(
            source=source,
            url=url,
            ok=False,
            items=[],
            error=str(exc),
        )

