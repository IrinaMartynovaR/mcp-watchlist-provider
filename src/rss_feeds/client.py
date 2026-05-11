from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

from domain.models import FeedItem, Source
from rss_feeds.settings import RSS_FETCH_TIMEOUT_SECONDS, RSS_USER_AGENT


@dataclass(frozen=True)
class RSSFetchResult:
    source: Source
    url: str
    ok: bool
    items: list[FeedItem]
    error: str | None = None
    status_code: int | None = None


def _parse_date(entry: Any) -> datetime:
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
    tags: list[str] = []
    for tag in getattr(entry, "tags", []) or []:
        term = getattr(tag, "term", "")
        if term:
            tags.append(term)
    return tags


def _parse_items(parsed: Any, source: Source, limit: int) -> list[FeedItem]:
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
    limit: int = 20,
    timeout: float = RSS_FETCH_TIMEOUT_SECONDS,
) -> RSSFetchResult:
    url = str(source.url)

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

        return RSSFetchResult(
            source=source,
            url=url,
            ok=bool(items),
            items=items,
            error=error,
            status_code=response.status_code,
        )
    except Exception as exc:
        return RSSFetchResult(
            source=source,
            url=url,
            ok=False,
            items=[],
            error=str(exc),
        )


def fetch_rss_source(source: Source, limit: int = 20) -> list[FeedItem]:
    return fetch_rss_source_result(source, limit=limit).items

