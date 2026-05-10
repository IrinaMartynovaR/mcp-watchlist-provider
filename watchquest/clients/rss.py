from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser

from watchquest.models import FeedItem, Source


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


def fetch_rss_source(source: Source, limit: int = 20) -> list[FeedItem]:
    parsed = feedparser.parse(str(source.url))
    items: list[FeedItem] = []

    for entry in parsed.entries[:limit]:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""

        if not title or not link:
            continue

        tags = []
        for tag in getattr(entry, "tags", []) or []:
            term = getattr(tag, "term", "")
            if term:
                tags.append(term)

        items.append(
            FeedItem(
                title=title,
                url=link,
                source=source.name,
                source_language=source.language,
                category=source.category,
                summary=summary,
                published_at=_parse_date(entry),
                tags=tags,
            )
        )

    return items
