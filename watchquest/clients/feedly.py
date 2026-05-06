from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from watchquest.config import FEEDLY_ACCESS_TOKEN, FEEDLY_STREAM_IDS
from watchquest.models import FeedItem

FEEDLY_API = "https://cloud.feedly.com/v3"


def _ms_to_dt(value: int | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _item_from_feedly(raw: dict[str, Any], stream_id: str) -> FeedItem | None:
    title = str(raw.get("title") or "").strip()
    url = str(raw.get("canonicalUrl") or raw.get("originId") or raw.get("id") or "").strip()
    if not title or not url:
        return None

    origin = raw.get("origin") or {}
    source = origin.get("title") or stream_id
    summary_obj = raw.get("summary") or raw.get("content") or {}
    summary = summary_obj.get("content") if isinstance(summary_obj, dict) else ""
    tags = [tag.get("label") or tag.get("id") for tag in raw.get("tags", []) if isinstance(tag, dict)]

    return FeedItem(
        title=title,
        url=url,
        source=str(source),
        source_language="unknown",
        category="mixed",
        summary=str(summary or ""),
        published_at=_ms_to_dt(raw.get("published")),
        tags=[str(tag) for tag in tags if tag],
    )


def fetch_feedly_items(count: int = 50) -> list[FeedItem]:
    if not FEEDLY_ACCESS_TOKEN or not FEEDLY_STREAM_IDS:
        return []

    headers = {"Authorization": f"Bearer {FEEDLY_ACCESS_TOKEN}"}
    items: list[FeedItem] = []

    with httpx.Client(timeout=20.0, headers=headers) as client:
        for stream_id in FEEDLY_STREAM_IDS:
            response = client.get(
                f"{FEEDLY_API}/streams/contents",
                params={"streamId": stream_id, "count": count},
            )
            response.raise_for_status()
            payload = response.json()

            for raw in payload.get("items", []):
                item = _item_from_feedly(raw, stream_id)
                if item:
                    items.append(item)

    return items
