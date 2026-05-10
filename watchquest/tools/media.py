from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from watchquest.clients.feedly import fetch_feedly_items
from watchquest.clients.rss import fetch_rss_source
from watchquest.config import CACHE_FILE, PROFILE_FILE, SOURCES_FILE, WATCHLIST_FILE
from watchquest.models import Category, FeedItem, Source, WatchlistItem, parse_media_type
from watchquest.observability import observe
from watchquest.storage.json_store import read_json, write_json


def _load_sources() -> list[Source]:
    data: dict[str, Any] = read_json(SOURCES_FILE, {"feeds": []})
    return [Source.model_validate(item) for item in data.get("feeds", [])]


def _as_dicts(items: list[FeedItem]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def _dedupe(items: list[FeedItem]) -> list[FeedItem]:
    seen: set[str] = set()
    result: list[FeedItem] = []
    for item in items:
        key = item.url.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _matches_category(item: FeedItem, category: Category) -> bool:
    return category == "all" or item.category in {category, "mixed"}


def _matches_query(item: FeedItem, query: str) -> bool:
    haystack = " ".join([item.title, item.summary, " ".join(item.tags)]).lower()
    terms = [term.strip().lower() for term in query.split() if term.strip()]
    return all(term in haystack for term in terms) if terms else True


@observe("get_profile")
def get_profile_data() -> dict[str, Any]:
    return read_json(PROFILE_FILE, {})


@observe("update_profile")
def update_profile_data(likes: list[str] | None = None, dislikes: list[str] | None = None) -> dict[str, Any]:
    profile: dict[str, Any] = read_json(PROFILE_FILE, {})
    if likes:
        profile["likes"] = sorted(set(profile.get("likes", []) + likes))
    if dislikes:
        profile["dislikes"] = sorted(set(profile.get("dislikes", []) + dislikes))
    write_json(PROFILE_FILE, profile)
    return profile


@observe("list_sources")
def list_sources_data() -> list[dict[str, Any]]:
    return [source.model_dump(mode="json") for source in _load_sources()]


@observe("fetch_latest_items")
def fetch_latest_items_data(category: Category = "all", limit_per_source: int = 20) -> list[dict[str, Any]]:
    items: list[FeedItem] = []

    for source in _load_sources():
        if category != "all" and source.category not in {category, "mixed"}:
            continue
        items.extend(fetch_rss_source(source, limit=limit_per_source))

    items.extend(fetch_feedly_items(count=limit_per_source))
    items = _dedupe(items)
    items.sort(key=lambda item: item.published_at, reverse=True)

    write_json(CACHE_FILE, {"items": _as_dicts(items)})
    return _as_dicts(items)


@observe("search_cached_items")
def search_cached_items_data(
    query: str,
    category: Category = "all",
    days: int = 30,
    limit: int = 20,
) -> list[dict[str, Any]]:
    cached: dict[str, Any] = read_json(CACHE_FILE, {"items": []})
    cutoff = datetime.now(UTC) - timedelta(days=days)
    results: list[FeedItem] = []

    for raw in cached.get("items", []):
        item = FeedItem.model_validate(raw)
        if item.published_at < cutoff:
            continue
        if not _matches_category(item, category):
            continue
        if not _matches_query(item, query):
            continue
        results.append(item)

    return _as_dicts(results[:limit])


@observe("add_to_watchlist")
def add_to_watchlist_data(
    title: str,
    type: str = "unknown",  # noqa: A002
    url: str | None = None,
    reason: str = "",
    source: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = read_json(WATCHLIST_FILE, {"items": []})
    item = WatchlistItem(title=title, type=parse_media_type(type), url=url, reason=reason, source=source)
    data["items"].append(item.model_dump(mode="json"))
    write_json(WATCHLIST_FILE, data)
    return item.model_dump(mode="json")


@observe("list_watchlist")
def list_watchlist_data(type: str = "all", status: str = "planned") -> list[dict[str, Any]]:  # noqa: A002
    data: dict[str, Any] = read_json(WATCHLIST_FILE, {"items": []})
    items = data.get("items", [])
    if type != "all":
        items = [item for item in items if item.get("type") == type]
    if status != "all":
        items = [item for item in items if item.get("status") == status]
    return [dict(item) for item in items]


@observe("rate_watchlist_item")
def rate_watchlist_item_data(title: str, rating: int, comment: str = "") -> dict[str, Any]:
    data: dict[str, Any] = read_json(WATCHLIST_FILE, {"items": []})
    for item in data.get("items", []):
        if item.get("title", "").lower() == title.lower():
            item["rating"] = rating
            item["comment"] = comment
            write_json(WATCHLIST_FILE, data)
            return dict(item)
    raise ValueError(f"Item not found in watchlist: {title}")
