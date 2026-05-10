from __future__ import annotations

from typing import Any

from watchquest.clients.rss_bridge import build_bridge_feed_url, list_bridges
from watchquest.config import SOURCES_FILE
from watchquest.models import Category, parse_category
from watchquest.observability import observe
from watchquest.storage.json_store import read_json, write_json


@observe("list_rss_bridges")
def list_rss_bridges_data(query: str = "", limit: int = 20) -> list[dict[str, Any]]:
    bridges = list_bridges()
    needle = query.strip().lower()
    results: list[dict[str, Any]] = []

    for bridge_id, raw in bridges.items():
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or bridge_id)
        description = str(raw.get("description") or "")
        haystack = f"{bridge_id} {name} {description}".lower()
        if needle and needle not in haystack:
            continue
        results.append(
            {
                "id": bridge_id,
                "name": name,
                "description": description,
                "uri": raw.get("uri"),
                "parameters": raw.get("parameters"),
            }
        )
        if len(results) >= limit:
            break

    return results


@observe("build_rss_bridge_feed_url")
def build_rss_bridge_feed_url_data(
    bridge: str,
    params: dict[str, str] | None = None,
    format: str = "Atom",  # noqa: A002
) -> dict[str, str]:
    url = build_bridge_feed_url(bridge=bridge, params=params, format=format)
    return {"url": url}


@observe("add_rss_bridge_source")
def add_rss_bridge_source_data(
    name: str,
    bridge: str,
    params: dict[str, str] | None = None,
    category: Category = "mixed",
    language: str = "unknown",
    format: str = "Atom",  # noqa: A002
) -> dict[str, Any]:
    data: dict[str, Any] = read_json(SOURCES_FILE, {"feeds": []})
    feeds = data.setdefault("feeds", [])
    if not isinstance(feeds, list):
        raise ValueError("Invalid sources file: feeds must be a list")

    url = build_bridge_feed_url(bridge=bridge, params=params, format=format)
    source = {
        "name": name,
        "category": parse_category(category),
        "language": language,
        "url": url,
    }

    existing = [item for item in feeds if isinstance(item, dict) and item.get("url") == url]
    if not existing:
        feeds.append(source)
        write_json(SOURCES_FILE, data)

    return source
