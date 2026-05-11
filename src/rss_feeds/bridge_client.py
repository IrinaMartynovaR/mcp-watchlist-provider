from typing import Any
from urllib.parse import urlencode

import httpx

from rss_feeds.settings import RSS_BRIDGE_BASE_URL, RSS_BRIDGE_TIMEOUT_SECONDS


def list_bridges() -> dict[str, Any]:
    with httpx.Client(timeout=RSS_BRIDGE_TIMEOUT_SECONDS) as client:
        response = client.get(RSS_BRIDGE_BASE_URL, params={"action": "list"})
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise ValueError("Unexpected RSS-Bridge response: expected object")
    bridges = data.get("bridges")
    if not isinstance(bridges, dict):
        raise ValueError("Unexpected RSS-Bridge response: missing bridges")
    return bridges


def build_bridge_feed_url(bridge: str, params: dict[str, str] | None = None, format: str = "Atom") -> str:  # noqa: A002
    query = {"action": "display", "bridge": bridge, "format": format}
    if params:
        query.update(params)
    return f"{RSS_BRIDGE_BASE_URL}/?{urlencode(query)}"

