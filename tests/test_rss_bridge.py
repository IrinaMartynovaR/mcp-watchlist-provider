import json
from pathlib import Path

import pytest

from mcp_tools import rss_bridge
from rss_feeds.bridge_client import build_bridge_feed_url


def test_build_bridge_feed_url_encodes_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rss_feeds.bridge_client.RSS_BRIDGE_BASE_URL", "http://localhost:3001")

    url = build_bridge_feed_url("ExampleBridge", params={"q": "cozy rpg"}, format="Atom")

    assert url == "http://localhost:3001/?action=display&bridge=ExampleBridge&format=Atom&q=cozy+rpg"


def test_list_rss_bridges_filters_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rss_bridge,
        "list_bridges",
        lambda: {
            "SteamBridge": {"name": "Steam", "description": "Games"},
            "NewsBridge": {"name": "News", "description": "World news"},
        },
    )

    results = rss_bridge.list_rss_bridges_data(query="steam")

    assert [item["id"] for item in results] == ["SteamBridge"]


def test_add_rss_bridge_source_writes_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources_file = tmp_path / "sources.json"
    monkeypatch.setattr(rss_bridge, "SOURCES_FILE", sources_file)
    monkeypatch.setattr(rss_bridge, "build_bridge_feed_url", lambda **_: "http://localhost:3001/feed")

    source = rss_bridge.add_rss_bridge_source_data(
        name="Bridge Feed",
        bridge="ExampleBridge",
        params={"q": "games"},
        category="games",
        language="en",
    )

    assert source["url"] == "http://localhost:3001/feed"
    persisted = json.loads(sources_file.read_text(encoding="utf-8"))
    assert persisted["feeds"] == [source]

