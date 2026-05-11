import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from domain.models import FeedItem, Source, parse_category, parse_media_type
from mcp_tools import media
from rss_feeds.client import RSSFetchResult


def test_parse_category_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported category"):
        parse_category("books")


def test_parse_media_type_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported media type"):
        parse_media_type("book")


def test_watchlist_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    watchlist_file = tmp_path / "watchlist.json"
    monkeypatch.setattr(media, "WATCHLIST_FILE", watchlist_file)

    created = media.add_to_watchlist_data(
        title="Outer Wilds",
        type="game",
        url="https://example.com/outer-wilds",
        reason="Curious exploration loop",
        source="manual",
    )

    assert created["title"] == "Outer Wilds"
    assert created["type"] == "game"

    items = media.list_watchlist_data(type="game", status="planned")
    assert [item["title"] for item in items] == ["Outer Wilds"]

    rated = media.rate_watchlist_item_data("outer wilds", rating=10, comment="Worth it")
    assert rated["rating"] == 10
    assert rated["comment"] == "Worth it"

    persisted = json.loads(watchlist_file.read_text(encoding="utf-8"))
    assert persisted["items"][0]["rating"] == 10


def test_validate_sources_reports_per_source_status(monkeypatch: pytest.MonkeyPatch) -> None:
    good_source = Source.model_validate(
        {"name": "Good", "category": "games", "language": "en", "url": "https://example.com/good.xml"}
    )
    bad_source = Source.model_validate(
        {"name": "Bad", "category": "games", "language": "en", "url": "https://example.com/bad.xml"}
    )
    item = FeedItem(
        title="Good news",
        url="https://example.com/good-news",
        source="Good",
        source_language="en",
        category="games",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    def fake_sources() -> list[Source]:
        return [good_source, bad_source]

    def fake_fetch(source: Source, limit: int = 20) -> RSSFetchResult:
        if source.name == "Good":
            return RSSFetchResult(source=source, url=str(source.url), ok=True, items=[item], status_code=200)
        return RSSFetchResult(source=source, url=str(source.url), ok=False, items=[], error="timeout")

    monkeypatch.setattr(media, "_load_sources", fake_sources)
    monkeypatch.setattr(media, "fetch_rss_source_result", fake_fetch)

    result = media.validate_sources_data(category="games")

    assert result["ok"] is False
    assert result["checked_count"] == 2
    assert result["errors"][0]["name"] == "Bad"
    assert result["sources"][0]["item_count"] == 1


def test_refresh_feeds_keeps_working_when_one_source_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_file = tmp_path / "feed_cache.json"
    good_source = Source.model_validate(
        {"name": "Good", "category": "mixed", "language": "en", "url": "https://example.com/good.xml"}
    )
    bad_source = Source.model_validate(
        {"name": "Bad", "category": "mixed", "language": "en", "url": "https://example.com/bad.xml"}
    )
    item = FeedItem(
        title="Useful RSS item",
        url="https://example.com/useful",
        source="Good",
        source_language="en",
        category="mixed",
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    def fake_sources() -> list[Source]:
        return [good_source, bad_source]

    def fake_fetch(source: Source, limit: int = 20) -> RSSFetchResult:
        if source.name == "Good":
            return RSSFetchResult(source=source, url=str(source.url), ok=True, items=[item], status_code=200)
        return RSSFetchResult(source=source, url=str(source.url), ok=False, items=[], error="broken feed")

    monkeypatch.setattr(media, "CACHE_FILE", cache_file)
    monkeypatch.setattr(media, "_load_sources", fake_sources)
    monkeypatch.setattr(media, "fetch_rss_source_result", fake_fetch)

    result = media.refresh_feeds_data(category="all", limit_per_source=5)

    assert result["ok"] is True
    assert result["total_count"] == 1
    assert result["items"][0]["title"] == "Useful RSS item"
    assert result["errors"][0]["name"] == "Bad"

    persisted = json.loads(cache_file.read_text(encoding="utf-8"))
    assert persisted["items"][0]["url"] == "https://example.com/useful"
    assert persisted["sources"][1]["error"] == "broken feed"

