import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.config import RuntimeConfig
from domain.models import FeedItem, Source, category_matches, parse_category, parse_media_type
from mcp_tools import media
from rss_feeds.client import RSSFetchResult


def test_parse_category_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported category"):
        parse_category("books")


def test_parse_media_type_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported media type"):
        parse_media_type("book")


def test_category_matches_bridges_movies_series() -> None:
    assert category_matches("movies_series", "series")
    assert category_matches("movies_series", "movies")
    assert category_matches("movies", "movies_series")
    assert category_matches("series", "movies_series")
    assert category_matches("mixed", "series")
    assert category_matches("games", "all")
    assert not category_matches("games", "series")
    assert not category_matches("movies_series", "games")


def test_watchlist_roundtrip(tmp_path: Path, runtime_config: RuntimeConfig) -> None:
    watchlist_file = tmp_path / "watchlist.json"

    created = media.add_to_watchlist_data(
        runtime_config,
        title="Outer Wilds",
        media_type="game",
        url="https://example.com/outer-wilds",
        reason="Curious exploration loop",
        source="manual",
    )

    assert created["title"] == "Outer Wilds"
    assert created["type"] == "game"

    items = media.list_watchlist_data(runtime_config, media_type="game", status="planned")
    assert [item["title"] for item in items] == ["Outer Wilds"]

    rated = media.rate_watchlist_item_data(runtime_config, "outer wilds", rating=10, comment="Worth it")
    assert rated["rating"] == 10
    assert rated["comment"] == "Worth it"

    persisted = json.loads(watchlist_file.read_text(encoding="utf-8"))
    assert persisted["items"][0]["rating"] == 10


def test_validate_sources_reports_per_source_status(runtime_config: RuntimeConfig) -> None:
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

    def fake_fetch(source: Source, settings: Any, limit: int = 20) -> RSSFetchResult:
        if source.name == "Good":
            return RSSFetchResult(source=source, url=str(source.url), ok=True, items=[item], status_code=200)
        return RSSFetchResult(source=source, url=str(source.url), ok=False, items=[], error="timeout")

    runtime_config.backend.sources_file.write_text(
        json.dumps({"feeds": [good_source.model_dump(mode="json"), bad_source.model_dump(mode="json")]}),
        encoding="utf-8",
    )

    result = media.validate_sources_data(runtime_config, category="games", fetch_source=fake_fetch)

    assert result["ok"] is False
    assert result["checked_count"] == 2
    assert result["errors"][0]["name"] == "Bad"
    assert result["sources"][0]["item_count"] == 1


def test_refresh_feeds_keeps_working_when_one_source_fails(
    tmp_path: Path,
    runtime_config: RuntimeConfig,
) -> None:
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

    def fake_fetch(source: Source, settings: Any, limit: int = 20) -> RSSFetchResult:
        if source.name == "Good":
            return RSSFetchResult(source=source, url=str(source.url), ok=True, items=[item], status_code=200)
        return RSSFetchResult(source=source, url=str(source.url), ok=False, items=[], error="broken feed")

    runtime_config.backend.sources_file.write_text(
        json.dumps({"feeds": [good_source.model_dump(mode="json"), bad_source.model_dump(mode="json")]}),
        encoding="utf-8",
    )

    result = media.refresh_feeds_data(runtime_config, category="all", limit_per_source=5, fetch_source=fake_fetch)

    assert result["ok"] is True
    assert result["total_count"] == 1
    assert result["items"][0]["title"] == "Useful RSS item"
    assert result["errors"][0]["name"] == "Bad"

    persisted = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert persisted["items"][0]["url"] == "https://example.com/useful"
    assert persisted["sources"][1]["error"] == "broken feed"


def test_scoped_refresh_preserves_other_sources_cache(tmp_path: Path, runtime_config: RuntimeConfig) -> None:
    games_source = Source.model_validate(
        {"name": "GamesFeed", "category": "games", "language": "en", "url": "https://example.com/games.xml"}
    )
    movies_source = Source.model_validate(
        {"name": "MoviesFeed", "category": "movies_series", "language": "en", "url": "https://example.com/movies.xml"}
    )

    def make_item(title: str, url: str, source: str, category: str) -> FeedItem:
        return FeedItem(
            title=title,
            url=url,
            source=source,
            source_language="en",
            category=parse_category(category),
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def fake_fetch(source: Source, settings: Any, limit: int = 20) -> RSSFetchResult:
        item = make_item(f"{source.name} item", f"https://example.com/{source.name}", source.name, source.category)
        return RSSFetchResult(source=source, url=str(source.url), ok=True, items=[item], status_code=200)

    runtime_config.backend.sources_file.write_text(
        json.dumps({"feeds": [games_source.model_dump(mode="json"), movies_source.model_dump(mode="json")]}),
        encoding="utf-8",
    )

    media.refresh_feeds_data(runtime_config, category="all", limit_per_source=5, fetch_source=fake_fetch)
    scoped = media.refresh_feeds_data(runtime_config, category="games", limit_per_source=5, fetch_source=fake_fetch)

    cached_sources = {item["source"] for item in scoped["items"]}
    assert cached_sources == {"GamesFeed", "MoviesFeed"}

    persisted = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert {item["source"] for item in persisted["items"]} == {"GamesFeed", "MoviesFeed"}


def test_load_sources_substitutes_rss_bridge_placeholder(
    runtime_config: RuntimeConfig,
) -> None:
    runtime_config.backend.sources_file.write_text(
        json.dumps(
            {
                "feeds": [
                    {
                        "name": "TG Channel",
                        "category": "games",
                        "language": "ru",
                        "url": "{rss_bridge}/?action=display&bridge=TelegramBridge&username=demo&format=Atom",
                    },
                    {
                        "name": "Plain",
                        "category": "games",
                        "language": "en",
                        "url": "https://example.com/feed.xml",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        runtime_config,
        rss=runtime_config.rss.model_copy(update={"rss_bridge_url": "http://rss-bridge:80"}),
    )

    sources = media._load_sources(config)

    # HttpUrl нормализует стандартный порт http (:80), опуская его.
    assert str(sources[0].url) == "http://rss-bridge/?action=display&bridge=TelegramBridge&username=demo&format=Atom"
    assert str(sources[1].url) == "https://example.com/feed.xml"
