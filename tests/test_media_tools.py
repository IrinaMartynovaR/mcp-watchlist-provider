from __future__ import annotations

import json
from pathlib import Path

import pytest

from watchquest.models import parse_category, parse_media_type
from watchquest.tools import media


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
