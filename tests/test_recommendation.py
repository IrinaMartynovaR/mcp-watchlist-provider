from __future__ import annotations

from typing import Any

import pytest

from watchquest.tools import recommendation


def test_recommend_media_orchestrates_refresh_search_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fetch_latest_items_data(**_: Any) -> list[dict[str, Any]]:
        calls.append("fetch")
        return [{"title": "Fetched"}]

    def search_cached_items_data(query: str, **_: Any) -> list[dict[str, Any]]:
        calls.append(f"search:{query}")
        return [{"title": "Outer Wilds", "category": "games"}]

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", fetch_latest_items_data)
    monkeypatch.setattr(recommendation, "search_cached_items_data", search_cached_items_data)
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {"likes": ["exploration"]})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [{"title": "Disco Elysium"}])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "z-ai", "model": "glm-4.7-flash", "response": f"ok: {prompt[:20]}"},
    )

    result = recommendation.recommend_media_data(query="cozy mystery", category="games")

    assert calls == ["fetch", "search:cozy mystery"]
    assert result["fetched_count"] == 1
    assert result["candidate_count"] == 1
    assert result["model"] == "glm-4.7-flash"
    assert "ok:" in result["recommendation"]


def test_recommend_media_falls_back_to_recent_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    def search_cached_items_data(query: str, **_: Any) -> list[dict[str, Any]]:
        queries.append(query)
        if query:
            return []
        return [{"title": "Recent Item"}]

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "search_cached_items_data", search_cached_items_data)
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "z-ai", "model": "glm-4.7-flash", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="very specific", refresh=False)

    assert queries == ["very specific", ""]
    assert result["refreshed"] is False
    assert result["candidates"] == [{"title": "Recent Item"}]


def test_recommend_media_prefers_exact_category_on_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def search_cached_items_data(query: str, **_: Any) -> list[dict[str, Any]]:
        if query:
            return []
        return [
            {"title": "Movie news", "category": "mixed"},
            {"title": "Game news", "category": "games"},
        ]

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "search_cached_items_data", search_cached_items_data)
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "z-ai", "model": "glm-4.7-flash", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="vibe", category="games", refresh=False)

    assert [item["title"] for item in result["candidates"]] == ["Game news", "Movie news"]
    assert "Do not invent titles" in result["recommendation"]


def test_recommend_media_expands_vibe_query(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    def search_cached_items_data(query: str, **_: Any) -> list[dict[str, Any]]:
        queries.append(query)
        if query == "уютная":
            return [{"title": "SUMMERHOUSE", "category": "games", "url": "https://example.com/summerhouse"}]
        return []

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "search_cached_items_data", search_cached_items_data)
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "z-ai", "model": "glm-4.7-flash", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="вайбовая игра", category="games", refresh=False)

    assert queries[:2] == ["вайбовая игра", "уютная"]
    assert result["candidates"] == [
        {"title": "SUMMERHOUSE", "category": "games", "url": "https://example.com/summerhouse"}
    ]
