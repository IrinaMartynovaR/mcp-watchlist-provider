from typing import Any

import pytest

from mcp_tools import recommendation

VIBE_GAME_QUERY = "\u0432\u0430\u0439\u0431\u043e\u0432\u0430\u044f \u0438\u0433\u0440\u0430"
COZY_QUERY = "\u0443\u044e\u0442\u043d\u0430\u044f"


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
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": f"ok: {prompt[:20]}"},
    )

    result = recommendation.recommend_media_data(query="cozy mystery", category="games")

    assert calls == ["fetch", "search:cozy mystery"]
    assert result["fetched_count"] == 1
    assert result["candidate_count"] == 1
    assert result["model"] == "test-model"
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
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt},
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
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="vibe", category="games", refresh=False)

    assert [item["title"] for item in result["candidates"]] == ["Game news", "Movie news"]
    assert "Do not invent titles" in result["recommendation"]


def test_recommend_media_expands_vibe_query(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    def search_cached_items_data(query: str, **_: Any) -> list[dict[str, Any]]:
        queries.append(query)
        if query == COZY_QUERY:
            return [{"title": "SUMMERHOUSE", "category": "games", "url": "https://example.com/summerhouse"}]
        return []

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "search_cached_items_data", search_cached_items_data)
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt},
    )

    result = recommendation.recommend_media_data(query=VIBE_GAME_QUERY, category="games", refresh=False)

    assert queries[:2] == [VIBE_GAME_QUERY, COZY_QUERY]
    assert result["candidates"] == [
        {"title": "SUMMERHOUSE", "category": "games", "url": "https://example.com/summerhouse"}
    ]
