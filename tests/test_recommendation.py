from typing import Any

import pytest

from mcp_tools import recommendation

VIBE_GAME_QUERY = "\u0432\u0430\u0439\u0431\u043e\u0432\u0430\u044f \u0438\u0433\u0440\u0430"
COZY_QUERY = "\u0443\u044e\u0442\u043d\u0430\u044f"


@pytest.fixture(autouse=True)
def skip_recommendation_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recommendation, "save_recommendation_result", lambda _: None)


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
    assert result["candidates"][0]["title"] == "Recent Item"
    assert result["candidates"][0]["preference_score"] == 0


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
    assert result["candidates"][0]["title"] == "SUMMERHOUSE"
    assert result["candidates"][0]["preference_score"] == 0


def test_recommend_media_ranks_candidates_by_learned_preferences(monkeypatch: pytest.MonkeyPatch) -> None:
    def search_cached_items_data(query: str, **_: Any) -> list[dict[str, Any]]:
        if not query:
            return []
        return [
            {
                "title": "Weak source",
                "category": "movies_series",
                "source": "Noisy",
                "tags": [],
                "url": "https://example.com/weak",
            },
            {
                "title": "Liked source",
                "category": "movies_series",
                "source": "Trusted",
                "tags": ["Comedy"],
                "url": "https://example.com/liked",
            },
        ]

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "search_cached_items_data", search_cached_items_data)
    monkeypatch.setattr(
        recommendation,
        "get_profile_data",
        lambda: {
            "learned_preferences": {
                "sources": {"Trusted": 3, "Noisy": -2},
                "tags": {"comedy": 2},
            }
        },
    )
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="comedy", category="movies_series", refresh=False)

    assert [item["title"] for item in result["candidates"]] == ["Liked source", "Weak source"]
    assert result["candidates"][0]["preference_score"] == 5


def test_recommend_media_returns_rss_candidates_when_llm_is_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "search_cached_items_data",
        lambda **_: [{"title": "RSS Candidate", "category": "games", "url": "https://example.com/rss"}],
    )
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])

    def raise_rate_limit(_: str) -> dict[str, str]:
        raise RuntimeError("LLM rate limit or quota exceeded.")

    monkeypatch.setattr(recommendation, "ask_llm_data", raise_rate_limit)

    result = recommendation.recommend_media_data(query="посоветуй игру", category="games", refresh=False)

    assert result["llm_error"] == "LLM rate limit or quota exceeded."
    assert result["candidates"][0]["title"] == "RSS Candidate"
    assert "RSS Candidate" in result["recommendation"]
