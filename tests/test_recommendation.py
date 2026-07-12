from pathlib import Path
from typing import Any

import pytest

from app.settings import BackendSettings
from llm_core import embeddings as llm_embeddings
from llm_core.settings import llm_settings
from mcp_tools import memory, rag, recommendation
from mcp_tools.settings import tool_settings

VIBE_GAME_QUERY = "\u0432\u0430\u0439\u0431\u043e\u0432\u0430\u044f \u0438\u0433\u0440\u0430"
COZY_QUERY = "\u0443\u044e\u0442\u043d\u0430\u044f"


@pytest.fixture(autouse=True)
def skip_recommendation_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recommendation, "save_recommendation_result", lambda _: None)
    monkeypatch.setattr(recommendation, "semantic_search_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "semantic_memory_scores_data", lambda **_: {})


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


def test_recommend_media_prefers_semantic_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    naive_queries: list[str] = []

    def search_cached_items_data(query: str, **_: Any) -> list[dict[str, Any]]:
        naive_queries.append(query)
        return [{"title": "Naive Candidate"}]

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "search_cached_items_data", search_cached_items_data)
    monkeypatch.setattr(
        recommendation,
        "semantic_search_data",
        lambda **_: [{"title": "Semantic Hit", "category": "games", "url": "https://example.com/hit"}],
    )
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="space adventure", category="games", refresh=False)

    assert [item["title"] for item in result["candidates"]] == ["Semantic Hit"]
    assert naive_queries == []
    assert result["tool_usage"]["semantic_search"] is True
    assert result["mcp_tool_usage"]["semantic_search"] is True
    assert result["tool_usage"]["fallback_recent_search"] is False


def test_recommend_media_falls_back_to_naive_when_semantic_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "search_cached_items_data",
        lambda **_: [{"title": "Naive Candidate", "url": "https://example.com/naive"}],
    )
    monkeypatch.setattr(recommendation, "semantic_search_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="space adventure", refresh=False)

    assert [item["title"] for item in result["candidates"]] == ["Naive Candidate"]
    assert result["tool_usage"]["semantic_search"] is False
    assert result["mcp_tool_usage"]["semantic_search"] is False


def test_recommend_media_blends_memory_score_into_preference_score(monkeypatch: pytest.MonkeyPatch) -> None:
    def search_cached_items_data(query: str, **_: Any) -> list[dict[str, Any]]:
        if not query:
            return []
        return [
            {"title": "Plain", "url": "https://example.com/plain"},
            {"title": "Memorable", "url": "https://example.com/memorable"},
        ]

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "search_cached_items_data", search_cached_items_data)
    monkeypatch.setattr(
        recommendation,
        "semantic_memory_scores_data",
        lambda **_: {"url:https://example.com/memorable": 2.6},
    )
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="anything", refresh=False)

    assert [item["title"] for item in result["candidates"]] == ["Memorable", "Plain"]
    assert result["candidates"][0]["preference_score"] == 3
    assert result["candidates"][0]["preference_memory_score"] == 2.6
    assert result["candidates"][1]["preference_score"] == 0
    assert result["candidates"][1]["preference_memory_score"] == 0.0
    assert result["tool_usage"]["semantic_memory_score"] is True


def test_candidate_kind_adjusts_preference_score() -> None:
    review = recommendation._candidate_with_preference_score({"title": "Rev", "kind": "review"}, learned={})
    news = recommendation._candidate_with_preference_score({"title": "News", "kind": "news"}, learned={})
    noise = recommendation._candidate_with_preference_score({"title": "Deals", "kind": "noise"}, learned={})

    assert review["preference_score"] == 1
    assert "kind:review:+1" in review["preference_reasons"]
    assert news["preference_score"] == 0
    assert noise["preference_score"] == -2
    assert "kind:noise:-2" in noise["preference_reasons"]


def test_interleave_by_source_round_robins_candidates() -> None:
    candidates = [
        {"title": "A1", "source": "A"},
        {"title": "A2", "source": "A"},
        {"title": "A3", "source": "A"},
        {"title": "B1", "source": "B"},
    ]

    interleaved = recommendation._interleave_by_source(candidates)

    assert [item["title"] for item in interleaved] == ["A1", "B1", "A2", "A3"]
    assert [item["source"] for item in interleaved] == ["A", "B", "A", "A"]


def test_interleave_by_source_keeps_single_source_order() -> None:
    candidates = [
        {"title": "A1", "source": "A"},
        {"title": "A2", "source": "A"},
    ]

    assert recommendation._interleave_by_source(candidates) == candidates


def test_recommend_media_semantic_failure_is_soft_and_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison_network: list[str],
) -> None:
    """RAG включён, но LLM_API_KEY пуст: fast-fail до сети, наивный путь работает."""
    isolated_backend = BackendSettings(WATCHQUEST_DATA_DIR=tmp_path)
    monkeypatch.setattr(llm_embeddings, "backend_settings", isolated_backend)
    monkeypatch.setattr(memory, "backend_settings", isolated_backend)

    monkeypatch.setattr(llm_settings, "provider", "openrouter")
    monkeypatch.setattr(llm_settings, "api_key", "")
    monkeypatch.setattr(llm_settings, "embedding_model", "openai/text-embedding-3-small")
    monkeypatch.setattr(tool_settings, "rag_enabled", True)

    monkeypatch.setattr(recommendation, "semantic_search_data", rag.semantic_search_data)
    monkeypatch.setattr(recommendation, "semantic_memory_scores_data", memory.semantic_memory_scores_data)

    monkeypatch.setattr(
        rag,
        "search_cached_items_data",
        lambda **_: [{"title": "Pool Item", "url": "https://example.com/pool"}],
    )

    def naive_search(query: str, **_: Any) -> list[dict[str, Any]]:
        if not query:
            return []
        return [{"title": "Naive Candidate", "url": "https://example.com/naive"}]

    monkeypatch.setattr(recommendation, "fetch_latest_items_data", lambda **_: [])
    monkeypatch.setattr(recommendation, "search_cached_items_data", naive_search)
    monkeypatch.setattr(recommendation, "get_profile_data", lambda: {})
    monkeypatch.setattr(recommendation, "list_watchlist_data", lambda **_: [])
    monkeypatch.setattr(
        recommendation,
        "ask_llm_data",
        lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt},
    )

    result = recommendation.recommend_media_data(query="space adventure", refresh=False)

    assert poison_network == []
    assert [item["title"] for item in result["candidates"]] == ["Naive Candidate"]
    assert result["tool_usage"]["semantic_search"] is False
    assert result["llm_error"] is None
