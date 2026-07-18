from dataclasses import replace
from functools import partial
from typing import Any

from app.config import RuntimeConfig
from llm_core.client import create_llm_client
from llm_core.embeddings import create_embeddings
from mcp_tools import rag, recommendation

VIBE_GAME_QUERY = "вайбовая игра"
COZY_QUERY = "уютная"


def _service(
    config: RuntimeConfig,
    *,
    search: Any,
    fetch: Any = None,
    profile: Any = None,
    watchlist: Any = None,
    analyze_query: Any = None,
    semantic: Any = None,
    memory_scores: Any = None,
    ask_llm: Any = None,
) -> recommendation.RecommendationService:
    """Собирает recommendation service с явными test doubles."""
    dependencies = recommendation.RecommendationDependencies(
        fetch_latest=fetch or (lambda **_: []),
        search_cached=search,
        get_profile=profile or (lambda: {}),
        list_watchlist=watchlist or (lambda **_: []),
        analyze_query=analyze_query or (lambda **_: {"medium": None}),
        semantic_search=semantic or (lambda **_: []),
        semantic_memory_scores=memory_scores or (lambda **_: {}),
        ask_llm=ask_llm or (lambda prompt: {"provider": "test_provider", "model": "test-model", "response": prompt}),
        save_result=lambda _: None,
    )
    return recommendation.RecommendationService(config, dependencies)


def test_recommend_media_orchestrates_refresh_search_and_llm(runtime_config: RuntimeConfig) -> None:
    calls: list[str] = []

    def fetch(**_: Any) -> list[dict[str, Any]]:
        calls.append("fetch")
        return [{"title": "Fetched"}]

    def search(query: str, **_: Any) -> list[dict[str, Any]]:
        calls.append(f"search:{query}")
        return [{"title": "Outer Wilds", "category": "games"}]

    service = _service(
        runtime_config,
        search=search,
        fetch=fetch,
        profile=lambda: {"likes": ["exploration"]},
        watchlist=lambda **_: [{"title": "Disco Elysium"}],
        ask_llm=lambda prompt: {"provider": "test_provider", "model": "test-model", "response": f"ok: {prompt[:20]}"},
    )
    result = service.recommend(query="cozy mystery", category="games")

    assert calls == ["fetch", "search:cozy mystery"]
    assert result["fetched_count"] == 1
    assert result["candidate_count"] == 1
    assert result["model"] == "test-model"
    assert "ok:" in result["recommendation"]


def test_recommend_media_falls_back_to_recent_candidates(runtime_config: RuntimeConfig) -> None:
    queries: list[str] = []

    def search(query: str, **_: Any) -> list[dict[str, Any]]:
        queries.append(query)
        return [] if query else [{"title": "Recent Item"}]

    result = _service(runtime_config, search=search).recommend(query="very specific", refresh=False)

    assert queries == ["very specific", ""]
    assert result["refreshed"] is False
    assert result["candidates"][0]["title"] == "Recent Item"
    assert result["candidates"][0]["preference_score"] == 0


def test_recommend_media_prefers_exact_category_on_fallback(runtime_config: RuntimeConfig) -> None:
    def search(query: str, **_: Any) -> list[dict[str, Any]]:
        if query:
            return []
        return [
            {"title": "Movie news", "category": "mixed"},
            {"title": "Game news", "category": "games"},
        ]

    result = _service(runtime_config, search=search).recommend(query="vibe", category="games", refresh=False)

    assert [item["title"] for item in result["candidates"]] == ["Game news", "Movie news"]
    assert "Do not invent titles" in result["recommendation"]


def test_recommend_media_expands_vibe_query(runtime_config: RuntimeConfig) -> None:
    queries: list[str] = []

    def search(query: str, **_: Any) -> list[dict[str, Any]]:
        queries.append(query)
        if query == COZY_QUERY:
            return [{"title": "SUMMERHOUSE", "category": "games", "url": "https://example.com/summerhouse"}]
        return []

    result = _service(runtime_config, search=search).recommend(
        query=VIBE_GAME_QUERY,
        category="games",
        refresh=False,
    )

    assert queries[:2] == [VIBE_GAME_QUERY, COZY_QUERY]
    assert result["candidates"][0]["title"] == "SUMMERHOUSE"


def test_recommend_media_ranks_candidates_by_learned_preferences(runtime_config: RuntimeConfig) -> None:
    def search(query: str, **_: Any) -> list[dict[str, Any]]:
        if not query:
            return []
        return [
            {"title": "Weak source", "category": "movies_series", "source": "Noisy", "tags": []},
            {
                "title": "Liked source",
                "category": "movies_series",
                "source": "Trusted",
                "tags": ["Comedy"],
            },
        ]

    def profile() -> dict[str, Any]:
        return {"learned_preferences": {"sources": {"Trusted": 3, "Noisy": -2}, "tags": {"comedy": 2}}}

    result = _service(runtime_config, search=search, profile=profile).recommend(
        query="comedy",
        category="movies_series",
        refresh=False,
    )

    assert [item["title"] for item in result["candidates"]] == ["Liked source", "Weak source"]
    assert result["candidates"][0]["preference_score"] == 5


def test_recommend_media_returns_candidates_when_llm_is_rate_limited(runtime_config: RuntimeConfig) -> None:
    def rate_limited(_: str) -> dict[str, str]:
        raise RuntimeError("LLM rate limit or quota exceeded.")

    service = _service(
        runtime_config,
        search=lambda **_: [{"title": "RSS Candidate", "category": "games"}],
        ask_llm=rate_limited,
    )
    result = service.recommend(query="посоветуй игру", category="games", refresh=False)

    assert result["llm_error"] == "LLM rate limit or quota exceeded."
    assert "RSS Candidate" in result["recommendation"]


def test_recommend_media_prefers_semantic_candidates(runtime_config: RuntimeConfig) -> None:
    naive_queries: list[str] = []

    def search(query: str, **_: Any) -> list[dict[str, Any]]:
        naive_queries.append(query)
        return [{"title": "Naive Candidate"}]

    def semantic(**_: Any) -> list[dict[str, Any]]:
        return [{"title": "Semantic Hit", "category": "games"}]

    result = _service(runtime_config, search=search, semantic=semantic).recommend(
        query="space adventure",
        category="games",
        refresh=False,
    )

    assert [item["title"] for item in result["candidates"]] == ["Semantic Hit"]
    assert naive_queries == []
    assert result["tool_usage"]["semantic_search"] is True


def test_recommend_media_falls_back_to_naive_when_semantic_is_empty(runtime_config: RuntimeConfig) -> None:
    result = _service(
        runtime_config,
        search=lambda **_: [{"title": "Naive Candidate"}],
    ).recommend(query="space adventure", refresh=False)

    assert [item["title"] for item in result["candidates"]] == ["Naive Candidate"]
    assert result["tool_usage"]["semantic_search"] is False


def test_recommend_media_blends_memory_score(runtime_config: RuntimeConfig) -> None:
    def search(query: str, **_: Any) -> list[dict[str, Any]]:
        if not query:
            return []
        return [
            {"title": "Plain", "url": "https://example.com/plain"},
            {"title": "Memorable", "url": "https://example.com/memorable"},
        ]

    def memory_scores(**_: Any) -> dict[str, float]:
        return {"url:https://example.com/memorable": 2.6}

    result = _service(runtime_config, search=search, memory_scores=memory_scores).recommend(
        query="anything",
        refresh=False,
    )

    assert [item["title"] for item in result["candidates"]] == ["Memorable", "Plain"]
    assert result["candidates"][0]["preference_score"] == 3
    assert result["candidates"][0]["preference_memory_score"] == 2.6
    assert result["tool_usage"]["semantic_memory_score"] is True


def test_candidate_kind_adjusts_preference_score(runtime_config: RuntimeConfig) -> None:
    review = recommendation._candidate_with_preference_score(
        {"title": "Rev", "kind": "review"},
        learned={},
        settings=runtime_config.tools,
    )
    news = recommendation._candidate_with_preference_score(
        {"title": "News", "kind": "news"},
        learned={},
        settings=runtime_config.tools,
    )
    noise = recommendation._candidate_with_preference_score(
        {"title": "Deals", "kind": "noise"},
        learned={},
        settings=runtime_config.tools,
    )

    assert review["preference_score"] == 1
    assert "kind:review:+1" in review["preference_reasons"]
    assert news["preference_score"] == 0
    assert noise["preference_score"] == -2


def test_candidate_medium_match_boosts_preference_score(runtime_config: RuntimeConfig) -> None:
    anime = recommendation._candidate_with_preference_score(
        {"title": "Cocoon trailer", "kind": "news", "query_match_medium": "anime"},
        learned={},
        settings=runtime_config.tools,
    )

    assert anime["preference_score"] == runtime_config.tools.recommendation_medium_match_weight
    assert f"medium:anime:+{runtime_config.tools.recommendation_medium_match_weight}" in anime["preference_reasons"]


def test_interleave_by_source_round_robins_candidates() -> None:
    candidates = [
        {"title": "A1", "source": "A"},
        {"title": "A2", "source": "A"},
        {"title": "A3", "source": "A"},
        {"title": "B1", "source": "B"},
    ]
    interleaved = recommendation._interleave_by_source(candidates)
    assert [item["title"] for item in interleaved] == ["A1", "B1", "A2", "A3"]


def test_recommend_media_semantic_failure_is_soft_and_offline(
    runtime_config: RuntimeConfig,
) -> None:
    """RAG с пустым API key откатывается на наивный путь без сети."""
    config = replace(
        runtime_config,
        tools=runtime_config.tools.model_copy(update={"rag_enabled": True}),
    )
    embeddings = create_embeddings(
        create_llm_client(config.llm),
        config.backend.embeddings_cache_file,
        "test",
    )
    semantic = partial(
        rag.semantic_search_data,
        config=config,
        search_cached=lambda **_: [{"title": "Pool Item", "url": "https://example.com/pool"}],
        embeddings=embeddings,
        ask_llm=lambda *_args, **_kwargs: {"response": "unused"},
    )

    def naive_search(query: str, **_: Any) -> list[dict[str, Any]]:
        return [] if not query else [{"title": "Naive Candidate", "url": "https://example.com/naive"}]

    result = _service(config, search=naive_search, semantic=semantic).recommend(
        query="space adventure",
        refresh=False,
    )

    assert [item["title"] for item in result["candidates"]] == ["Naive Candidate"]
    assert result["tool_usage"]["semantic_search"] is False
