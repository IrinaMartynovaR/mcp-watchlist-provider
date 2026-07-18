from dataclasses import replace
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import RuntimeConfig
from mcp_tools import rag


class KeywordEmbeddings(Embeddings):
    """Возвращает детерминированные векторы по ключевому слову в тексте."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "space" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


SPACE_ITEM = {
    "title": "Space Odyssey",
    "summary": "Exploration in deep space",
    "tags": ["space"],
    "url": "https://example.com/space",
}
FARM_ITEM = {
    "title": "Farm Story",
    "summary": "Cozy farming simulator",
    "tags": ["farm"],
    "url": "https://example.com/farm",
}
CITY_ITEM = {
    "title": "City Lights",
    "summary": "Neon urban noir thriller",
    "tags": ["noir"],
    "url": "https://example.com/city",
}


class BlendEmbeddings(Embeddings):
    """Разводит сырой запрос и HyDE-текст по разным осям для merge-теста."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "space" in lowered:
            return [1.0, 0.0]
        if "farm" in lowered:
            return [0.0, 1.0]
        return [0.8, 0.6]


def _rag_config(config: RuntimeConfig, *, hyde: bool = False) -> RuntimeConfig:
    return replace(
        config,
        tools=config.tools.model_copy(update={"rag_enabled": True, "hyde_enabled": hyde}),
    )


def _unused_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise AssertionError("LLM must not be called")


def test_semantic_search_returns_empty_when_disabled(runtime_config: RuntimeConfig) -> None:
    pool_calls: list[str] = []

    def record_pool(**_: Any) -> list[dict[str, Any]]:
        pool_calls.append("pool")
        return []

    assert (
        rag.semantic_search_data(
            query="space",
            config=runtime_config,
            search_cached=record_pool,
            embeddings=KeywordEmbeddings(),
            ask_llm=_unused_llm,
            category="games",
            limit=5,
        )
        == []
    )
    assert pool_calls == []


def test_semantic_search_returns_empty_for_blank_query(runtime_config: RuntimeConfig) -> None:
    assert (
        rag.semantic_search_data(
            query="   ",
            config=_rag_config(runtime_config),
            search_cached=lambda **_: [dict(SPACE_ITEM)],
            embeddings=KeywordEmbeddings(),
            ask_llm=_unused_llm,
            category="all",
            limit=5,
        )
        == []
    )


def test_semantic_search_returns_empty_for_empty_pool(runtime_config: RuntimeConfig) -> None:
    assert (
        rag.semantic_search_data(
            query="space",
            config=_rag_config(runtime_config),
            search_cached=lambda **_: [],
            embeddings=KeywordEmbeddings(),
            ask_llm=_unused_llm,
            category="all",
            limit=5,
        )
        == []
    )


def test_semantic_search_swallows_provider_failure(runtime_config: RuntimeConfig) -> None:
    class BrokenEmbeddings(KeywordEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("LLM_API_KEY is not configured")

    assert (
        rag.semantic_search_data(
            query="space",
            config=_rag_config(runtime_config),
            search_cached=lambda **_: [dict(SPACE_ITEM)],
            embeddings=BrokenEmbeddings(),
            ask_llm=_unused_llm,
            category="all",
            limit=5,
        )
        == []
    )


def test_semantic_search_ranks_similar_candidates_first(
    runtime_config: RuntimeConfig,
) -> None:
    pool_kwargs: dict[str, Any] = {}

    def fake_pool(**kwargs: Any) -> list[dict[str, Any]]:
        pool_kwargs.update(kwargs)
        return [dict(FARM_ITEM), dict(SPACE_ITEM)]

    config = _rag_config(runtime_config)
    results = rag.semantic_search_data(
        query="space adventure",
        config=config,
        search_cached=fake_pool,
        embeddings=KeywordEmbeddings(),
        ask_llm=_unused_llm,
        category="games",
        limit=1,
    )

    assert [item["title"] for item in results] == ["Space Odyssey"]
    assert pool_kwargs["query"] == ""
    assert pool_kwargs["category"] == "games"
    # Пул семантического поиска не сжимается до limit * multiplier:
    # индексируется весь свежий кеш в пределах rag_pool_limit.
    assert pool_kwargs["limit"] == config.tools.rag_pool_limit


def test_generate_hyde_document_returns_none_on_llm_failure(runtime_config: RuntimeConfig) -> None:
    def broken_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("LLM is down")

    assert rag._generate_hyde_document("вайбовая игра", "games", runtime_config, broken_llm) is None


def test_semantic_search_uses_raw_query_when_hyde_disabled(
    runtime_config: RuntimeConfig,
) -> None:
    results = rag.semantic_search_data(
        query="space adventure",
        config=_rag_config(runtime_config),
        search_cached=lambda **_: [dict(FARM_ITEM), dict(SPACE_ITEM)],
        embeddings=KeywordEmbeddings(),
        ask_llm=_unused_llm,
        category="all",
        limit=1,
    )

    assert [item["title"] for item in results] == ["Space Odyssey"]


def test_semantic_search_blends_hyde_scores_when_enabled(runtime_config: RuntimeConfig) -> None:
    results = rag.semantic_search_data(
        query="space adventure",
        config=_rag_config(runtime_config, hyde=True),
        search_cached=lambda **_: [dict(SPACE_ITEM), dict(FARM_ITEM), dict(CITY_ITEM)],
        embeddings=BlendEmbeddings(),
        ask_llm=lambda *_, **__: {"response": "A cozy farm life simulator with relaxing chores."},
        category="all",
        limit=2,
    )

    assert {item["title"] for item in results} == {"Space Odyssey", "Farm Story"}


def test_semantic_search_falls_back_when_hyde_generation_fails(
    runtime_config: RuntimeConfig,
) -> None:
    def broken_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("LLM is down")

    def search(**_: Any) -> list[dict[str, Any]]:
        return [dict(FARM_ITEM), dict(SPACE_ITEM)]

    with_hyde_failure = rag.semantic_search_data(
        query="space adventure",
        config=_rag_config(runtime_config, hyde=True),
        search_cached=search,
        embeddings=KeywordEmbeddings(),
        ask_llm=broken_llm,
        category="all",
        limit=2,
    )
    without_hyde = rag.semantic_search_data(
        query="space adventure",
        config=_rag_config(runtime_config),
        search_cached=search,
        embeddings=KeywordEmbeddings(),
        ask_llm=_unused_llm,
        category="all",
        limit=2,
    )

    assert with_hyde_failure == without_hyde
    assert [item["title"] for item in with_hyde_failure] == ["Space Odyssey", "Farm Story"]


def test_semantic_search_flags_and_boosts_medium_matches(runtime_config: RuntimeConfig) -> None:
    anime_item = {
        "title": "Cocoon trailer",
        "summary": "Wartime drama",
        "source": "Anime News Network",
        "medium": "anime",
        "url": "https://example.com/cocoon",
    }
    drama_item = {
        "title": "Korean drama hit",
        "summary": "A palace intrigue everyone loves",
        "source": "Variety",
        "medium": "series",
        "url": "https://example.com/palace",
    }

    class FlatEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    results = rag.semantic_search_data(
        query="аниме драма",
        config=_rag_config(runtime_config),
        search_cached=lambda **_: [dict(drama_item), dict(anime_item)],
        embeddings=FlatEmbeddings(),
        ask_llm=_unused_llm,
        category="all",
        limit=2,
        medium="anime",
    )

    assert results[0]["title"] == "Cocoon trailer"
    assert results[0]["query_match_medium"] == "anime"
    assert "query_match_medium" not in results[1]
    assert results[0]["semantic_score"] > results[1]["semantic_score"]


def test_score_documents_boosts_reviews_and_penalizes_noise(runtime_config: RuntimeConfig) -> None:
    items_by_key = {
        "url:review": {"title": "Review", "kind": "review", "category": "games"},
        "url:news": {"title": "News", "kind": "news", "category": "games"},
        "url:noise": {"title": "Deals", "kind": "noise", "category": "games"},
    }
    scored_documents = [(Document(page_content="", metadata={"candidate_key": key}), 0.5) for key in items_by_key]

    scored = rag._score_documents(scored_documents, items_by_key, category="all", config=runtime_config)

    assert scored["url:review"][1] == pytest.approx(0.55)
    assert scored["url:news"][1] == pytest.approx(0.5)
    assert scored["url:noise"][1] == pytest.approx(0.3)


def test_diversify_by_source_caps_dominant_source() -> None:
    ranked = [
        {"title": "A1", "source": "A"},
        {"title": "A2", "source": "A"},
        {"title": "A3", "source": "A"},
        {"title": "A4", "source": "A"},
        {"title": "B1", "source": "B"},
        {"title": "C1", "source": "C"},
    ]

    result = rag._diversify_by_source(ranked, limit=4)

    assert len(result) == 4
    titles = [item["title"] for item in result]
    assert "B1" in titles
    assert "C1" in titles
    assert sum(1 for item in result if item["source"] == "A") <= 2


def test_diversify_by_source_backfills_when_sources_scarce() -> None:
    ranked = [
        {"title": "A1", "source": "A"},
        {"title": "A2", "source": "A"},
        {"title": "A3", "source": "A"},
    ]

    result = rag._diversify_by_source(ranked, limit=3)

    assert [item["title"] for item in result] == ["A1", "A2", "A3"]


def test_diversify_by_source_preserves_relevance_order_within_cap() -> None:
    ranked = [
        {"title": "A1", "source": "A"},
        {"title": "B1", "source": "B"},
        {"title": "A2", "source": "A"},
    ]

    result = rag._diversify_by_source(ranked, limit=2)

    assert [item["title"] for item in result] == ["A1", "B1"]
