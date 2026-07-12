from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from mcp_tools import rag
from mcp_tools.settings import tool_settings


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


@pytest.fixture()
def enable_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_settings, "rag_enabled", True)
    monkeypatch.setattr(rag, "create_embeddings", KeywordEmbeddings)


def test_semantic_search_returns_empty_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    pool_calls: list[str] = []
    monkeypatch.setattr(tool_settings, "rag_enabled", False)

    def record_pool(**_: Any) -> list[dict[str, Any]]:
        pool_calls.append("pool")
        return []

    monkeypatch.setattr(rag, "search_cached_items_data", record_pool)

    assert rag.semantic_search_data(query="space", category="games", limit=5) == []
    assert pool_calls == []


def test_semantic_search_returns_empty_for_blank_query(enable_rag: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "search_cached_items_data", lambda **_: [dict(SPACE_ITEM)])

    assert rag.semantic_search_data(query="   ", category="all", limit=5) == []


def test_semantic_search_returns_empty_for_empty_pool(enable_rag: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "search_cached_items_data", lambda **_: [])

    assert rag.semantic_search_data(query="space", category="all", limit=5) == []


def test_semantic_search_swallows_provider_failure(enable_rag: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_embeddings() -> Embeddings:
        raise RuntimeError("LLM_API_KEY is not configured")

    monkeypatch.setattr(rag, "search_cached_items_data", lambda **_: [dict(SPACE_ITEM)])
    monkeypatch.setattr(rag, "create_embeddings", broken_embeddings)

    assert rag.semantic_search_data(query="space", category="all", limit=5) == []


def test_semantic_search_ranks_similar_candidates_first(
    enable_rag: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_kwargs: dict[str, Any] = {}

    def fake_pool(**kwargs: Any) -> list[dict[str, Any]]:
        pool_kwargs.update(kwargs)
        return [dict(FARM_ITEM), dict(SPACE_ITEM)]

    monkeypatch.setattr(rag, "search_cached_items_data", fake_pool)

    results = rag.semantic_search_data(query="space adventure", category="games", limit=1)

    assert [item["title"] for item in results] == ["Space Odyssey"]
    assert pool_kwargs["query"] == ""
    assert pool_kwargs["category"] == "games"
    assert pool_kwargs["limit"] == 1 * tool_settings.recommendation_fallback_candidate_multiplier


def test_generate_hyde_document_returns_none_on_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_llm(prompt: str, system: str | None = None) -> dict[str, Any]:
        raise RuntimeError("LLM is down")

    monkeypatch.setattr(rag, "ask_llm_data", broken_llm)

    assert rag._generate_hyde_document("вайбовая игра", "games") is None


def test_semantic_search_uses_raw_query_when_hyde_disabled(
    enable_rag: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def poison_llm(prompt: str, system: str | None = None) -> dict[str, Any]:
        raise AssertionError("ask_llm_data must not be called when HyDE is disabled")

    monkeypatch.setattr(tool_settings, "hyde_enabled", False)
    monkeypatch.setattr(rag, "ask_llm_data", poison_llm)
    monkeypatch.setattr(rag, "search_cached_items_data", lambda **_: [dict(FARM_ITEM), dict(SPACE_ITEM)])

    results = rag.semantic_search_data(query="space adventure", category="all", limit=1)

    assert [item["title"] for item in results] == ["Space Odyssey"]


def test_semantic_search_blends_hyde_scores_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_settings, "rag_enabled", True)
    monkeypatch.setattr(tool_settings, "hyde_enabled", True)
    monkeypatch.setattr(rag, "create_embeddings", BlendEmbeddings)
    monkeypatch.setattr(
        rag,
        "ask_llm_data",
        lambda prompt, system=None: {"response": "A cozy farm life simulator with relaxing chores."},
    )
    monkeypatch.setattr(
        rag,
        "search_cached_items_data",
        lambda **_: [dict(SPACE_ITEM), dict(FARM_ITEM), dict(CITY_ITEM)],
    )

    results = rag.semantic_search_data(query="space adventure", category="all", limit=2)

    assert {item["title"] for item in results} == {"Space Odyssey", "Farm Story"}


def test_semantic_search_falls_back_when_hyde_generation_fails(
    enable_rag: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_llm(prompt: str, system: str | None = None) -> dict[str, Any]:
        raise RuntimeError("LLM is down")

    monkeypatch.setattr(rag, "ask_llm_data", broken_llm)
    monkeypatch.setattr(rag, "search_cached_items_data", lambda **_: [dict(FARM_ITEM), dict(SPACE_ITEM)])

    monkeypatch.setattr(tool_settings, "hyde_enabled", True)
    with_hyde_failure = rag.semantic_search_data(query="space adventure", category="all", limit=2)

    monkeypatch.setattr(tool_settings, "hyde_enabled", False)
    without_hyde = rag.semantic_search_data(query="space adventure", category="all", limit=2)

    assert with_hyde_failure == without_hyde
    assert [item["title"] for item in with_hyde_failure] == ["Space Odyssey", "Farm Story"]


def test_score_documents_boosts_reviews_and_penalizes_noise() -> None:
    items_by_key = {
        "url:review": {"title": "Review", "kind": "review", "category": "games"},
        "url:news": {"title": "News", "kind": "news", "category": "games"},
        "url:noise": {"title": "Deals", "kind": "noise", "category": "games"},
    }
    scored_documents = [
        (Document(page_content="", metadata={"candidate_key": key}), 0.5) for key in items_by_key
    ]

    scored = rag._score_documents(scored_documents, items_by_key, category="all")

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
