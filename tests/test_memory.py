from typing import Any

import pytest

from domain.models import FeedbackAction, RecommendationFeedback
from mcp_tools import memory
from mcp_tools.settings import tool_settings


class FakeMemoryClient:
    """Записывает вызовы Mem0-клиента и возвращает канонические результаты."""

    def __init__(self, search_results: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self._search_results = search_results or {}

    def add(self, messages: str, *, user_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.add_calls.append({"messages": messages, "user_id": user_id, "metadata": metadata})
        return {"results": []}

    def search(self, query: str, *, user_id: str, limit: int) -> dict[str, Any]:
        self.search_calls.append({"query": query, "user_id": user_id, "limit": limit})
        results = next(
            (items for keyword, items in self._search_results.items() if keyword in query.lower()),
            [],
        )
        return {"results": results}


class BrokenMemoryClient:
    """Кидает исключение на любой вызов Mem0-клиента."""

    def add(self, messages: str, *, user_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        raise RuntimeError("memgraph is down")

    def search(self, query: str, *, user_id: str, limit: int) -> dict[str, Any]:
        raise RuntimeError("memgraph is down")


SPACE_CANDIDATE = {
    "title": "Space Odyssey",
    "summary": "Exploration in deep space",
    "tags": ["space"],
    "url": "https://example.com/space",
}
FARM_CANDIDATE = {
    "title": "Farm Story",
    "summary": "Cozy farming simulator",
    "tags": ["farm"],
    "url": "https://example.com/farm",
}


def _feedback(action: FeedbackAction = "like") -> RecommendationFeedback:
    return RecommendationFeedback(
        recommendation_id="rec-1",
        action=action,
        query="космическая игра",
        category="games",
        title="Space Odyssey",
        source="Trusted",
    )


def _forbidden_client() -> Any:
    raise AssertionError("Mem0 client must not be created when memory is disabled")


def test_record_preference_note_adds_memory_with_signed_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMemoryClient()
    monkeypatch.setattr(tool_settings, "memory_enabled", True)
    monkeypatch.setattr(memory, "_get_memory_client", lambda: fake)

    memory.record_preference_note_data(feedback=_feedback("block_similar"), candidate=dict(SPACE_CANDIDATE))

    assert len(fake.add_calls) == 1
    call = fake.add_calls[0]
    assert "Space Odyssey" in call["messages"]
    assert call["user_id"] == tool_settings.mem0_user_id
    assert call["metadata"]["weight"] == -3
    assert call["metadata"]["category"] == "games"
    assert call["metadata"]["action"] == "block_similar"


def test_record_preference_note_disabled_never_touches_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_settings, "memory_enabled", False)
    monkeypatch.setattr(memory, "_get_memory_client", _forbidden_client)

    memory.record_preference_note_data(feedback=_feedback("like"), candidate=dict(SPACE_CANDIDATE))


def test_record_preference_note_swallows_add_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_settings, "memory_enabled", True)
    monkeypatch.setattr(memory, "_get_memory_client", lambda: BrokenMemoryClient())

    memory.record_preference_note_data(feedback=_feedback("like"), candidate=dict(SPACE_CANDIDATE))


def test_semantic_memory_scores_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_settings, "memory_enabled", False)
    monkeypatch.setattr(memory, "_get_memory_client", _forbidden_client)

    assert memory.semantic_memory_scores_data(candidates=[dict(SPACE_CANDIDATE)], top_k=5) == {}


def test_semantic_memory_scores_empty_without_memories(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMemoryClient(search_results={})
    monkeypatch.setattr(tool_settings, "memory_enabled", True)
    monkeypatch.setattr(memory, "_get_memory_client", lambda: fake)

    assert memory.semantic_memory_scores_data(candidates=[dict(SPACE_CANDIDATE)], top_k=5) == {}
    assert len(fake.search_calls) == 1


def test_semantic_memory_scores_blend_signed_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMemoryClient(
        search_results={
            "space": [
                {"memory": "User likes space exploration", "score": 0.9, "metadata": {"weight": 2}},
                {"memory": "User likes sci-fi", "score": 0.5, "metadata": {"weight": 1}},
            ],
            "farm": [
                {"memory": "User blocks farming sims", "score": 0.8, "metadata": {"weight": -3}},
            ],
        }
    )
    monkeypatch.setattr(tool_settings, "memory_enabled", True)
    monkeypatch.setattr(memory, "_get_memory_client", lambda: fake)

    scores = memory.semantic_memory_scores_data(
        candidates=[dict(SPACE_CANDIDATE), dict(FARM_CANDIDATE)],
        top_k=5,
    )

    assert scores["url:https://example.com/space"] == pytest.approx((0.9 * 2 + 0.5 * 1) / 2)
    assert scores["url:https://example.com/space"] > 0
    assert scores["url:https://example.com/farm"] == pytest.approx(0.8 * -3)
    assert scores["url:https://example.com/farm"] < 0
    assert all(call["limit"] == 5 for call in fake.search_calls)
    assert all(call["user_id"] == tool_settings.mem0_user_id for call in fake.search_calls)


def test_semantic_memory_scores_swallow_search_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_settings, "memory_enabled", True)
    monkeypatch.setattr(memory, "_get_memory_client", lambda: BrokenMemoryClient())

    assert memory.semantic_memory_scores_data(candidates=[dict(SPACE_CANDIDATE)], top_k=5) == {}
