import json
from pathlib import Path
from typing import Any

import pytest

from app.settings import BackendSettings
from mcp_tools import feedback, media, memory
from mcp_tools.settings import tool_settings


class RecordingMemoryClient:
    """Записывает вызовы `.add()` Mem0-клиента для проверок."""

    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []

    def add(self, messages: str, *, user_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.add_calls.append({"messages": messages, "user_id": user_id, "metadata": metadata})
        return {"results": []}


def test_feedback_updates_profile_preferences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_file = tmp_path / "profile.json"
    backend_settings = BackendSettings(WATCHQUEST_DATA_DIR=tmp_path)
    fake_memory = RecordingMemoryClient()

    monkeypatch.setattr(feedback, "backend_settings", backend_settings)
    monkeypatch.setattr(tool_settings, "memory_enabled", True)
    monkeypatch.setattr(memory, "_get_memory_client", lambda: fake_memory)

    feedback.save_recommendation_result(
        {
            "id": "rec-1",
            "query": "посоветуй комедию",
            "category": "movies_series",
            "candidates": [
                {
                    "title": "Light Movie",
                    "source": "IndieWire",
                    "tags": ["Comedy", "Evening"],
                }
            ],
            "recommendation": "Try Light Movie.",
            "model": "test-model",
            "provider": "test-provider",
        }
    )

    result = feedback.apply_recommendation_feedback("rec-1", "like")

    assert result["action"] == "like"
    assert result["title"] == "Light Movie"

    profile = json.loads(profile_file.read_text(encoding="utf-8"))
    learned = profile["learned_preferences"]
    assert learned["categories"]["movies_series"] == 1
    assert learned["sources"]["IndieWire"] == 1
    assert learned["tags"]["comedy"] == 1

    assert len(fake_memory.add_calls) == 1
    add_call = fake_memory.add_calls[0]
    assert add_call["metadata"]["weight"] == 1
    assert "Light Movie" in add_call["messages"]


def test_watchlist_feedback_adds_top_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    watchlist_file = tmp_path / "watchlist.json"
    backend_settings = BackendSettings(WATCHQUEST_DATA_DIR=tmp_path)
    fake_memory = RecordingMemoryClient()

    monkeypatch.setattr(feedback, "backend_settings", backend_settings)
    monkeypatch.setattr(media, "backend_settings", backend_settings)
    monkeypatch.setattr(tool_settings, "memory_enabled", True)
    monkeypatch.setattr(memory, "_get_memory_client", lambda: fake_memory)

    feedback.save_recommendation_result(
        {
            "id": "rec-2",
            "query": "посоветуй фильм",
            "category": "movies_series",
            "candidates": [
                {
                    "title": "Movie Candidate",
                    "source": "Manual",
                    "url": "https://example.com/movie",
                }
            ],
            "recommendation": "Try Movie Candidate.",
            "model": "test-model",
            "provider": "test-provider",
        }
    )

    result = feedback.apply_recommendation_feedback("rec-2", "watchlist")

    assert result["watchlist_item"]["title"] == "Movie Candidate"
    assert result["watchlist_item"]["type"] == "movie"

    watchlist = json.loads(watchlist_file.read_text(encoding="utf-8"))
    assert watchlist["items"][0]["url"] == "https://example.com/movie"

    assert len(fake_memory.add_calls) == 1
    assert fake_memory.add_calls[0]["metadata"]["weight"] == 2


def test_feedback_rejects_unknown_recommendation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feedback, "backend_settings", BackendSettings(WATCHQUEST_DATA_DIR=tmp_path))

    with pytest.raises(ValueError, match="Recommendation not found"):
        feedback.apply_recommendation_feedback("missing", "like")
