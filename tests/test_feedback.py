import json
from pathlib import Path

import pytest

from app.settings import BackendSettings
from mcp_tools import feedback, media


def test_feedback_updates_profile_preferences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_file = tmp_path / "profile.json"

    monkeypatch.setattr(feedback, "backend_settings", BackendSettings(WATCHQUEST_DATA_DIR=tmp_path))

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


def test_watchlist_feedback_adds_top_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    watchlist_file = tmp_path / "watchlist.json"
    backend_settings = BackendSettings(WATCHQUEST_DATA_DIR=tmp_path)

    monkeypatch.setattr(feedback, "backend_settings", backend_settings)
    monkeypatch.setattr(media, "backend_settings", backend_settings)

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


def test_feedback_rejects_unknown_recommendation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feedback, "backend_settings", BackendSettings(WATCHQUEST_DATA_DIR=tmp_path))

    with pytest.raises(ValueError, match="Recommendation not found"):
        feedback.apply_recommendation_feedback("missing", "like")
