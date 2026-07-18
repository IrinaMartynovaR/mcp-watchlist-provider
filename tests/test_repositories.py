from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from domain.models import RecommendationRecord, WatchlistItem
from domain.repositories.profile import ProfileRepository
from domain.repositories.recommendations import RecommendationRepository
from domain.repositories.watchlist import WatchlistRepository


def test_profile_repository_preserves_concurrent_learning(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path / "profile.json")

    def learn(_: int) -> None:
        repository.learn(category="games", source="Example", tags=["cozy"], weight=1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(learn, range(50)))

    learned = repository.get()["learned_preferences"]
    assert learned["categories"]["games"] == 50
    assert learned["sources"]["Example"] == 50
    assert learned["tags"]["cozy"] == 50


def test_watchlist_repository_preserves_concurrent_additions(tmp_path: Path) -> None:
    repository = WatchlistRepository(tmp_path / "watchlist.json")

    def add(index: int) -> None:
        repository.add(WatchlistItem(title=f"Title {index}", type="movie"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(add, range(50)))

    assert len(repository.list(media_type="all", status="all")) == 50


def test_recommendation_repository_replaces_record_by_id(tmp_path: Path) -> None:
    repository = RecommendationRepository(tmp_path / "recommendations.json")
    original = RecommendationRecord(
        id="rec-1",
        query="first",
        recommendation="Old",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    updated = original.model_copy(update={"query": "second", "recommendation": "New"})

    repository.save(original)
    repository.save(updated)

    stored = repository.get("rec-1")
    assert stored is not None
    assert stored.query == "second"
    assert stored.recommendation == "New"
