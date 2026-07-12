from typing import Any

import pytest

from domain.models import Category
from mcp_tools import myshows_import
from mcp_tools.myshows_import import import_myshows_history_data


class FakeMyShowsClient:
    """Возвращает канонические данные MyShows без сети."""

    def __init__(self, movies: list[dict[str, Any]], shows: list[dict[str, Any]]) -> None:
        self.movies = movies
        self.shows = shows

    def iter_all_watched_movies(self) -> list[dict[str, Any]]:
        return self.movies

    def get_shows(self) -> list[dict[str, Any]]:
        return self.shows


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    movies: list[dict[str, Any]] | None = None,
    shows: list[dict[str, Any]] | None = None,
) -> None:
    client = FakeMyShowsClient(movies or [], shows or [])
    monkeypatch.setattr(myshows_import, "MyShowsClient", lambda settings: client)


def _capture_notes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []

    def record(
        text: str,
        weight: int,
        category: Category,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        notes.append({"text": text, "weight": weight, "category": category, "extra_metadata": extra_metadata})

    monkeypatch.setattr(myshows_import, "record_import_note_data", record)
    return notes


def _poison_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    def poisoned(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("record_import_note_data must not be called in dry_run")

    monkeypatch.setattr(myshows_import, "record_import_note_data", poisoned)


def test_movie_rating_buckets_map_to_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(
        monkeypatch,
        movies=[
            {"title": "Loved", "rating": 9},
            {"title": "Boundary High", "rating": 8},
            {"title": "Fine", "rating": 7},
            {"title": "Boundary Mid", "rating": 6},
            {"title": "Boundary Low", "rating": 5},
            {"title": "Awful", "rating": 2},
        ],
    )
    notes = _capture_notes(monkeypatch)

    summary = import_myshows_history_data(dry_run=False)

    assert [(note["text"], note["weight"]) for note in notes] == [
        ("Loved", 1.5),
        ("Boundary High", 1.5),
        ("Fine", 0.5),
        ("Boundary Mid", 0.5),
        ("Boundary Low", -0.5),
        ("Awful", -0.5),
    ]
    assert all(note["category"] == "movies_series" for note in notes)
    assert all(note["extra_metadata"] == {"source": "myshows_import"} for note in notes)
    assert summary == {"movies_processed": 6, "shows_processed": 0, "notes_recorded": 6, "skipped": 0}


def test_unrated_movie_is_skipped_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, movies=[{"title": "Unrated"}, {"title": "Rated", "rating": 10}])
    notes = _capture_notes(monkeypatch)

    summary = import_myshows_history_data(dry_run=False)

    assert [note["text"] for note in notes] == ["Rated"]
    assert summary == {"movies_processed": 2, "shows_processed": 0, "notes_recorded": 1, "skipped": 1}


def test_show_statuses_map_to_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(
        monkeypatch,
        shows=[
            {"title": "Watching", "watchStatus": "watching"},
            {"title": "Finished", "watchStatus": "finished"},
            {"title": "Dropped", "watchStatus": "cancelled"},
            {"title": "Planned", "watchStatus": "later"},
            {"title": "No Status"},
        ],
    )
    notes = _capture_notes(monkeypatch)

    summary = import_myshows_history_data(dry_run=False)

    assert [(note["text"], note["weight"]) for note in notes] == [
        ("Watching", 1.0),
        ("Finished", 1.0),
        ("Dropped", -1.0),
    ]
    assert all(note["category"] == "series" for note in notes)
    assert summary == {"movies_processed": 0, "shows_processed": 5, "notes_recorded": 3, "skipped": 2}


def test_note_text_includes_genres_and_nested_show_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(
        monkeypatch,
        movies=[{"title": "Dune", "rating": 10, "genres": ["Sci-Fi", "Drama"]}],
        shows=[{"watchStatus": "watching", "show": {"title": "Nested", "genres": ["Thriller"]}}],
    )
    notes = _capture_notes(monkeypatch)

    import_myshows_history_data(dry_run=False)

    assert [note["text"] for note in notes] == ["Dune (Sci-Fi, Drama)", "Nested (Thriller)"]


def test_item_without_title_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch, movies=[{"rating": 9}])
    notes = _capture_notes(monkeypatch)

    summary = import_myshows_history_data(dry_run=False)

    assert notes == []
    assert summary == {"movies_processed": 1, "shows_processed": 0, "notes_recorded": 0, "skipped": 1}


def test_dry_run_counts_without_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(
        monkeypatch,
        movies=[{"title": "Loved", "rating": 9}, {"title": "Unrated"}],
        shows=[{"title": "Dropped", "watchStatus": "cancelled"}],
    )
    _poison_recording(monkeypatch)

    summary = import_myshows_history_data(dry_run=True)

    assert summary == {"movies_processed": 2, "shows_processed": 1, "notes_recorded": 2, "skipped": 1}


def test_show_rating_overrides_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(
        monkeypatch,
        shows=[
            {"title": "Loved", "watchStatus": "cancelled", "rating": 5},
            {"title": "Hated", "watchStatus": "finished", "rating": 1},
            {"title": "Neutral", "watchStatus": "finished", "rating": 3},
            {"title": "Unrated", "watchStatus": "finished", "rating": 0},
        ],
    )
    notes = _capture_notes(monkeypatch)

    import_myshows_history_data(dry_run=False)

    assert [(note["text"], note["weight"]) for note in notes] == [
        ("Loved", 1.5),
        ("Hated", -1.0),
        ("Neutral", 1.0),
        ("Unrated", 1.0),
    ]


def test_note_text_handles_genre_objects_and_original_title(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(
        monkeypatch,
        shows=[
            {
                "watchStatus": "finished",
                "show": {
                    "title": "Дом дракона",
                    "titleOriginal": "House of the Dragon",
                    "genres": [{"id": 12, "title": "Фэнтези", "alias": "fantasy"}, {"id": 4, "title": "Драма"}],
                },
            }
        ],
    )
    notes = _capture_notes(monkeypatch)

    import_myshows_history_data(dry_run=False)

    assert notes[0]["text"] == "Дом дракона / House of the Dragon (Фэнтези, Драма)"
