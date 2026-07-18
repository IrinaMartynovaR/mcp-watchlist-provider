import json
from datetime import UTC, datetime, timedelta

from app.config import RuntimeConfig
from app.telegram_bot import (
    format_recommendation_response,
    format_watchlist_response,
    infer_category,
    is_watchlist_request,
    recommendation_feedback_keyboard,
    should_refresh_rss_cache,
    split_telegram_text,
)

RU_GAME_QUERY = (
    "\u043f\u043e\u0441\u043e\u0432\u0435\u0442\u0443\u0439 "
    "\u0432\u0430\u0439\u0431\u043e\u0432\u0443\u044e "
    "\u0438\u0433\u0440\u0443"
)
RU_MOVIE_QUERY = "\u0445\u043e\u0447\u0443 \u0444\u0438\u043b\u044c\u043c \u043d\u0430 \u0432\u0435\u0447\u0435\u0440"
RU_SERIES_QUERY = "\u043f\u043e\u0441\u043e\u0432\u0435\u0442\u0443\u0439 \u0441\u0435\u0440\u0438\u0430\u043b"
RU_TITLE = "\u0423\u044e\u0442\u043d\u0430\u044f \u0438\u0433\u0440\u0430 SUMMERHOUSE"


def test_infer_category_detects_games() -> None:
    assert infer_category(RU_GAME_QUERY) == "games"
    assert infer_category("cozy game for switch") == "games"


def test_infer_category_detects_movies_and_series() -> None:
    assert infer_category(RU_MOVIE_QUERY) == "movies_series"
    assert infer_category(RU_SERIES_QUERY) == "series"


def test_format_recommendation_response_includes_candidates(runtime_config: RuntimeConfig) -> None:
    result = {
        "recommendation": "Try SUMMERHOUSE.",
        "candidates": [
            {
                "title": RU_TITLE,
                "source": "StopGame",
                "url": "https://example.com/summerhouse",
            }
        ],
    }

    formatted = format_recommendation_response(result, runtime_config.backend)

    assert "Try SUMMERHOUSE." in formatted
    assert "RSS" in formatted
    assert f'<a href="https://example.com/summerhouse">{RU_TITLE}</a>' in formatted
    assert "<i>StopGame</i>" in formatted


def test_format_recommendation_response_escapes_html_in_candidates(runtime_config: RuntimeConfig) -> None:
    result = {
        "recommendation": "<b>Take Tom & Jerry.</b>",
        "candidates": [
            {
                "title": "Tom & Jerry <Deluxe>",
                "source": "R&D <feed>",
                "url": "https://example.com/tom?a=1&b=2",
            }
        ],
    }

    formatted = format_recommendation_response(result, runtime_config.backend)

    assert "<b>Take Tom & Jerry.</b>" in formatted
    assert 'href="https://example.com/tom?a=1&amp;b=2"' in formatted
    assert "Tom &amp; Jerry &lt;Deluxe&gt;" in formatted
    assert "<i>R&amp;D &lt;feed&gt;</i>" in formatted


def test_split_telegram_text_keeps_chunks_under_limit(runtime_config: RuntimeConfig) -> None:
    chunks = split_telegram_text("word\n\n" * 100, runtime_config.backend, limit=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_watchlist_request_detects_ru_and_en_text(runtime_config: RuntimeConfig) -> None:
    assert is_watchlist_request("покажи мой вотчлист", runtime_config.backend)
    assert is_watchlist_request("show watchlist", runtime_config.backend)


def test_format_watchlist_response_handles_items() -> None:
    formatted = format_watchlist_response(
        [
            {
                "title": "Outer Wilds",
                "type": "game",
                "status": "planned",
                "source": "manual",
            }
        ]
    )

    assert "Outer Wilds" in formatted
    assert "manual" in formatted


def test_recommendation_feedback_keyboard_contains_expected_callbacks(runtime_config: RuntimeConfig) -> None:
    keyboard = recommendation_feedback_keyboard("rec-1", runtime_config.backend)
    callback_data = [button.callback_data for row in keyboard.inline_keyboard for button in row]

    assert "feedback:like:rec-1" in callback_data
    assert "feedback:watchlist:rec-1" in callback_data


def test_should_refresh_rss_cache_respects_ttl(runtime_config: RuntimeConfig) -> None:
    cached_at = datetime(2026, 1, 1, tzinfo=UTC)
    runtime_config.backend.cache_file.write_text(
        json.dumps({"items": [], "refreshed_at_by_category": {"all": cached_at.isoformat()}}),
        encoding="utf-8",
    )

    assert not should_refresh_rss_cache(runtime_config.backend, "games", cached_at + timedelta(minutes=5))
    assert should_refresh_rss_cache(runtime_config.backend, "games", cached_at + timedelta(minutes=31))


def test_should_refresh_rss_cache_rejects_other_scoped_category(runtime_config: RuntimeConfig) -> None:
    cached_at = datetime(2026, 1, 1, tzinfo=UTC)
    runtime_config.backend.cache_file.write_text(
        json.dumps({"items": [], "refreshed_at_by_category": {"games": cached_at.isoformat()}}),
        encoding="utf-8",
    )

    assert should_refresh_rss_cache(runtime_config.backend, "series", cached_at + timedelta(minutes=5))
