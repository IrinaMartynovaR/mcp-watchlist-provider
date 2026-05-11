from __future__ import annotations

from watchquest.telegram_bot import format_recommendation_response, infer_category, split_telegram_text


def test_infer_category_detects_games() -> None:
    assert infer_category("посоветуй вайбовую игру") == "games"
    assert infer_category("cozy game for switch") == "games"


def test_infer_category_detects_movies_and_series() -> None:
    assert infer_category("хочу фильм на вечер") == "movies_series"
    assert infer_category("посоветуй сериал") == "series"


def test_format_recommendation_response_includes_candidates() -> None:
    result = {
        "recommendation": "Попробуй SUMMERHOUSE.",
        "candidates": [
            {
                "title": "Уютная игра SUMMERHOUSE",
                "source": "StopGame",
                "url": "https://example.com/summerhouse",
            }
        ],
    }

    formatted = format_recommendation_response(result)

    assert "Попробуй SUMMERHOUSE." in formatted
    assert "Кандидаты из RSS" in formatted
    assert "https://example.com/summerhouse" in formatted


def test_split_telegram_text_keeps_chunks_under_limit() -> None:
    chunks = split_telegram_text("word\n\n" * 100, limit=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
