from app.telegram_bot import format_recommendation_response, infer_category, split_telegram_text

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


def test_format_recommendation_response_includes_candidates() -> None:
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

    formatted = format_recommendation_response(result)

    assert "Try SUMMERHOUSE." in formatted
    assert "RSS" in formatted
    assert "https://example.com/summerhouse" in formatted


def test_split_telegram_text_keeps_chunks_under_limit() -> None:
    chunks = split_telegram_text("word\n\n" * 100, limit=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
