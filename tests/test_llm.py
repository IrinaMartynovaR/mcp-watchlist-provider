from __future__ import annotations

from typing import Any

import httpx
import pytest

from watchquest.llm.zai import ChatMessage, ZAIClient, ZAISettings, _extract_content, _format_http_error
from watchquest.tools.llm import _candidate_items, _recommendation_prompt


def test_extract_content_from_zai_response() -> None:
    data: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Try Outer Wilds.",
                }
            }
        ]
    }

    assert _extract_content(data) == "Try Outer Wilds."


def test_extract_content_rejects_empty_response() -> None:
    with pytest.raises(ValueError, match="missing choices"):
        _extract_content({"choices": []})


def test_format_http_error_for_rate_limit() -> None:
    response = httpx.Response(
        status_code=429,
        json={"error": {"message": "quota exceeded"}},
        request=httpx.Request("POST", "https://example.com/chat/completions"),
    )

    assert _format_http_error(response) == "Z.AI rate limit or quota exceeded. quota exceeded"


def test_zai_client_wraps_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx.Client, "post", raise_timeout)
    client = ZAIClient(settings=ZAISettings(api_key="test-key"))
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    with pytest.raises(RuntimeError, match="timed out"):
        client.chat(messages)


def test_candidate_items_filters_by_category() -> None:
    items = [
        {"title": "A", "category": "games"},
        {"title": "B", "category": "movies"},
        {"title": "C", "category": "mixed"},
    ]

    assert [item["title"] for item in _candidate_items(items, category="games", limit=3)] == ["A", "C"]


def test_recommendation_prompt_contains_context() -> None:
    prompt = _recommendation_prompt(
        query="cozy RPG",
        profile={"likes": ["story-rich games"]},
        watchlist=[{"title": "Disco Elysium"}],
        candidates=[{"title": "Outer Wilds", "category": "games"}],
    )

    assert "cozy RPG" in prompt
    assert "Disco Elysium" in prompt
    assert "Outer Wilds" in prompt
