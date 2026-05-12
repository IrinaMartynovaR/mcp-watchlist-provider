from typing import Any

import httpx
import pytest

from llm_core.providers.zai_glm import ZAIGLMClient, ZAIGLMSettings, _extract_content, _format_http_error
from llm_core.schemas import ChatMessage


def test_extract_content_from_chat_completion_response() -> None:
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


def test_extract_content_rejects_missing_choices() -> None:
    with pytest.raises(ValueError, match="missing choices"):
        _extract_content({"choices": []})


def test_extract_content_rejects_empty_final_content() -> None:
    with pytest.raises(ValueError, match="empty final content"):
        _extract_content({"choices": [{"finish_reason": "length", "message": {"content": ""}}]})


def test_format_http_error_for_rate_limit() -> None:
    response = httpx.Response(
        status_code=429,
        json={"error": {"message": "quota exceeded"}},
        request=httpx.Request("POST", "https://example.com/chat/completions"),
    )

    assert _format_http_error(response) == "LLM rate limit or quota exceeded. quota exceeded"


def test_provider_client_wraps_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx.Client, "post", raise_timeout)
    client = ZAIGLMClient(
        settings=ZAIGLMSettings(
            api_key="test-key",
            base_url="https://example.com",
            model="test-model",
            timeout_seconds=1,
        )
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    with pytest.raises(RuntimeError, match="timed out"):
        client.chat(messages)


