from typing import Any

import httpx
import pytest

from llm_core.providers.openrouter import (
    OpenRouterClient,
    OpenRouterSettings,
    _extract_content,
    _extract_embeddings,
    _format_http_error,
)
from llm_core.schemas import ChatMessage

EMBEDDINGS_URL = "https://example.com/embeddings"


def _client(api_key: str = "test-key", embedding_model: str = "openai/text-embedding-3-small") -> OpenRouterClient:
    return OpenRouterClient(
        settings=OpenRouterSettings(
            api_key=api_key,
            base_url="https://example.com",
            model="openai/gpt-4o-mini",
            embedding_model=embedding_model,
            timeout_seconds=1,
        )
    )


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


def test_chat_wraps_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx.Client, "post", raise_timeout)
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    with pytest.raises(RuntimeError, match="timed out"):
        _client().chat(messages)


def test_embed_fast_fails_without_api_key(poison_network: list[str]) -> None:
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        _client(api_key="").embed(["hello"])
    assert poison_network == []


def test_embed_fast_fails_without_embedding_model(poison_network: list[str]) -> None:
    with pytest.raises(RuntimeError, match="LLM_EMBEDDING_MODEL"):
        _client(embedding_model="").embed(["hello"])
    assert poison_network == []


def test_embed_wraps_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx.Client, "post", raise_timeout)

    with pytest.raises(RuntimeError, match="timed out"):
        _client().embed(["hello"])


def test_embed_wraps_missing_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            json={"error": {"message": "no embeddings endpoint"}},
            request=httpx.Request("POST", EMBEDDINGS_URL),
        )

    monkeypatch.setattr(httpx.Client, "post", not_found)

    with pytest.raises(RuntimeError, match="HTTP 404"):
        _client().embed(["hello"])


def test_embed_rejects_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def malformed(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"data": [{"index": 0}]},
            request=httpx.Request("POST", EMBEDDINGS_URL),
        )

    monkeypatch.setattr(httpx.Client, "post", malformed)

    with pytest.raises(ValueError, match="Unexpected embeddings response"):
        _client().embed(["hello"])


def test_embed_orders_vectors_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    def out_of_order(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
            request=httpx.Request("POST", EMBEDDINGS_URL),
        )

    monkeypatch.setattr(httpx.Client, "post", out_of_order)

    assert _client().embed(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_returns_empty_for_empty_input(poison_network: list[str]) -> None:
    assert _client().embed([]) == []
    assert poison_network == []


def test_extract_embeddings_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="expected 2 data rows"):
        _extract_embeddings({"data": [{"index": 0, "embedding": [0.1]}]}, expected_count=2)


def test_extract_embeddings_rejects_non_contiguous_indexes() -> None:
    payload = {"data": [{"index": 0, "embedding": [0.1]}, {"index": 2, "embedding": [0.2]}]}

    with pytest.raises(ValueError, match="non-contiguous"):
        _extract_embeddings(payload, expected_count=2)
