import json
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


def _client(
    api_key: str = "test-key",
    embedding_model: str = "openai/text-embedding-3-small",
    handler: Any | None = None,
) -> OpenRouterClient:
    return OpenRouterClient(
        settings=OpenRouterSettings(
            api_key=api_key,
            base_url="https://example.com",
            model="openai/gpt-4o-mini",
            embedding_model=embedding_model,
            timeout_seconds=1,
        ),
        transport=httpx.MockTransport(handler) if handler is not None else None,
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


def test_chat_wraps_timeouts() -> None:
    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    with pytest.raises(RuntimeError, match="timed out"):
        _client(handler=raise_timeout).chat(messages)


def test_chat_model_override_replaces_default_model() -> None:
    payloads: list[dict[str, Any]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        payloads.append(dict(json.loads(request.content)))
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=request,
        )

    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    _client(handler=capture).chat(messages, model="google/gemma-4-26b-a4b-it")
    _client(handler=capture).chat(messages)

    assert payloads[0]["model"] == "google/gemma-4-26b-a4b-it"
    assert payloads[1]["model"] == "openai/gpt-4o-mini"


def test_chat_reasoning_effort_from_settings_and_per_call_optout() -> None:
    payloads: list[dict[str, Any]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        payloads.append(dict(json.loads(request.content)))
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=request,
        )

    client = OpenRouterClient(
        settings=OpenRouterSettings(
            api_key="test-key",
            base_url="https://example.com",
            model="openai/gpt-5-mini",
            reasoning_effort="minimal",
            timeout_seconds=1,
        ),
        transport=httpx.MockTransport(capture),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    client.chat(messages)
    client.chat(messages, model="google/gemma-4-26b-a4b-it", reasoning_effort="")

    assert payloads[0]["reasoning"] == {"effort": "minimal"}
    # Пустая строка отключает reasoning: не-reasoning модель (классификатор)
    # иначе тратит весь max_tokens на thinking и возвращает пустой ответ.
    assert "reasoning" not in payloads[1]


def test_embed_fast_fails_without_api_key() -> None:
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        _client(api_key="").embed(["hello"])


def test_embed_fast_fails_without_embedding_model() -> None:
    with pytest.raises(RuntimeError, match="LLM_EMBEDDING_MODEL"):
        _client(embedding_model="").embed(["hello"])


def test_embed_wraps_timeouts() -> None:
    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(RuntimeError, match="timed out"):
        _client(handler=raise_timeout).embed(["hello"])


def test_embed_wraps_missing_endpoint() -> None:
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            json={"error": {"message": "no embeddings endpoint"}},
            request=request,
        )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        _client(handler=not_found).embed(["hello"])


def test_embed_rejects_malformed_response() -> None:
    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"data": [{"index": 0}]},
            request=request,
        )

    with pytest.raises(ValueError, match="Unexpected embeddings response"):
        _client(handler=malformed).embed(["hello"])


def test_embed_orders_vectors_by_index() -> None:
    def out_of_order(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
            request=request,
        )

    assert _client(handler=out_of_order).embed(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_returns_empty_for_empty_input() -> None:
    assert _client().embed([]) == []


def test_extract_embeddings_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="expected 2 data rows"):
        _extract_embeddings({"data": [{"index": 0, "embedding": [0.1]}]}, expected_count=2)


def test_extract_embeddings_rejects_non_contiguous_indexes() -> None:
    payload = {"data": [{"index": 0, "embedding": [0.1]}, {"index": 2, "embedding": [0.2]}]}

    with pytest.raises(ValueError, match="non-contiguous"):
        _extract_embeddings(payload, expected_count=2)
