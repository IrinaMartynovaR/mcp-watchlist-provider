from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

import httpx

from watchquest.config import ZAI_API_KEY, ZAI_BASE_URL, ZAI_MODEL, ZAI_TIMEOUT_SECONDS

ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: ChatRole
    content: str


class ChatCompletionRequest(TypedDict):
    model: str
    messages: list[ChatMessage]
    temperature: float
    stream: bool
    max_tokens: NotRequired[int]


@dataclass(frozen=True)
class ZAISettings:
    api_key: str = ZAI_API_KEY
    base_url: str = ZAI_BASE_URL
    model: str = ZAI_MODEL
    timeout_seconds: float = ZAI_TIMEOUT_SECONDS
    temperature: float = 0.7


class ZAIClient:
    def __init__(self, settings: ZAISettings | None = None) -> None:
        self.settings = settings or ZAISettings()

    def chat(self, messages: list[ChatMessage], max_tokens: int = 900) -> str:
        if not self.settings.api_key:
            raise RuntimeError("ZAI_API_KEY is not configured")

        payload: ChatCompletionRequest = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "stream": False,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en",
        }

        with httpx.Client(timeout=self.settings.timeout_seconds, headers=headers) as client:
            try:
                response = client.post(f"{self.settings.base_url}/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                raise RuntimeError("Z.AI request timed out. Try again or increase ZAI_TIMEOUT_SECONDS.") from exc
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(_format_http_error(exc.response)) from exc
            data = response.json()

        return _extract_content(data)


def _extract_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise ValueError("Unexpected Z.AI response: expected object")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Unexpected Z.AI response: missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("Unexpected Z.AI response: invalid choice")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Unexpected Z.AI response: missing message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Unexpected Z.AI response: empty content")

    return content.strip()


def _format_http_error(response: httpx.Response) -> str:
    details = _response_error_details(response)
    if response.status_code == 401:
        return "Z.AI authentication failed. Check ZAI_API_KEY and model access."
    if response.status_code == 429:
        return f"Z.AI rate limit or quota exceeded. {details}".strip()
    return f"Z.AI request failed with HTTP {response.status_code}. {details}".strip()


def _response_error_details(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:500] if text else ""

    if not isinstance(data, dict):
        return ""

    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
        return str(message) if message else ""

    message = data.get("message") or data.get("msg")
    return str(message) if message else ""
