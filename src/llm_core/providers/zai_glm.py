import logging
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

import httpx

from llm_core.schemas import ChatMessage

logger = logging.getLogger(__name__)


class ChatCompletionRequest(TypedDict):
    """Описывает payload chat-completions запроса к Z.AI."""
    model: str
    messages: list[ChatMessage]
    temperature: float
    stream: bool
    thinking: dict[str, str]
    max_tokens: NotRequired[int]


@dataclass(frozen=True)
class ZAIGLMSettings:
    """Хранит провайдерные настройки Z.AI GLM-клиента."""
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float = 0.2


class ZAIGLMClient:
    """Реализует текстовый клиент для Z.AI GLM API."""

    def __init__(self, settings: ZAIGLMSettings) -> None:
        """Создаёт клиента с заранее валидированными настройками.

        Args:
            settings: Конфигурация доступа к Z.AI API.
        """
        self.settings = settings

    def chat(self, messages: list[ChatMessage], max_tokens: int = 1500) -> str:
        """Отправляет chat-completions запрос в Z.AI.

        Args:
            messages: История сообщений для модели.
            max_tokens: Максимальная длина ответа.

        Returns:
            Финальный текст ответа модели.

        Raises:
            RuntimeError: Если API-ключ не задан, запрос завершился таймаутом
                или провайдер вернул ошибочный HTTP-статус.
            ValueError: Если структура ответа провайдера некорректна.
        """
        if not self.settings.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")

        payload: ChatCompletionRequest = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "stream": False,
            "thinking": {"type": "disabled"},
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en",
        }
        logger.info(
            "LLM chat request started",
            extra={"provider": "zai_glm", "model": self.settings.model, "message_count": len(messages)},
        )

        with httpx.Client(timeout=self.settings.timeout_seconds, headers=headers) as client:
            try:
                response = client.post(f"{self.settings.base_url}/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                logger.warning("LLM request timed out", extra={"model": self.settings.model})
                raise RuntimeError("LLM request timed out. Try again or increase LLM_TIMEOUT_SECONDS.") from exc
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "LLM request failed",
                    extra={"model": self.settings.model, "status_code": exc.response.status_code},
                )
                raise RuntimeError(_format_http_error(exc.response)) from exc
            data = response.json()

        content = _extract_content(data)
        logger.info("LLM chat request completed", extra={"model": self.settings.model, "response_chars": len(content)})
        return content


def _extract_content(data: Any) -> str:
    """Извлекает текст ответа из провайдерного JSON.

    Args:
        data: Десериализованный JSON-ответ Z.AI.

    Returns:
        Непустой текст сообщения ассистента.

    Raises:
        ValueError: Если ответ не соответствует ожидаемой структуре.
    """
    if not isinstance(data, dict):
        raise ValueError("Unexpected LLM response: expected object")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Unexpected LLM response: missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("Unexpected LLM response: invalid choice")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Unexpected LLM response: missing message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        finish_reason = first_choice.get("finish_reason")
        raise ValueError(f"Unexpected LLM response: empty final content, finish_reason={finish_reason}")

    return content.strip()


def _format_http_error(response: httpx.Response) -> str:
    """Преобразует HTTP-ошибку провайдера в человекочитаемый текст.

    Args:
        response: Ошибочный HTTP-ответ Z.AI.

    Returns:
        Сообщение об ошибке для верхнего уровня приложения.
    """
    details = _response_error_details(response)
    if response.status_code == 401:
        return "LLM authentication failed. Check LLM_API_KEY and model access."
    if response.status_code == 429:
        return f"LLM rate limit or quota exceeded. {details}".strip()
    return f"LLM request failed with HTTP {response.status_code}. {details}".strip()


def _response_error_details(response: httpx.Response) -> str:
    """Извлекает краткие детали ошибки из ответа провайдера.

    Args:
        response: Ошибочный HTTP-ответ Z.AI.

    Returns:
        Короткое текстовое пояснение ошибки либо пустую строку.
    """
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

