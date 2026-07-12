import logging
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

import httpx

from llm_core.schemas import ChatMessage

logger = logging.getLogger(__name__)


class OpenRouterChatRequest(TypedDict):
    """Описывает payload chat-completions запроса к OpenRouter."""
    model: str
    messages: list[ChatMessage]
    temperature: float
    stream: bool
    max_tokens: NotRequired[int]


class EmbeddingRequest(TypedDict):
    """Описывает payload embeddings-запроса к OpenRouter."""
    model: str
    input: list[str]


@dataclass(frozen=True)
class OpenRouterSettings:
    """Хранит провайдерные настройки OpenRouter-клиента."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    embedding_model: str = ""
    timeout_seconds: float = 90.0
    temperature: float = 0.2
    http_referer: str = ""
    x_title: str = ""


class OpenRouterClient:
    """Реализует текстовый и embedding-клиент для OpenRouter API."""

    def __init__(self, settings: OpenRouterSettings) -> None:
        """Создаёт клиента с заранее валидированными настройками.

        Args:
            settings: Конфигурация доступа к OpenRouter API.
        """
        self.settings = settings

    def chat(self, messages: list[ChatMessage], max_tokens: int = 1500, model: str | None = None) -> str:
        """Отправляет chat-completions запрос в OpenRouter.

        Args:
            messages: История сообщений для модели.
            max_tokens: Максимальная длина ответа.
            model: Необязательная модель вместо дефолтной из настроек —
                для служебных задач на более дешёвой модели.

        Returns:
            Финальный текст ответа модели.

        Raises:
            RuntimeError: Если API-ключ не задан, запрос завершился таймаутом
                или провайдер вернул ошибочный HTTP-статус.
            ValueError: Если структура ответа провайдера некорректна.
        """
        if not self.settings.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")

        chat_model = model or self.settings.model
        payload: OpenRouterChatRequest = {
            "model": chat_model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "stream": False,
            "max_tokens": max_tokens,
        }
        logger.info(
            "LLM chat request started",
            extra={"provider": "openrouter", "model": chat_model, "message_count": len(messages)},
        )

        with httpx.Client(timeout=self.settings.timeout_seconds, headers=self._headers()) as client:
            data = _post_json(
                client,
                f"{self.settings.base_url}/chat/completions",
                payload,
                provider="openrouter",
                model=chat_model,
                action="chat",
            )

        content = _extract_content(data)
        logger.info("LLM chat request completed", extra={"model": chat_model, "response_chars": len(content)})
        return content

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Строит embedding-векторы для списка текстов через OpenRouter.

        Args:
            texts: Тексты для векторизации.

        Returns:
            Список embedding-векторов в порядке входных текстов.

        Raises:
            RuntimeError: Если API-ключ или embedding-модель не заданы, запрос
                завершился таймаутом или провайдер вернул ошибочный HTTP-статус.
            ValueError: Если структура ответа провайдера некорректна.
        """
        if not self.settings.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")
        if not self.settings.embedding_model:
            raise RuntimeError("LLM_EMBEDDING_MODEL is not configured")
        if not texts:
            return []

        payload: EmbeddingRequest = {
            "model": self.settings.embedding_model,
            "input": texts,
        }
        logger.info(
            "LLM embeddings request started",
            extra={"provider": "openrouter", "model": self.settings.embedding_model, "text_count": len(texts)},
        )

        with httpx.Client(timeout=self.settings.timeout_seconds, headers=self._headers()) as client:
            data = _post_json(
                client,
                f"{self.settings.base_url}/embeddings",
                payload,
                provider="openrouter",
                model=self.settings.embedding_model,
                action="embeddings",
            )

        vectors = _extract_embeddings(data, expected_count=len(texts))
        logger.info(
            "LLM embeddings request completed",
            extra={"model": self.settings.embedding_model, "vector_count": len(vectors)},
        )
        return vectors

    def _headers(self) -> dict[str, str]:
        """Собирает HTTP-заголовки запроса к OpenRouter.

        Returns:
            Заголовки авторизации и необязательной атрибуции приложения.
        """
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.http_referer:
            headers["HTTP-Referer"] = self.settings.http_referer
        if self.settings.x_title:
            headers["X-Title"] = self.settings.x_title
        return headers


def _post_json(
    client: httpx.Client,
    url: str,
    payload: Any,
    *,
    provider: str,
    model: str,
    action: str,
) -> Any:
    """Отправляет POST-запрос к OpenAI-совместимому API и возвращает JSON.

    Args:
        client: Открытый httpx-клиент с настроенными заголовками и timeout.
        url: Полный URL запроса.
        payload: JSON-тело запроса.
        provider: Имя провайдера для логирования.
        model: Имя модели для логирования.
        action: Тип запроса (`chat`/`embeddings`) для сообщений в логах.

    Returns:
        Десериализованный JSON-ответ.

    Raises:
        RuntimeError: Если запрос завершился таймаутом или провайдер вернул
            ошибочный HTTP-статус.
    """
    try:
        response = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        logger.warning(f"LLM {action} request timed out", extra={"provider": provider, "model": model})
        raise RuntimeError("LLM request timed out. Try again or increase LLM_TIMEOUT_SECONDS.") from exc
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            f"LLM {action} request failed",
            extra={"provider": provider, "model": model, "status_code": exc.response.status_code},
        )
        raise RuntimeError(_format_http_error(exc.response)) from exc
    return response.json()


def _extract_content(data: Any) -> str:
    """Извлекает текст ответа из провайдерного JSON.

    Args:
        data: Десериализованный JSON-ответ OpenRouter.

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


def _extract_embeddings(data: Any, expected_count: int) -> list[list[float]]:
    """Извлекает embedding-векторы из провайдерного JSON.

    Args:
        data: Десериализованный JSON-ответ OpenAI-совместимого embeddings API.
        expected_count: Ожидаемое число векторов.

    Returns:
        Векторы, упорядоченные по полю `index`.

    Raises:
        ValueError: Если ответ не соответствует ожидаемой структуре.
    """
    if not isinstance(data, dict):
        raise ValueError("Unexpected embeddings response: expected object")

    rows = data.get("data")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"Unexpected embeddings response: expected {expected_count} data rows")

    indexed: list[tuple[int, list[float]]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Unexpected embeddings response: invalid data row")
        index = row.get("index")
        embedding = row.get("embedding")
        if not isinstance(index, int) or not isinstance(embedding, list) or not embedding:
            raise ValueError("Unexpected embeddings response: missing index or embedding")
        if not all(isinstance(value, int | float) for value in embedding):
            raise ValueError("Unexpected embeddings response: non-numeric embedding values")
        indexed.append((index, [float(value) for value in embedding]))

    indexed.sort(key=lambda pair: pair[0])
    if [pair[0] for pair in indexed] != list(range(expected_count)):
        raise ValueError("Unexpected embeddings response: non-contiguous indexes")
    return [pair[1] for pair in indexed]


def _format_http_error(response: httpx.Response) -> str:
    """Преобразует HTTP-ошибку провайдера в человекочитаемый текст.

    Args:
        response: Ошибочный HTTP-ответ OpenRouter.

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
        response: Ошибочный HTTP-ответ OpenRouter.

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
