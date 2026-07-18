from typing import Protocol

from llm_core.providers.openrouter import OpenRouterClient, OpenRouterSettings
from llm_core.schemas import ChatMessage
from llm_core.settings import LLMSettings


class LLMClient(Protocol):
    """Задаёт общий контракт для текстового LLM-клиента."""

    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1500,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Отправляет диалог в LLM и возвращает итоговый текст.

        Args:
            messages: Сообщения диалога в формате выбранного провайдера.
            max_tokens: Максимальное число токенов в ответе.
            model: Необязательная модель вместо дефолтной из настроек.
            reasoning_effort: None — из настроек; пустая строка — отключить.

        Returns:
            Текст финального ответа модели.
        """
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Строит embedding-векторы для списка текстов.

        Args:
            texts: Тексты для векторизации.

        Returns:
            Список embedding-векторов в порядке входных текстов.
        """
        ...


def create_llm_client(settings: LLMSettings) -> LLMClient:
    """Создаёт LLM-клиент на основе переданной конфигурации.

    Args:
        settings: Настройки LLM-провайдера.

    Returns:
        Провайдерный клиент, реализующий общий LLM-контракт.

    Raises:
        ValueError: Если выбран неподдерживаемый LLM-провайдер.
    """
    if settings.normalized_provider == "openrouter":
        return OpenRouterClient(
            settings=OpenRouterSettings(
                api_key=settings.normalized_api_key,
                base_url=settings.normalized_base_url,
                model=settings.normalized_model,
                embedding_model=settings.normalized_embedding_model,
                timeout_seconds=settings.timeout_seconds,
                temperature=settings.temperature,
                reasoning_effort=settings.normalized_reasoning_effort,
            )
        )
    raise ValueError(f"Unsupported LLM provider: {settings.normalized_provider}")
