from typing import Protocol

from llm_core.providers.openrouter import OpenRouterClient, OpenRouterSettings
from llm_core.schemas import ChatMessage
from llm_core.settings import llm_settings


class LLMClient(Protocol):
    """Задаёт общий контракт для текстового LLM-клиента."""

    def chat(self, messages: list[ChatMessage], max_tokens: int = 1500, model: str | None = None) -> str:
        """Отправляет диалог в LLM и возвращает итоговый текст.

        Args:
            messages: Сообщения диалога в формате выбранного провайдера.
            max_tokens: Максимальное число токенов в ответе.
            model: Необязательная модель вместо дефолтной из настроек.

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


def create_llm_client() -> LLMClient:
    """Создаёт LLM-клиент на основе текущей конфигурации.

    Returns:
        Провайдерный клиент, реализующий общий LLM-контракт.

    Raises:
        ValueError: Если выбран неподдерживаемый LLM-провайдер.
    """
    if llm_settings.normalized_provider == "openrouter":
        return OpenRouterClient(
            settings=OpenRouterSettings(
                api_key=llm_settings.normalized_api_key,
                base_url=llm_settings.normalized_base_url,
                model=llm_settings.normalized_model,
                embedding_model=llm_settings.normalized_embedding_model,
                timeout_seconds=llm_settings.timeout_seconds,
                temperature=llm_settings.temperature,
            )
        )
    raise ValueError(f"Unsupported LLM provider: {llm_settings.normalized_provider}")
