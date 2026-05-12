from typing import Protocol

from llm_core.providers.zai_glm import ZAIGLMClient, ZAIGLMSettings
from llm_core.schemas import ChatMessage
from llm_core.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)


class LLMClient(Protocol):
    """Задаёт общий контракт для текстового LLM-клиента."""

    def chat(self, messages: list[ChatMessage], max_tokens: int = 1500) -> str:
        """Отправляет диалог в LLM и возвращает итоговый текст.

        Args:
            messages: Сообщения диалога в формате выбранного провайдера.
            max_tokens: Максимальное число токенов в ответе.

        Returns:
            Текст финального ответа модели.
        """
        ...


def create_llm_client() -> LLMClient:
    """Создаёт LLM-клиент на основе текущей конфигурации.

    Returns:
        Провайдерный клиент, реализующий общий LLM-контракт.

    Raises:
        ValueError: Если выбран неподдерживаемый LLM-провайдер.
    """
    if LLM_PROVIDER == "zai_glm":
        return ZAIGLMClient(
            settings=ZAIGLMSettings(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                model=LLM_MODEL,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
                temperature=LLM_TEMPERATURE,
            )
        )
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
