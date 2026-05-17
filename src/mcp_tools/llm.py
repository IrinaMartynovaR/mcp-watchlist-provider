from typing import Any

from langfuse import observe

from llm_core.client import create_llm_client
from llm_core.prompts.recommendations import SYSTEM_PROMPT
from llm_core.schemas import ChatMessage
from llm_core.settings import llm_settings


@observe(name="ask_llm", as_type="generation")
def ask_llm_data(prompt: str, system: str | None = None) -> dict[str, Any]:
    """Отправляет prompt в LLM и возвращает нормализованный ответ.

    Args:
        prompt: Пользовательский prompt для модели.
        system: Необязательный системный prompt вместо дефолтного.

    Returns:
        Словарь с провайдером, моделью и текстом ответа.
    """
    client = create_llm_client()
    messages: list[ChatMessage] = [
        {"role": "system", "content": system or SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = client.chat(messages)
    return {
        "provider": llm_settings.normalized_provider,
        "model": llm_settings.normalized_model,
        "response": response,
    }

