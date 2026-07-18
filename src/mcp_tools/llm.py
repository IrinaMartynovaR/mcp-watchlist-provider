from typing import Any

from langfuse import observe

from llm_core.client import create_llm_client
from llm_core.schemas import ChatMessage
from llm_core.settings import LLMSettings


@observe(name="ask_llm", as_type="generation")
def ask_llm_data(
    prompt: str,
    settings: LLMSettings,
    default_system: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 4000,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Отправляет prompt в LLM и возвращает нормализованный ответ.

    Args:
        prompt: Пользовательский prompt для модели.
        settings: Настройки LLM-провайдера.
        default_system: Системный prompt по умолчанию.
        system: Необязательный системный prompt вместо дефолтного.
        model: Необязательная модель вместо дефолтной из настроек —
            для служебных задач (например, классификации) на дешёвой модели.
        max_tokens: Максимальная длина ответа в токенах.
        reasoning_effort: None — из настроек; пустая строка — отключить
            reasoning для этого вызова (служебные вызовы на не-reasoning моделях).

    Returns:
        Словарь с провайдером, моделью и текстом ответа.
    """
    client = create_llm_client(settings)
    messages: list[ChatMessage] = [
        {"role": "system", "content": system or default_system},
        {"role": "user", "content": prompt},
    ]
    response = client.chat(messages, max_tokens=max_tokens, model=model, reasoning_effort=reasoning_effort)
    return {
        "provider": settings.normalized_provider,
        "model": model or settings.normalized_model,
        "response": response,
    }
