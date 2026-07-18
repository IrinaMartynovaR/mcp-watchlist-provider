"""Общие утилиты для LLM-ответов: разбор JSON и ключи персистентных кешей.

Единая точка для повторяющихся механик всех LLM-сценариев проекта
(классификация записей, анализ запроса, эмбеддинги): модели оборачивают
JSON в markdown-ограждение, а результаты кешируются по хешу контента.
"""

import hashlib
import json
from typing import Any


def parse_llm_json(response: str) -> Any:
    """Извлекает JSON из текста LLM-ответа.

    Модели часто оборачивают JSON в markdown-ограждение (```json ... ```) —
    ограждение снимается перед разбором.

    Args:
        response: Сырой текст ответа модели.

    Returns:
        Десериализованное JSON-значение.

    Raises:
        ValueError: Если после очистки текст не является валидным JSON.
    """
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc


def content_cache_key(namespace: str, text: str) -> str:
    """Строит ключ персистентного кеша, привязанный к неймспейсу и контенту.

    Args:
        namespace: Префикс, кодирующий версию схемы и/или модель.
        text: Контент, по хешу которого кешируется результат.

    Returns:
        Ключ вида ``namespace::sha256(text)``.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{namespace}::{digest}"
