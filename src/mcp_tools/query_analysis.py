"""LLM-анализ пользовательского запроса рекомендации.

Единый паттерн проекта: LLM — основной путь (с персистентным кешем),
эвристика по ключевым словам — только fail-soft fallback. Модуль извлекает
из запроса явно названный медиум (аниме/игра/фильм/сериал), чтобы retrieval
и персональное ранжирование не вытесняли такие кандидаты общим вкусовым
профилем.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from langfuse import observe

from app.config import RuntimeConfig
from domain.storage.json_store import read_json, update_json
from llm_core.parsing import content_cache_key, parse_llm_json
from llm_core.prompts.classification import build_query_analysis_prompt
from mcp_tools.llm import ask_llm_data

logger = logging.getLogger(__name__)

AskLLM = Callable[..., dict[str, Any]]

_VALID_MEDIUMS = frozenset({"anime", "game", "movie", "series"})
_CACHE_SCHEMA_VERSION = "v1"

# Fallback-эвристика на случай сбоя или выключенного LLM-классификатора:
# русские основы без границ слова (покрывают словоформы), английские — со
# границами. Основной путь — LLM, эти маркеры сознательно консервативны.
_MEDIUM_FALLBACK_MARKERS: dict[str, re.Pattern[str]] = {
    "anime": re.compile(r"аниме|anime", re.IGNORECASE),
    "game": re.compile(r"\bигр\w*|\bgames?\b", re.IGNORECASE),
    "movie": re.compile(r"фильм|кино\b|\bmovies?\b|\bfilms?\b", re.IGNORECASE),
    "series": re.compile(r"сериал|\bseries\b|\btv show\b", re.IGNORECASE),
}


@observe(name="analyze_query", as_type="tool")
def analyze_query_data(
    query: str,
    config: RuntimeConfig,
    ask_llm: AskLLM = ask_llm_data,
) -> dict[str, Any]:
    """Извлекает структурные ограничения из запроса пользователя.

    Результат кешируется по хешу запроса и модели. Fail-soft: при любом
    сбое LLM возвращается эвристический разбор по ключевым словам.

    Args:
        query: Пользовательский запрос рекомендации.
        config: Runtime-настройки приложения.
        ask_llm: Инъецируемая функция LLM-вызова.

    Returns:
        Словарь ограничений, сейчас `{"medium": str | None}`.
    """
    normalized_query = query.strip().lower()
    if not normalized_query:
        return {"medium": None}
    if not config.tools.classifier_enabled:
        return {"medium": _heuristic_medium(query)}

    cache_file = config.backend.query_analysis_cache_file
    model = config.tools.normalized_classifier_model
    key = content_cache_key(f"{_CACHE_SCHEMA_VERSION}::{model}", normalized_query)

    cache: dict[str, Any] = read_json(cache_file, {})
    cached = cache.get(key)
    if isinstance(cached, dict):
        return dict(cached)

    try:
        llm = ask_llm(
            build_query_analysis_prompt(query),
            config.llm,
            config.prompts.recommendation_system,
            system=config.prompts.query_analyzer_system,
            model=model,
            max_tokens=200,
            reasoning_effort="",
        )
        analysis = _parse_analysis(str(llm["response"]))
    except Exception:
        logger.warning(
            "LLM query analysis failed; falling back to keyword heuristic",
            extra={"query": query, "model": model},
            exc_info=True,
        )
        return {"medium": _heuristic_medium(query)}

    def merge(current: dict[str, Any]) -> dict[str, Any]:
        current[key] = analysis
        return current

    empty_cache: dict[str, Any] = {}
    update_json(cache_file, empty_cache, merge)
    logger.info("Query analyzed", extra={"query": query, "medium": analysis.get("medium")})
    return dict(analysis)


def item_matches_medium(item: dict[str, Any], medium: str) -> bool:
    """Проверяет, относится ли RSS-кандидат к запрошенному медиуму.

    Основной сигнал — поле `medium`, проставленное LLM-классификатором;
    для неклассифицированных записей — эвристика по тексту кандидата.

    Args:
        item: RSS-кандидат.
        medium: Медиум из `analyze_query_data`.

    Returns:
        True, если кандидат принадлежит запрошенному медиуму.
    """
    item_medium = str(item.get("medium") or "")
    if item_medium:
        return item_medium == medium

    pattern = _MEDIUM_FALLBACK_MARKERS.get(medium)
    if pattern is None:
        return False
    haystack = " ".join(
        [
            str(item.get("source") or ""),
            str(item.get("title_entity") or ""),
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
        ]
    )
    return bool(pattern.search(haystack))


def _heuristic_medium(query: str) -> str | None:
    """Определяет медиум запроса по ключевым словам (fallback-путь).

    Args:
        query: Пользовательский запрос.

    Returns:
        Ключ медиума либо None.
    """
    for medium, pattern in _MEDIUM_FALLBACK_MARKERS.items():
        if pattern.search(query):
            return medium
    return None


def _parse_analysis(response: str) -> dict[str, Any]:
    """Разбирает JSON-ответ анализатора запроса.

    Args:
        response: Сырой текст ответа модели.

    Returns:
        Нормализованный словарь `{"medium": str | None}`.

    Raises:
        ValueError: Если ответ не JSON-объект ожидаемой формы.
    """
    data = parse_llm_json(response)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object with query constraints")

    medium = data.get("medium")
    return {"medium": medium if medium in _VALID_MEDIUMS else None}
