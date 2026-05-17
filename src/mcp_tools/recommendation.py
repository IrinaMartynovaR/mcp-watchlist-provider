import logging
from typing import Any
from uuid import uuid4

from langfuse import observe

from domain.models import Category
from llm_core.prompts.recommendations import build_feed_recommendation_prompt
from llm_core.settings import llm_settings
from mcp_tools.feedback import save_recommendation_result
from mcp_tools.llm import ask_llm_data
from mcp_tools.media import (
    fetch_latest_items_data,
    get_profile_data,
    list_watchlist_data,
    search_cached_items_data,
)
from mcp_tools.settings import tool_settings

logger = logging.getLogger(__name__)


@observe(name="recommend_media", as_type="chain")
def recommend_media_data(
    query: str,
    category: Category = "all",
    refresh: bool = True,
    limit: int = tool_settings.recommendation_limit,
    limit_per_source: int = tool_settings.recommendation_source_limit,
) -> dict[str, Any]:
    """Собирает рекомендацию из RSS-кандидатов и LLM-ответа.

    Args:
        query: Пользовательский запрос.
        category: Категория поиска.
        refresh: Нужно ли обновить RSS-кеш перед подбором.
        limit: Максимальное число кандидатов для передачи в LLM.
        limit_per_source: Лимит RSS-записей на один источник.

    Returns:
        Словарь с кандидатами, LLM-рекомендацией и метаданными запроса.
    """
    fetched_count = 0
    if refresh:
        fetched_count = len(fetch_latest_items_data(category=category, limit_per_source=limit_per_source))

    candidates = _search_recommendation_candidates(query=query, category=category, limit=limit)
    fallback_used = False
    if not candidates:
        recent_candidates = search_cached_items_data(
            query="",
            category=category,
            days=tool_settings.cache_search_days,
            limit=limit * tool_settings.recommendation_fallback_candidate_multiplier,
        )
        candidates = _prefer_exact_category(recent_candidates, category=category, limit=limit)
        fallback_used = True
        logger.info(
            "Recommendation fallback candidates selected",
            extra={"query": query, "category": category, "candidate_count": len(candidates)},
        )

    profile = get_profile_data()
    candidates = _rank_candidates_by_learned_preferences(candidates, profile)
    watchlist = list_watchlist_data(media_type="all", status="all")
    prompt = _recommend_media_prompt(
        query=query,
        category=category,
        profile=profile,
        watchlist=watchlist,
        candidates=candidates,
    )
    llm_error: str | None = None
    try:
        llm = ask_llm_data(prompt)
    except RuntimeError as exc:
        llm_error = str(exc)
        llm = _fallback_llm_result(error=llm_error, candidates=candidates)
        logger.warning(
            "Recommendation LLM fallback used",
            extra={"query": query, "category": category, "error": llm_error},
        )
    logger.info(
        "Recommendation generated",
        extra={
            "query": query,
            "category": category,
            "refreshed": refresh,
            "fetched_count": fetched_count,
            "candidate_count": len(candidates),
            "provider": llm["provider"],
            "model": llm["model"],
        },
    )

    result = {
        "id": uuid4().hex,
        "query": query,
        "category": category,
        "refreshed": refresh,
        "fetched_count": fetched_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "recommendation": llm["response"],
        "model": llm["model"],
        "provider": llm["provider"],
        "llm_error": llm_error,
        "tool_usage": {
            "fetch_latest_items": refresh,
            "search_cached_items": True,
            "fallback_recent_search": fallback_used,
            "get_profile": True,
            "list_watchlist": True,
            "ask_llm": True,
        },
        "mcp_tool_usage": {
            "get_profile": True,
            "update_profile": False,
            "list_sources": False,
            "validate_sources": False,
            "refresh_feeds": refresh,
            "search_cached_items": True,
            "add_to_watchlist": False,
            "list_watchlist": True,
            "rate_watchlist_item": False,
            "recommend_media": True,
        },
    }
    if llm_error:
        result["llm_error"] = llm_error
    save_recommendation_result(result)
    return result


def _search_recommendation_candidates(query: str, category: Category, limit: int) -> list[dict[str, Any]]:
    """Ищет уникальных RSS-кандидатов по вариантам запроса.

    Args:
        query: Исходный пользовательский запрос.
        category: Категория поиска.
        limit: Максимальное число кандидатов.

    Returns:
        Дедуплицированный список RSS-кандидатов.
    """
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for variant in _query_variants(query):
        for item in search_cached_items_data(
            query=variant,
            category=category,
            days=tool_settings.cache_search_days,
            limit=limit,
        ):
            key = _candidate_key(item)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(item)
            if len(candidates) >= limit:
                return candidates

    return candidates


def _candidate_key(item: dict[str, Any]) -> str:
    """Строит стабильный ключ RSS-кандидата для дедупликации.

    Args:
        item: RSS-кандидат.

    Returns:
        Ключ по URL, если он есть, иначе ключ по названию.
    """
    url = str(item.get("url") or "").strip().lower()
    if url:
        return f"url:{url}"
    return f"title:{str(item.get('title') or '').strip().lower()}"


def _fallback_llm_result(error: str, candidates: list[dict[str, Any]]) -> dict[str, str]:
    """Строит безопасный LLM-результат, когда провайдер временно недоступен.

    Args:
        error: Текст ошибки LLM-провайдера.
        candidates: Уже найденные RSS-кандидаты.

    Returns:
        Нормализованный LLM-результат с объяснением fallback.
    """
    titles = [
        str(candidate.get("title") or "").strip()
        for candidate in candidates[:3]
        if isinstance(candidate, dict) and str(candidate.get("title") or "").strip()
    ]
    context = "\n".join(f"- {title}" for title in titles)
    details = f"\n\nRSS-кандидаты для ручной проверки:\n{context}" if context else ""
    return {
        "provider": llm_settings.normalized_provider,
        "model": llm_settings.normalized_model,
        "response": (
            "LLM сейчас не смогла собрать финальную рекомендацию: "
            f"{error}. RSS уже проверен, поэтому можно посмотреть найденные кандидаты ниже."
            f"{details}"
        ),
    }


def _query_variants(query: str) -> list[str]:
    """Строит варианты поискового запроса для кеша.

    Args:
        query: Исходный пользовательский запрос.

    Returns:
        Список базового и дополнительных вариантов запроса.
    """
    variants = [query]
    normalized = query.lower()
    if tool_settings.normalized_keyword_marker in normalized or "vibe" in normalized:
        variants.extend(tool_settings.normalized_keyword_variants)
    return variants


def _prefer_exact_category(candidates: list[dict[str, Any]], category: Category, limit: int) -> list[dict[str, Any]]:
    """Сортирует fallback-кандидатов по близости категории.

    Args:
        candidates: Набор RSS-кандидатов.
        category: Желаемая категория.
        limit: Максимальное число элементов в результате.

    Returns:
        Приоритетный список exact → mixed → other.
    """
    if category == "all":
        return candidates[:limit]

    exact = [item for item in candidates if item.get("category") == category]
    mixed = [item for item in candidates if item.get("category") == "mixed"]
    other = [item for item in candidates if item.get("category") not in {category, "mixed"}]
    return [*exact, *mixed, *other][:limit]


def _rank_candidates_by_learned_preferences(
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ранжирует кандидатов по learned_preferences пользователя.

    Args:
        candidates: RSS-кандидаты для LLM.
        profile: Профиль пользователя с накопленными feedback-весами.

    Returns:
        Кандидаты, отсортированные по персональному score.
    """
    learned = profile.get("learned_preferences", {})
    if not isinstance(learned, dict):
        return candidates

    scored_candidates = [
        _candidate_with_preference_score(candidate, learned)
        for candidate in candidates
    ]
    return sorted(scored_candidates, key=lambda item: int(item.get("preference_score", 0)), reverse=True)


def _candidate_with_preference_score(candidate: dict[str, Any], learned: dict[str, Any]) -> dict[str, Any]:
    """Добавляет кандидату score и причины совпадения с learned_preferences.

    Args:
        candidate: RSS-кандидат.
        learned: Накопленные веса предпочтений.

    Returns:
        Копию кандидата с `preference_score` и `preference_reasons`.
    """
    score = 0
    reasons: list[str] = []
    score += _score_field(learned, "categories", str(candidate.get("category") or ""), reasons)
    score += _score_field(learned, "sources", str(candidate.get("source") or ""), reasons)

    tags = candidate.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            score += _score_field(learned, "tags", str(tag).strip().lower(), reasons)

    result = dict(candidate)
    result["preference_score"] = score
    result["preference_reasons"] = reasons
    return result


def _score_field(learned: dict[str, Any], section: str, key: str, reasons: list[str]) -> int:
    """Возвращает вес learned preference для одного поля."""
    if not key:
        return 0
    values = learned.get(section, {})
    if not isinstance(values, dict):
        return 0
    raw_score = values.get(key)
    if not isinstance(raw_score, int) or raw_score == 0:
        return 0
    reasons.append(f"{section}:{key}:{raw_score:+d}")
    return raw_score


def _recommend_media_prompt(
    query: str,
    category: Category,
    profile: dict[str, Any],
    watchlist: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    """Проксирует сборку LLM-prompt для рекомендаций.

    Args:
        query: Исходный пользовательский запрос.
        category: Категория поиска.
        profile: Профиль пользователя.
        watchlist: Текущий watchlist.
        candidates: RSS-кандидаты.

    Returns:
        Готовый prompt для LLM.
    """
    return build_feed_recommendation_prompt(
        query=query,
        category=category,
        profile=profile,
        watchlist=watchlist,
        candidates=candidates,
    )

