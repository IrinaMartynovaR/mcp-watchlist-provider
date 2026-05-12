import logging
from typing import Any

from langfuse import observe

from domain.models import Category
from llm_core.prompts.recommendations import build_feed_recommendation_prompt
from mcp_tools.llm import ask_llm_data
from mcp_tools.media import (
    fetch_latest_items_data,
    get_profile_data,
    list_watchlist_data,
    search_cached_items_data,
)
from mcp_tools.settings import (
    CACHE_SEARCH_DAYS,
    RECOMMENDATION_FALLBACK_CANDIDATE_MULTIPLIER,
    RECOMMENDATION_KEYWORD_MARKER,
    RECOMMENDATION_KEYWORD_VARIANTS,
    RECOMMENDATION_LIMIT,
    RECOMMENDATION_SOURCE_LIMIT,
)

logger = logging.getLogger(__name__)


@observe(name="recommend_media", as_type="chain")
def recommend_media_data(
    query: str,
    category: Category = "all",
    refresh: bool = True,
    limit: int = RECOMMENDATION_LIMIT,
    limit_per_source: int = RECOMMENDATION_SOURCE_LIMIT,
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
            days=CACHE_SEARCH_DAYS,
            limit=limit * RECOMMENDATION_FALLBACK_CANDIDATE_MULTIPLIER,
        )
        candidates = _prefer_exact_category(recent_candidates, category=category, limit=limit)
        fallback_used = True
        logger.info(
            "Recommendation fallback candidates selected",
            extra={"query": query, "category": category, "candidate_count": len(candidates)},
        )

    profile = get_profile_data()
    watchlist = list_watchlist_data(media_type="all", status="all")
    prompt = _recommend_media_prompt(
        query=query,
        category=category,
        profile=profile,
        watchlist=watchlist,
        candidates=candidates,
    )
    llm = ask_llm_data(prompt)
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

    return {
        "query": query,
        "category": category,
        "refreshed": refresh,
        "fetched_count": fetched_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "recommendation": llm["response"],
        "model": llm["model"],
        "provider": llm["provider"],
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
        for item in search_cached_items_data(query=variant, category=category, days=CACHE_SEARCH_DAYS, limit=limit):
            url = str(item.get("url", "")).lower()
            if url in seen:
                continue
            seen.add(url)
            candidates.append(item)
            if len(candidates) >= limit:
                return candidates

    return candidates


def _query_variants(query: str) -> list[str]:
    """Строит варианты поискового запроса для кеша.

    Args:
        query: Исходный пользовательский запрос.

    Returns:
        Список базового и дополнительных вариантов запроса.
    """
    variants = [query]
    normalized = query.lower()
    if RECOMMENDATION_KEYWORD_MARKER in normalized or "vibe" in normalized:
        variants.extend(RECOMMENDATION_KEYWORD_VARIANTS)
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

