from typing import Any

from app.observability import observe
from domain.models import Category
from llm_core.prompts.recommendations import build_feed_recommendation_prompt
from mcp_tools.llm import ask_llm_data
from mcp_tools.media import (
    fetch_latest_items_data,
    get_profile_data,
    list_watchlist_data,
    search_cached_items_data,
)

VIBE_MARKER = "\u0432\u0430\u0439\u0431"
COZY_RU = "\u0443\u044e\u0442\u043d\u0430\u044f"
ATMOSPHERIC_RU = "\u0430\u0442\u043c\u043e\u0441\u0444\u0435\u0440\u043d\u0430\u044f"


@observe("recommend_media")
def recommend_media_data(
    query: str,
    category: Category = "all",
    refresh: bool = True,
    limit: int = 8,
    limit_per_source: int = 10,
) -> dict[str, Any]:
    fetched_count = 0
    if refresh:
        fetched_count = len(fetch_latest_items_data(category=category, limit_per_source=limit_per_source))

    candidates = _search_recommendation_candidates(query=query, category=category, limit=limit)
    if not candidates:
        recent_candidates = search_cached_items_data(query="", category=category, days=30, limit=limit * 3)
        candidates = _prefer_exact_category(recent_candidates, category=category, limit=limit)

    profile = get_profile_data()
    watchlist = list_watchlist_data(type="all", status="all")
    prompt = _recommend_media_prompt(
        query=query,
        category=category,
        profile=profile,
        watchlist=watchlist,
        candidates=candidates,
    )
    llm = ask_llm_data(prompt)

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
    }


def _search_recommendation_candidates(query: str, category: Category, limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for variant in _query_variants(query):
        for item in search_cached_items_data(query=variant, category=category, days=30, limit=limit):
            url = str(item.get("url", "")).lower()
            if url in seen:
                continue
            seen.add(url)
            candidates.append(item)
            if len(candidates) >= limit:
                return candidates

    return candidates


def _query_variants(query: str) -> list[str]:
    variants = [query]
    normalized = query.lower()
    if VIBE_MARKER in normalized or "vibe" in normalized:
        variants.extend([COZY_RU, ATMOSPHERIC_RU, "cozy", "vibe"])
    return variants


def _prefer_exact_category(candidates: list[dict[str, Any]], category: Category, limit: int) -> list[dict[str, Any]]:
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
    return build_feed_recommendation_prompt(
        query=query,
        category=category,
        profile=profile,
        watchlist=watchlist,
        candidates=candidates,
    )

