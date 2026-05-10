from __future__ import annotations

from typing import Any

from watchquest.models import Category
from watchquest.observability import observe
from watchquest.tools.llm import ask_llm_data
from watchquest.tools.media import (
    fetch_latest_items_data,
    get_profile_data,
    list_watchlist_data,
    search_cached_items_data,
)


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
    if "вайб" in normalized:
        variants.extend(["уютная", "атмосферная", "cozy", "vibe"])
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
    return (
        "You are creating a practical WatchQuest recommendation. "
        "Return 1-5 recommendations in the user's language. "
        "Use only the provided candidates. Do not invent titles that are not present in candidates. "
        "If the user asks for games, recommend games mentioned in the candidates, not generic industry articles. "
        "If there are fewer than 3 solid matches, recommend fewer and say the feed context is limited. "
        "For each recommendation include title, type/category, why it fits, and a concrete next action.\n\n"
        f"User query: {query}\n"
        f"Category filter: {category}\n"
        f"Profile: {profile}\n"
        f"Watchlist: {watchlist[:20]}\n"
        f"Candidates: {candidates}"
    )
