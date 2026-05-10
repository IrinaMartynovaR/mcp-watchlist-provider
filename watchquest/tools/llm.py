from __future__ import annotations

import json
from typing import Any

from watchquest.config import CACHE_FILE, ZAI_MODEL
from watchquest.llm.zai import ChatMessage, ZAIClient
from watchquest.models import Category
from watchquest.observability import observe
from watchquest.storage.json_store import read_json
from watchquest.tools.media import get_profile_data, list_watchlist_data

SYSTEM_PROMPT = (
    "You are WatchQuest, a concise bilingual media recommendation assistant. "
    "Recommend games, movies, series, and articles using the user's taste profile, watchlist, and recent feed items. "
    "Prefer concrete titles and explain why each recommendation fits. Answer in the user's language."
)


@observe("ask_llm")
def ask_llm_data(prompt: str, system: str | None = None) -> dict[str, Any]:
    client = ZAIClient()
    messages: list[ChatMessage] = [
        {"role": "system", "content": system or SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = client.chat(messages)
    return {"provider": "z-ai", "model": ZAI_MODEL, "response": response}


@observe("recommend_with_llm")
def recommend_with_llm_data(query: str = "", category: Category = "all", limit: int = 8) -> dict[str, Any]:
    profile = get_profile_data()
    watchlist = list_watchlist_data(type="all", status="all")
    cached: dict[str, Any] = read_json(CACHE_FILE, {"items": []})
    candidates = _candidate_items(cached.get("items", []), category=category, limit=limit)
    prompt = _recommendation_prompt(query=query, profile=profile, watchlist=watchlist, candidates=candidates)
    return ask_llm_data(prompt)


def _candidate_items(items: Any, category: Category, limit: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_category = item.get("category")
        if category != "all" and item_category not in {category, "mixed"}:
            continue
        results.append(dict(item))
        if len(results) >= limit:
            break
    return results


def _recommendation_prompt(
    query: str,
    profile: dict[str, Any],
    watchlist: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    context = {
        "user_query": query,
        "profile": profile,
        "watchlist": watchlist[:20],
        "recent_candidates": candidates,
    }
    return (
        "Use this JSON context to produce 3-5 recommendations. "
        "For each recommendation include: title, type, why it fits, and next action.\n\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2, default=str)}"
    )
