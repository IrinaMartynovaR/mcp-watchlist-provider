from typing import Any

from app.observability import observe
from app.settings import CACHE_FILE
from domain.models import Category
from domain.storage.json_store import read_json
from llm_core.client import create_llm_client
from llm_core.prompts.recommendations import SYSTEM_PROMPT, build_context_recommendation_prompt
from llm_core.schemas import ChatMessage
from llm_core.settings import LLM_MODEL, LLM_PROVIDER
from mcp_tools.media import get_profile_data, list_watchlist_data


@observe("ask_llm")
def ask_llm_data(prompt: str, system: str | None = None) -> dict[str, Any]:
    client = create_llm_client()
    messages: list[ChatMessage] = [
        {"role": "system", "content": system or SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = client.chat(messages)
    return {"provider": LLM_PROVIDER, "model": LLM_MODEL, "response": response}


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
    return build_context_recommendation_prompt(
        query=query,
        profile=profile,
        watchlist=watchlist,
        candidates=candidates,
    )

