import logging
from collections import deque
from typing import Any, TypedDict, cast
from uuid import uuid4

from langfuse import observe
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from domain.models import Category, category_matches
from llm_core.prompts.recommendations import build_feed_recommendation_prompt
from llm_core.settings import llm_settings
from mcp_tools.feedback import save_recommendation_result
from mcp_tools.llm import ask_llm_data
from mcp_tools.media import (
    candidate_key,
    fetch_latest_items_data,
    get_profile_data,
    list_watchlist_data,
    search_cached_items_data,
)
from mcp_tools.memory import semantic_memory_scores_data
from mcp_tools.rag import semantic_search_data
from mcp_tools.settings import tool_settings

logger = logging.getLogger(__name__)

# Веса типа записи в одном масштабе с learned_preferences/feedback_weights:
# обзор конкретного тайтла — сильный кандидат, шум (скидки/промо) — вниз.
_KIND_PREFERENCE_WEIGHTS = {"review": 1, "noise": -2}


class RecommendationState(TypedDict, total=False):
    """Описывает состояние LangGraph-конвейера рекомендации."""
    query: str
    category: Category
    refresh: bool
    limit: int
    limit_per_source: int
    fetched_count: int
    candidates: list[dict[str, Any]]
    retrieval_mode: str
    fallback_used: bool
    memory_used: bool
    profile: dict[str, Any]
    watchlist: list[dict[str, Any]]
    prompt: str
    llm_result: dict[str, str]
    llm_error: str | None
    result: dict[str, Any]


@observe(name="recommend_media", as_type="chain")
def recommend_media_data(
    query: str,
    category: Category = "all",
    refresh: bool = True,
    limit: int = tool_settings.recommendation_limit,
    limit_per_source: int = tool_settings.recommendation_source_limit,
) -> dict[str, Any]:
    """Собирает рекомендацию из RSS-кандидатов и LLM-ответа.

    Оркестрация выполнена LangGraph-конвейером: семантический retrieval
    (если включён) с откатом на наивный подстрочный поиск, персональное
    ранжирование с семантической памятью предпочтений, генерация и сохранение.

    Args:
        query: Пользовательский запрос.
        category: Категория поиска.
        refresh: Нужно ли обновить RSS-кеш перед подбором.
        limit: Максимальное число кандидатов для передачи в LLM.
        limit_per_source: Лимит RSS-записей на один источник.

    Returns:
        Словарь с кандидатами, LLM-рекомендацией и метаданными запроса.
    """
    state: RecommendationState = {
        "query": query,
        "category": category,
        "refresh": refresh,
        "limit": limit,
        "limit_per_source": limit_per_source,
    }
    final_state = _RECOMMENDATION_GRAPH.invoke(state)
    return cast(dict[str, Any], final_state["result"])


def _refresh_node(state: RecommendationState) -> RecommendationState:
    """Обновляет RSS-кеш, если запрошен refresh.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с числом загруженных RSS-записей.
    """
    fetched_count = 0
    if state["refresh"]:
        fetched_count = len(
            fetch_latest_items_data(category=state["category"], limit_per_source=state["limit_per_source"])
        )
    return {"fetched_count": fetched_count}


@observe(name="retrieve_candidates", as_type="tool")
def _retrieve_node(state: RecommendationState) -> RecommendationState:
    """Подбирает кандидатов: семантический retrieval с наивным откатом.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с кандидатами и режимом retrieval.
    """
    query = state["query"]
    category = state["category"]
    limit = state["limit"]

    semantic_candidates = semantic_search_data(query=query, category=category, limit=limit)
    if semantic_candidates:
        return {"candidates": semantic_candidates, "retrieval_mode": "semantic", "fallback_used": False}

    naive_candidates = _search_recommendation_candidates(query=query, category=category, limit=limit)
    return {"candidates": naive_candidates, "retrieval_mode": "naive", "fallback_used": False}


def _route_after_retrieve(state: RecommendationState) -> str:
    """Выбирает следующий узел после retrieve.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Имя узла: `load_profile`, если кандидаты найдены, иначе `fallback`.
    """
    return "load_profile" if state.get("candidates") else "fallback"


def _fallback_node(state: RecommendationState) -> RecommendationState:
    """Подбирает свежих кандидатов, когда основной retrieval пуст.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с fallback-кандидатами.
    """
    recent_candidates = search_cached_items_data(
        query="",
        category=state["category"],
        days=tool_settings.cache_search_days,
        limit=state["limit"] * tool_settings.recommendation_fallback_candidate_multiplier,
    )
    candidates = _prefer_exact_category(recent_candidates, category=state["category"], limit=state["limit"])
    logger.info(
        "Recommendation fallback candidates selected",
        extra={"query": state["query"], "category": state["category"], "candidate_count": len(candidates)},
    )
    return {"candidates": candidates, "fallback_used": True}


def _load_profile_node(state: RecommendationState) -> RecommendationState:
    """Загружает профиль пользовательских предпочтений.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с профилем.
    """
    return {"profile": get_profile_data()}


@observe(name="rank_candidates", as_type="tool")
def _rank_node(state: RecommendationState) -> RecommendationState:
    """Ранжирует кандидатов по learned preferences и семантической памяти.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с отранжированными кандидатами.
    """
    memory_scores = semantic_memory_scores_data(candidates=state["candidates"], top_k=tool_settings.memory_top_k)
    ranked = _rank_candidates_by_learned_preferences(state["candidates"], state["profile"], memory_scores)
    return {"candidates": ranked, "memory_used": bool(memory_scores)}


def _load_watchlist_node(state: RecommendationState) -> RecommendationState:
    """Загружает watchlist пользователя.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с watchlist.
    """
    return {"watchlist": list_watchlist_data(media_type="all", status="all")}


def _build_prompt_node(state: RecommendationState) -> RecommendationState:
    """Собирает LLM-prompt рекомендации.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с готовым prompt.
    """
    prompt = _recommend_media_prompt(
        query=state["query"],
        category=state["category"],
        profile=state["profile"],
        watchlist=state["watchlist"],
        candidates=state["candidates"],
    )
    return {"prompt": prompt}


def _generate_node(state: RecommendationState) -> RecommendationState:
    """Запрашивает LLM-рекомендацию с fail-soft откатом.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с LLM-результатом и ошибкой, если она была.
    """
    llm_error: str | None = None
    try:
        llm = ask_llm_data(state["prompt"])
    except RuntimeError as exc:
        llm_error = str(exc)
        llm = _fallback_llm_result(error=llm_error, candidates=state["candidates"])
        logger.warning(
            "Recommendation LLM fallback used",
            extra={"query": state["query"], "category": state["category"], "error": llm_error},
        )
    return {"llm_result": llm, "llm_error": llm_error}


def _persist_node(state: RecommendationState) -> RecommendationState:
    """Собирает итоговый результат рекомендации и сохраняет его в историю.

    Args:
        state: Текущее состояние конвейера.

    Returns:
        Обновление состояния с итоговым результатом.
    """
    candidates = state["candidates"]
    llm = state["llm_result"]
    llm_error = state["llm_error"]
    refresh = state["refresh"]
    fallback_used = state.get("fallback_used", False)
    semantic_search_used = state.get("retrieval_mode") == "semantic"
    semantic_memory_used = state.get("memory_used", False)
    logger.info(
        "Recommendation generated",
        extra={
            "query": state["query"],
            "category": state["category"],
            "refreshed": refresh,
            "fetched_count": state["fetched_count"],
            "candidate_count": len(candidates),
            "provider": llm["provider"],
            "model": llm["model"],
        },
    )

    result = {
        "id": uuid4().hex,
        "query": state["query"],
        "category": state["category"],
        "refreshed": refresh,
        "fetched_count": state["fetched_count"],
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
            "semantic_search": semantic_search_used,
            "semantic_memory_score": semantic_memory_used,
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
            "semantic_search": semantic_search_used,
            "semantic_memory_score": semantic_memory_used,
            "add_to_watchlist": False,
            "list_watchlist": True,
            "rate_watchlist_item": False,
            "recommend_media": True,
        },
    }
    if llm_error:
        result["llm_error"] = llm_error
    save_recommendation_result(result)
    return {"result": result}


_RecommendationGraph = CompiledStateGraph[RecommendationState, None, RecommendationState, RecommendationState]


def _build_recommendation_graph() -> _RecommendationGraph:
    """Собирает и компилирует LangGraph-конвейер рекомендации.

    Returns:
        Скомпилированный граф рекомендации.
    """
    graph: StateGraph[RecommendationState, None, RecommendationState, RecommendationState] = StateGraph(
        RecommendationState
    )
    graph.add_node("refresh", _refresh_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("fallback", _fallback_node)
    graph.add_node("load_profile", _load_profile_node)
    graph.add_node("rank", _rank_node)
    graph.add_node("load_watchlist", _load_watchlist_node)
    graph.add_node("build_prompt", _build_prompt_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("persist", _persist_node)

    graph.add_edge(START, "refresh")
    graph.add_edge("refresh", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"load_profile": "load_profile", "fallback": "fallback"},
    )
    graph.add_edge("fallback", "load_profile")
    graph.add_edge("load_profile", "rank")
    graph.add_edge("rank", "load_watchlist")
    graph.add_edge("load_watchlist", "build_prompt")
    graph.add_edge("build_prompt", "generate")
    graph.add_edge("generate", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


_RECOMMENDATION_GRAPH = _build_recommendation_graph()


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
            key = candidate_key(item)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(item)
            if len(candidates) >= limit:
                return _interleave_by_source(candidates)

    return _interleave_by_source(candidates)


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
        return _interleave_by_source(candidates)[:limit]

    def bucket(item: dict[str, Any]) -> str:
        item_category = str(item.get("category") or "")
        if item_category != "mixed" and category_matches(item_category, category):
            return "exact"
        return "mixed" if item_category == "mixed" else "other"

    exact = _interleave_by_source([item for item in candidates if bucket(item) == "exact"])
    mixed = _interleave_by_source([item for item in candidates if bucket(item) == "mixed"])
    other = _interleave_by_source([item for item in candidates if bucket(item) == "other"])
    return [*exact, *mixed, *other][:limit]


def _interleave_by_source(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Чередует кандидатов по источникам round-robin, не меняя их набор.

    Кандидаты группируются по полю `source` с сохранением относительного
    порядка внутри каждой группы, затем группы обходятся по кругу в порядке
    первого появления источника — так один источник не может занять
    несколько позиций подряд, пока у других источников остаются кандидаты.

    Не путать с `rag.py::_diversify_by_source`: там top-limit отбор из уже
    отранжированного по релевантности списка, здесь — чистая пересортировка
    recency-ordered кандидатов наивного/fallback путей без отбрасывания.

    Args:
        candidates: RSS-кандидаты в исходном порядке.

    Returns:
        Тот же набор кандидатов, переупорядоченный по источникам.
    """
    groups: dict[str, deque[dict[str, Any]]] = {}
    for item in candidates:
        source = str(item.get("source") or "")
        groups.setdefault(source, deque()).append(item)

    queues = list(groups.values())
    interleaved: list[dict[str, Any]] = []
    while queues:
        for queue in queues:
            interleaved.append(queue.popleft())
        queues = [queue for queue in queues if queue]
    return interleaved


def _rank_candidates_by_learned_preferences(
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
    memory_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Ранжирует кандидатов по learned_preferences пользователя.

    Args:
        candidates: RSS-кандидаты для LLM.
        profile: Профиль пользователя с накопленными feedback-весами.
        memory_scores: Знаковые memory-score по candidate_key.

    Returns:
        Кандидаты, отсортированные по персональному score.
    """
    learned = profile.get("learned_preferences", {})
    if not isinstance(learned, dict):
        return candidates

    scores = memory_scores or {}
    scored_candidates = [
        _candidate_with_preference_score(candidate, learned, scores.get(candidate_key(candidate)))
        for candidate in candidates
    ]
    return sorted(scored_candidates, key=lambda item: int(item.get("preference_score", 0)), reverse=True)


def _candidate_with_preference_score(
    candidate: dict[str, Any],
    learned: dict[str, Any],
    memory_score: float | None = None,
) -> dict[str, Any]:
    """Добавляет кандидату score и причины совпадения с learned_preferences.

    Args:
        candidate: RSS-кандидат.
        learned: Накопленные веса предпочтений.
        memory_score: Знаковый score семантической памяти предпочтений.

    Returns:
        Копию кандидата с `preference_score`, `preference_reasons`
        и информационным `preference_memory_score`.
    """
    score = 0
    reasons: list[str] = []
    score += _score_field(learned, "categories", str(candidate.get("category") or ""), reasons)
    score += _score_field(learned, "sources", str(candidate.get("source") or ""), reasons)

    kind = str(candidate.get("kind") or "")
    kind_score = _KIND_PREFERENCE_WEIGHTS.get(kind, 0)
    if kind_score:
        score += kind_score
        reasons.append(f"kind:{kind}:{kind_score:+d}")

    tags = candidate.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            score += _score_field(learned, "tags", str(tag).strip().lower(), reasons)

    if memory_score:
        score += round(memory_score)
        reasons.append(f"memory:{memory_score:+.2f}")

    result = dict(candidate)
    result["preference_score"] = score
    result["preference_reasons"] = reasons
    result["preference_memory_score"] = memory_score or 0.0
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
