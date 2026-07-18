import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict, cast
from uuid import uuid4

from langfuse import observe
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import RuntimeConfig
from domain.models import Category
from llm_core.prompts.recommendations import build_feed_recommendation_prompt
from mcp_tools.media import candidate_key
from mcp_tools.query_analysis import item_matches_medium
from mcp_tools.recommendation_candidates import (
    candidate_with_preference_score as _candidate_with_preference_score,
)
from mcp_tools.recommendation_candidates import (
    interleave_by_source as _interleave_by_source,
)
from mcp_tools.recommendation_candidates import (
    prefer_exact_category,
    query_variants,
    rank_candidates_by_learned_preferences,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RecommendationDependencies",
    "RecommendationService",
    "_candidate_with_preference_score",
    "_interleave_by_source",
]


@dataclass(frozen=True)
class RecommendationDependencies:
    """Описывает внешние зависимости recommendation pipeline.

    Attributes:
        fetch_latest: Обновление RSS-кеша.
        search_cached: Поиск кандидатов в RSS-кеше.
        get_profile: Загрузка профиля пользователя.
        list_watchlist: Загрузка watchlist.
        analyze_query: Извлечение структурных ограничений из запроса.
        semantic_search: Семантический поиск кандидатов.
        semantic_memory_scores: Расчёт знаковых memory-score.
        ask_llm: Вызов основной LLM.
        save_result: Сохранение recommendation-записи.
    """

    fetch_latest: Callable[..., list[dict[str, Any]]]
    search_cached: Callable[..., list[dict[str, Any]]]
    get_profile: Callable[[], dict[str, Any]]
    list_watchlist: Callable[..., list[dict[str, Any]]]
    analyze_query: Callable[..., dict[str, Any]]
    semantic_search: Callable[..., list[dict[str, Any]]]
    semantic_memory_scores: Callable[..., dict[str, float]]
    ask_llm: Callable[..., dict[str, str]]
    save_result: Callable[[dict[str, Any]], Any]


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


_RecommendationGraph = CompiledStateGraph[RecommendationState, None, RecommendationState, RecommendationState]


class RecommendationService:
    """Оркестрирует recommendation pipeline без скрытых глобальных зависимостей.

    Args:
        config: Единый runtime config приложения.
        dependencies: Уже собранные внешние зависимости pipeline.
    """

    def __init__(self, config: RuntimeConfig, dependencies: RecommendationDependencies) -> None:
        """Создаёт сервис и компилирует принадлежащий ему LangGraph.

        Args:
            config: Единый runtime config приложения.
            dependencies: Внешние зависимости recommendation pipeline.
        """
        self.config = config
        self.dependencies = dependencies
        self._graph = self._build_graph()

    @observe(name="recommend_media", as_type="chain")
    def recommend(
        self,
        query: str,
        category: Category = "all",
        refresh: bool = True,
        limit: int | None = None,
        limit_per_source: int | None = None,
    ) -> dict[str, Any]:
        """Собирает рекомендацию из RSS-кандидатов и LLM-ответа.

        Args:
            query: Пользовательский запрос.
            category: Категория поиска.
            refresh: Нужно ли обновить RSS-кеш перед подбором.
            limit: Максимальное число кандидатов для LLM.
            limit_per_source: Лимит RSS-записей на источник.

        Returns:
            Рекомендация, кандидаты и диагностические метаданные.
        """
        tools = self.config.tools
        state: RecommendationState = {
            "query": query,
            "category": category,
            "refresh": refresh,
            "limit": limit if limit is not None else tools.recommendation_limit,
            "limit_per_source": (
                limit_per_source if limit_per_source is not None else tools.recommendation_source_limit
            ),
        }
        final_state = self._graph.invoke(state)
        return cast(dict[str, Any], final_state["result"])

    def _refresh_node(self, state: RecommendationState) -> RecommendationState:
        """Обновляет RSS-кеш, когда refresh включён."""
        fetched_count = 0
        if state["refresh"]:
            fetched_count = len(
                self.dependencies.fetch_latest(
                    category=state["category"],
                    limit_per_source=state["limit_per_source"],
                )
            )
        return {"fetched_count": fetched_count}

    @observe(name="retrieve_candidates", as_type="tool")
    def _retrieve_node(self, state: RecommendationState) -> RecommendationState:
        """Подбирает кандидатов семантически с наивным откатом."""
        analysis = self.dependencies.analyze_query(query=state["query"])
        medium = analysis.get("medium")
        semantic_candidates = self.dependencies.semantic_search(
            query=state["query"],
            category=state["category"],
            limit=state["limit"],
            medium=medium,
        )
        if semantic_candidates:
            return {"candidates": semantic_candidates, "retrieval_mode": "semantic", "fallback_used": False}

        naive_candidates = self._search_candidates(state["query"], state["category"], state["limit"])
        if medium:
            naive_candidates = [
                {**item, "query_match_medium": medium} if item_matches_medium(item, medium) else item
                for item in naive_candidates
            ]
        return {"candidates": naive_candidates, "retrieval_mode": "naive", "fallback_used": False}

    @staticmethod
    def _route_after_retrieve(state: RecommendationState) -> str:
        """Выбирает основной или fallback-маршрут после retrieval."""
        return "load_profile" if state.get("candidates") else "fallback"

    def _fallback_node(self, state: RecommendationState) -> RecommendationState:
        """Подбирает свежие записи, когда основной retrieval пуст."""
        tools = self.config.tools
        recent_candidates = self.dependencies.search_cached(
            query="",
            category=state["category"],
            days=tools.cache_search_days,
            limit=state["limit"] * tools.recommendation_fallback_candidate_multiplier,
        )
        candidates = prefer_exact_category(recent_candidates, state["category"], state["limit"])
        logger.info(
            "Recommendation fallback candidates selected",
            extra={"query": state["query"], "category": state["category"], "candidate_count": len(candidates)},
        )
        return {"candidates": candidates, "fallback_used": True}

    def _load_profile_node(self, state: RecommendationState) -> RecommendationState:
        """Загружает профиль пользовательских предпочтений."""
        del state
        return {"profile": self.dependencies.get_profile()}

    @observe(name="rank_candidates", as_type="tool")
    def _rank_node(self, state: RecommendationState) -> RecommendationState:
        """Ранжирует кандидатов по профилю и семантической памяти."""
        memory_scores = self.dependencies.semantic_memory_scores(
            candidates=state["candidates"],
            top_k=self.config.tools.memory_top_k,
        )
        ranked = rank_candidates_by_learned_preferences(
            state["candidates"],
            state["profile"],
            self.config.tools,
            memory_scores,
        )
        return {"candidates": ranked, "memory_used": bool(memory_scores)}

    def _load_watchlist_node(self, state: RecommendationState) -> RecommendationState:
        """Загружает полный watchlist пользователя."""
        del state
        return {"watchlist": self.dependencies.list_watchlist(media_type="all", status="all")}

    @staticmethod
    def _build_prompt_node(state: RecommendationState) -> RecommendationState:
        """Собирает финальный prompt рекомендации."""
        prompt = build_feed_recommendation_prompt(
            query=state["query"],
            category=state["category"],
            profile=state["profile"],
            watchlist=state["watchlist"],
            candidates=state["candidates"],
        )
        return {"prompt": prompt}

    def _generate_node(self, state: RecommendationState) -> RecommendationState:
        """Генерирует LLM-ответ с fail-soft fallback."""
        llm_error: str | None = None
        try:
            llm = self.dependencies.ask_llm(state["prompt"])
        except RuntimeError as exc:
            llm_error = str(exc)
            llm = self._fallback_llm_result(llm_error, state["candidates"])
            logger.warning(
                "Recommendation LLM fallback used",
                extra={"query": state["query"], "category": state["category"], "error": llm_error},
            )
        return {"llm_result": llm, "llm_error": llm_error}

    def _persist_node(self, state: RecommendationState) -> RecommendationState:
        """Собирает итоговый payload и сохраняет recommendation-запись."""
        candidates = state["candidates"]
        llm = state["llm_result"]
        llm_error = state["llm_error"]
        refresh = state["refresh"]
        fallback_used = state.get("fallback_used", False)
        semantic_search_used = state.get("retrieval_mode") == "semantic"
        semantic_memory_used = state.get("memory_used", False)
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
        self.dependencies.save_result(result)
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
        return {"result": result}

    def _build_graph(self) -> _RecommendationGraph:
        """Создаёт LangGraph, принадлежащий текущему service instance.

        Returns:
            Скомпилированный recommendation-граф.
        """
        graph: StateGraph[RecommendationState, None, RecommendationState, RecommendationState] = StateGraph(
            RecommendationState
        )
        graph.add_node("refresh", self._refresh_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("fallback", self._fallback_node)
        graph.add_node("load_profile", self._load_profile_node)
        graph.add_node("rank", self._rank_node)
        graph.add_node("load_watchlist", self._load_watchlist_node)
        graph.add_node("build_prompt", self._build_prompt_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("persist", self._persist_node)
        graph.add_edge(START, "refresh")
        graph.add_edge("refresh", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
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

    def _search_candidates(self, query: str, category: Category, limit: int) -> list[dict[str, Any]]:
        """Ищет и дедуплицирует кандидатов по вариантам запроса."""
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for variant in query_variants(query, self.config.tools):
            for item in self.dependencies.search_cached(
                query=variant,
                category=category,
                days=self.config.tools.cache_search_days,
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

    def _fallback_llm_result(self, error: str, candidates: list[dict[str, Any]]) -> dict[str, str]:
        """Строит безопасный ответ при недоступности LLM."""
        titles = [
            str(candidate.get("title") or "").strip()
            for candidate in candidates[:3]
            if str(candidate.get("title") or "").strip()
        ]
        context = "\n".join(f"- {title}" for title in titles)
        details = f"\n\nRSS-кандидаты для ручной проверки:\n{context}" if context else ""
        llm = self.config.llm
        return {
            "provider": llm.normalized_provider,
            "model": llm.normalized_model,
            "response": (
                "LLM сейчас не смогла собрать финальную рекомендацию: "
                f"{error}. RSS уже проверен, поэтому можно посмотреть найденные кандидаты ниже.{details}"
            ),
        }
