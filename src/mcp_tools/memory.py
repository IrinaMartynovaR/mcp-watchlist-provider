import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from statistics import fmean
from typing import Any

from langfuse import observe

# Mem0 по умолчанию отправляет анонимную телеметрию в PostHog; флаг читается
# один раз при импорте модуля mem0, поэтому выключаем его заранее. setdefault
# позволяет сознательно включить телеметрию обычной env-переменной.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory

from app.settings import backend_settings
from domain.models import Category, RecommendationFeedback
from llm_core.settings import llm_settings
from mcp_tools.media import candidate_key
from mcp_tools.rag import candidate_text
from mcp_tools.settings import tool_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_memory_client() -> Memory:
    """Возвращает ленивый singleton Mem0-клиента графовой памяти.

    Клиент строится при первом обращении, а не на import-время,
    чтобы выключенная фича никогда не инициировала подключение
    к Memgraph/Chroma при старте приложения.

    Returns:
        Инициализированный Mem0-клиент с Chroma vector store
        и Memgraph graph store.
    """
    return Memory.from_config(
        {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "watchquest_preferences",
                    "path": str(backend_settings.mem0_vector_store_dir),
                },
            },
            "graph_store": {
                "provider": "memgraph",
                "config": {
                    "url": tool_settings.memgraph_url,
                    # Mem0 требует непустые credentials; Memgraph без включённой
                    # авторизации игнорирует их, поэтому подставляем безопасный fallback.
                    "username": tool_settings.memgraph_username or "memgraph",
                    "password": tool_settings.memgraph_password or "memgraph",
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_settings.normalized_model,
                    "openai_base_url": llm_settings.normalized_base_url,
                    "api_key": llm_settings.normalized_api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": llm_settings.normalized_embedding_model,
                    "openai_base_url": llm_settings.normalized_base_url,
                    "api_key": llm_settings.normalized_api_key,
                    # Обязательное поле для memgraph graph store (Mem0 читает его
                    # без fallback); 1536 — размерность text-embedding-3-small.
                    "embedding_dims": tool_settings.memory_embedding_dims,
                },
            },
        }
    )


def record_preference_note_data(feedback: RecommendationFeedback, candidate: dict[str, Any] | None) -> None:
    """Сохраняет заметку предпочтения по feedback-событию в графовую память.

    Функция спроектирована как fail-soft: сбой Mem0/Memgraph/Chroma логируется
    и не пробрасывается наверх, чтобы не ломать обработку feedback.

    Args:
        feedback: Feedback-событие пользователя.
        candidate: Кандидат рекомендации, к которому относится feedback.
    """
    if not tool_settings.memory_enabled:
        return
    try:
        text = _note_text(feedback, candidate)
        if not text:
            logger.debug(
                "Preference note skipped: empty text",
                extra={"recommendation_id": feedback.recommendation_id, "action": feedback.action},
            )
            return
        weight = tool_settings.feedback_weights[feedback.action]
        _get_memory_client().add(
            text,
            user_id=tool_settings.mem0_user_id,
            metadata={"weight": weight, "category": feedback.category, "action": feedback.action},
        )
        logger.info(
            "Preference note recorded",
            extra={"action": feedback.action, "weight": weight, "user_id": tool_settings.mem0_user_id},
        )
    except Exception:
        logger.warning(
            "Failed to record preference note",
            extra={"recommendation_id": feedback.recommendation_id, "action": feedback.action},
            exc_info=True,
        )


def record_import_note_data(
    text: str,
    weight: float,
    category: Category,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Записывает заметку предпочтения из внешнего источника (например, MyShows) в графовую память.

    Fail-soft: сбой Mem0/Memgraph логируется и не пробрасывается наверх.

    Args:
        text: Текст заметки предпочтения.
        weight: Знаковый вес заметки (положительный — нравится, отрицательный — не нравится).
        category: Категория, к которой относится заметка.
        extra_metadata: Дополнительные метаданные (например, источник импорта).
    """
    if not tool_settings.memory_enabled or not text:
        return
    try:
        metadata: dict[str, Any] = {"weight": weight, "category": category}
        if extra_metadata:
            metadata.update(extra_metadata)
        _get_memory_client().add(text, user_id=tool_settings.mem0_user_id, metadata=metadata)
    except Exception:
        logger.warning("Failed to record imported preference note", exc_info=True)


@observe(name="semantic_memory_score", as_type="tool")
def semantic_memory_scores_data(
    candidates: list[dict[str, Any]],
    top_k: int = tool_settings.memory_top_k,
) -> dict[str, float]:
    """Считает знаковый memory-score кандидатов по графовой памяти предпочтений.

    Для каждого кандидата запрашиваются top-k релевантных воспоминаний Mem0,
    score — среднее произведений релевантности воспоминания на его знаковый вес.
    Функция спроектирована как fail-soft: любая ошибка приводит
    к пустому результату, а не к исключению.

    Args:
        candidates: RSS-кандидаты рекомендации.
        top_k: Число ближайших воспоминаний на кандидата.

    Returns:
        Словарь candidate_key -> знаковый score либо пустой словарь,
        если фича выключена, кандидатов или воспоминаний нет, или произошла ошибка.
    """
    if not tool_settings.memory_enabled:
        return {}
    if not candidates or top_k <= 0:
        return {}

    try:
        client = _get_memory_client()

        def _search(candidate: dict[str, Any]) -> dict[str, Any]:
            result: dict[str, Any] = client.search(
                query=candidate_text(candidate),
                user_id=tool_settings.mem0_user_id,
                limit=top_k,
            )
            return result

        # Каждый поиск — независимый round-trip к Memgraph/Chroma; распараллеливаем
        # по кандидатам через потоки, чтобы не платить N последовательных задержек.
        with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
            responses = list(executor.map(_search, candidates))

        scores: dict[str, float] = {}
        for candidate, response in zip(candidates, responses, strict=True):
            weighted = [
                float(item.get("score") or 0.0) * float((item.get("metadata") or {}).get("weight") or 0.0)
                for item in response.get("results", [])
                if isinstance(item, dict)
            ]
            if not weighted:
                continue
            scores[candidate_key(candidate)] = float(fmean(weighted))
        logger.info(
            "Semantic memory scores computed",
            extra={"candidate_count": len(candidates), "scored_count": len(scores), "top_k": top_k},
        )
        return scores
    except Exception:
        logger.warning("Semantic memory scoring failed", extra={"candidate_count": len(candidates)}, exc_info=True)
        return {}


def _note_text(feedback: RecommendationFeedback, candidate: dict[str, Any] | None) -> str:
    """Собирает текст заметки предпочтения.

    Args:
        feedback: Feedback-событие пользователя.
        candidate: Кандидат рекомендации, если он известен.

    Returns:
        Текст для записи в память либо пустую строку.
    """
    if candidate is not None:
        text = candidate_text(candidate)
        if text:
            return text
    parts = [feedback.title or "", feedback.query]
    return " ".join(part.strip() for part in parts if part.strip())
