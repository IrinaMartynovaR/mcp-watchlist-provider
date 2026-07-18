import logging
import os
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean
from typing import Any

from langfuse import observe

# Mem0 читает флаг телеметрии во время импорта. Пользователь по-прежнему может
# сознательно переопределить значение через окружение.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory

from app.config import RuntimeConfig
from domain.models import Category, RecommendationFeedback
from mcp_tools.media import candidate_key
from mcp_tools.rag import candidate_text

logger = logging.getLogger(__name__)


class MemoryService:
    """Инкапсулирует ленивый Mem0-клиент и операции графовой памяти.

    Args:
        config: Единый runtime config приложения.
        client: Необязательный готовый Mem0-совместимый клиент для тестов.
    """

    def __init__(self, config: RuntimeConfig, client: Any | None = None) -> None:
        """Создаёт сервис с ленивым или заранее переданным клиентом.

        Args:
            config: Единый runtime config приложения.
            client: Необязательный Mem0-совместимый клиент.
        """
        self.config = config
        self._client = client

    def _get_client(self) -> Any:
        """Лениво создаёт и кеширует Mem0-клиент внутри service instance.

        Returns:
            Mem0-совместимый клиент.
        """
        if self._client is None:
            self._client = Memory.from_config(self._memory_config())
        return self._client

    def _memory_config(self) -> dict[str, Any]:
        """Собирает конфигурацию Mem0 из единого runtime config.

        Returns:
            Конфигурация vector, graph, LLM и embedding providers.
        """
        backend = self.config.backend
        tools = self.config.tools
        llm = self.config.llm
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "watchquest_preferences",
                    "path": str(backend.mem0_vector_store_dir),
                },
            },
            "graph_store": {
                "provider": "memgraph",
                "config": {
                    "url": tools.memgraph_url,
                    "username": tools.memgraph_username or "memgraph",
                    "password": tools.memgraph_password or "memgraph",
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm.normalized_model,
                    "openai_base_url": llm.normalized_base_url,
                    "api_key": llm.normalized_api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": llm.normalized_embedding_model,
                    "openai_base_url": llm.normalized_base_url,
                    "api_key": llm.normalized_api_key,
                    "embedding_dims": tools.memory_embedding_dims,
                },
            },
        }

    def record_preference(
        self,
        feedback: RecommendationFeedback,
        candidate: dict[str, Any] | None,
    ) -> None:
        """Fail-soft сохраняет feedback-сигнал в графовую память.

        Args:
            feedback: Пользовательский feedback.
            candidate: Кандидат, к которому относится feedback.
        """
        tools = self.config.tools
        if not tools.memory_enabled:
            return
        try:
            text = _note_text(feedback, candidate)
            if not text:
                logger.debug(
                    "Preference note skipped: empty text",
                    extra={"recommendation_id": feedback.recommendation_id, "action": feedback.action},
                )
                return
            weight = tools.feedback_weights[feedback.action]
            self._get_client().add(
                text,
                user_id=tools.mem0_user_id,
                metadata={"weight": weight, "category": feedback.category, "action": feedback.action},
            )
            logger.info(
                "Preference note recorded",
                extra={"action": feedback.action, "weight": weight, "user_id": tools.mem0_user_id},
            )
        except Exception:
            logger.warning(
                "Failed to record preference note",
                extra={"recommendation_id": feedback.recommendation_id, "action": feedback.action},
                exc_info=True,
            )

    def record_import(
        self,
        text: str,
        weight: float,
        category: Category,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Fail-soft сохраняет сигнал из внешнего источника.

        Args:
            text: Текст заметки предпочтения.
            weight: Знаковый вес заметки.
            category: Категория заметки.
            extra_metadata: Дополнительные метаданные источника.
        """
        tools = self.config.tools
        if not tools.memory_enabled or not text:
            return
        try:
            metadata: dict[str, Any] = {"weight": weight, "category": category}
            if extra_metadata:
                metadata.update(extra_metadata)
            self._get_client().add(text, user_id=tools.mem0_user_id, metadata=metadata)
        except Exception:
            logger.warning("Failed to record imported preference note", exc_info=True)

    @observe(name="semantic_memory_score", as_type="tool")
    def semantic_scores(self, candidates: list[dict[str, Any]], top_k: int | None = None) -> dict[str, float]:
        """Считает знаковый memory-score для RSS-кандидатов.

        Args:
            candidates: Кандидаты рекомендации.
            top_k: Число ближайших воспоминаний; по умолчанию берётся из config.

        Returns:
            Словарь ``candidate_key -> score`` либо пустой словарь при сбое.
        """
        tools = self.config.tools
        memory_limit = top_k if top_k is not None else tools.memory_top_k
        if not tools.memory_enabled or not candidates or memory_limit <= 0:
            return {}

        try:
            client = self._get_client()

            def search(candidate: dict[str, Any]) -> dict[str, Any]:
                result: dict[str, Any] = client.search(
                    query=candidate_text(candidate),
                    user_id=tools.mem0_user_id,
                    limit=memory_limit,
                )
                return result

            with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
                responses = list(executor.map(search, candidates))

            scores: dict[str, float] = {}
            for candidate, response in zip(candidates, responses, strict=True):
                weighted = [
                    float(item.get("score") or 0.0) * float((item.get("metadata") or {}).get("weight") or 0.0)
                    for item in response.get("results", [])
                    if isinstance(item, dict)
                ]
                if weighted:
                    scores[candidate_key(candidate)] = float(fmean(weighted))
            logger.info(
                "Semantic memory scores computed",
                extra={"candidate_count": len(candidates), "scored_count": len(scores), "top_k": memory_limit},
            )
            return scores
        except Exception:
            logger.warning(
                "Semantic memory scoring failed",
                extra={"candidate_count": len(candidates)},
                exc_info=True,
            )
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
