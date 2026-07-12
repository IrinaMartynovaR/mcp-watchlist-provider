import logging
import math
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langfuse import observe

from domain.models import Category, category_matches
from llm_core.embeddings import create_embeddings
from llm_core.prompts.recommendations import HYDE_SYSTEM_PROMPT, build_hyde_prompt
from mcp_tools.llm import ask_llm_data
from mcp_tools.media import candidate_key, search_cached_items_data
from mcp_tools.settings import tool_settings

logger = logging.getLogger(__name__)

_EXACT_CATEGORY_BONUS = 0.05
# Обзоры конкретных тайтлов — лучший материал для рекомендации, шум
# (скидки/промо) не должен доходить до LLM даже при высокой similarity.
_KIND_SCORE_ADJUSTMENTS = {"review": 0.05, "noise": -0.2}


@observe(name="semantic_search", as_type="tool")
def semantic_search_data(
    query: str,
    category: Category = "all",
    limit: int = tool_settings.recommendation_limit,
) -> list[dict[str, Any]]:
    """Ищет RSS-кандидатов семантически через embedding-похожесть.

    Функция спроектирована как fail-soft: любая ошибка (сеть, авторизация,
    неожиданная схема ответа провайдера) приводит к пустому результату,
    а не к исключению — вызывающий код откатывается на наивный retrieval.

    Args:
        query: Пользовательский запрос.
        category: Категория поиска.
        limit: Максимальное число кандидатов.

    Returns:
        Семантически близкие RSS-кандидаты либо пустой список,
        если фича выключена, запрос пустой, пул пуст или произошла ошибка.
    """
    if not tool_settings.rag_enabled:
        return []
    if not query.strip():
        return []

    try:
        pool = search_cached_items_data(
            query="",
            category=category,
            days=tool_settings.cache_search_days,
            limit=limit * tool_settings.recommendation_fallback_candidate_multiplier,
        )
        if not pool:
            return []

        items_by_key: dict[str, dict[str, Any]] = {}
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for item in pool:
            key = candidate_key(item)
            if key in items_by_key:
                continue
            items_by_key[key] = item
            texts.append(candidate_text(item))
            metadatas.append({"candidate_key": key})

        store = InMemoryVectorStore(create_embeddings())
        store.add_texts(texts, metadatas=metadatas)
        scored_documents = store.similarity_search_with_score(query, k=len(texts))
        scored_by_key = _score_documents(scored_documents, items_by_key, category)

        if tool_settings.hyde_enabled:
            hyde_text = _generate_hyde_document(query, category)
            if hyde_text is not None:
                hyde_documents = store.similarity_search_with_score(hyde_text, k=len(texts))
                for key, (item, score) in _score_documents(hyde_documents, items_by_key, category).items():
                    if key not in scored_by_key or score > scored_by_key[key][1]:
                        scored_by_key[key] = (item, score)

        scored_candidates = sorted(scored_by_key.values(), key=lambda pair: pair[1], reverse=True)
        results = _diversify_by_source([item for item, _ in scored_candidates], limit=limit)
        logger.info(
            "Semantic search completed",
            extra={"query": query, "category": category, "result_count": len(results), "pool_size": len(pool)},
        )
        return results
    except Exception:
        logger.warning(
            "Semantic search failed; falling back to naive retrieval",
            extra={"query": query, "category": category},
            exc_info=True,
        )
        return []


def _generate_hyde_document(query: str, category: Category) -> str | None:
    """Генерирует гипотетический документ для cross-lingual semantic match.

    Fail-soft: любая ошибка LLM-генерации логируется и возвращает None,
    вызывающий код в этом случае откатывается на эмбеддинг сырого запроса.

    Args:
        query: Пользовательский запрос.
        category: Категория поиска.

    Returns:
        Текст гипотетического документа либо None при сбое генерации.
    """
    try:
        return str(ask_llm_data(build_hyde_prompt(query, category), system=HYDE_SYSTEM_PROMPT)["response"])
    except Exception:
        logger.warning(
            "HyDE document generation failed; falling back to raw query embedding",
            extra={"query": query, "category": category},
            exc_info=True,
        )
        return None


def _score_documents(
    scored_documents: list[tuple[Document, float]],
    items_by_key: dict[str, dict[str, Any]],
    category: Category,
) -> dict[str, tuple[dict[str, Any], float]]:
    """Сопоставляет scored-документы с кандидатами и применяет бонусы.

    Помимо exact-category бонуса к similarity-score добавляется поправка
    за тип записи (`kind`): бонус обзорам, штраф шуму (скидки/промо).

    Args:
        scored_documents: Пары (документ, score) из similarity search.
        items_by_key: Кандидаты пула, индексированные по candidate_key.
        category: Категория поиска для exact-match бонуса.

    Returns:
        Словарь candidate_key -> (кандидат, score с учётом бонусов).
    """
    scored_by_key: dict[str, tuple[dict[str, Any], float]] = {}
    for document, score in scored_documents:
        key = str(document.metadata.get("candidate_key"))
        matched = items_by_key.get(key)
        if matched is None:
            continue
        adjusted = score + _KIND_SCORE_ADJUSTMENTS.get(str(matched.get("kind") or ""), 0.0)
        item_category = str(matched.get("category") or "")
        if category != "all" and item_category != "mixed" and category_matches(item_category, category):
            adjusted += _EXACT_CATEGORY_BONUS
        scored_by_key[key] = (matched, adjusted)
    return scored_by_key


def _diversify_by_source(ranked_candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Отбирает top-`limit` кандидатов с мягким ограничением по источнику.

    Не путать с `recommendation.py::_interleave_by_source`: там чистая
    пересортировка recency-ordered кандидатов без отбрасывания, здесь —
    отбор top-limit из ранжированного списка с сохранением порядка релевантности.

    Кандидаты уже отсортированы по релевантности (убыв.). Жадно набираем
    результат, ограничивая число элементов одного источника капом, чтобы один
    источник не монополизировал выдачу; если после прохода с ограничением
    набралось меньше `limit`, недостающие места добираются лучшими по
    релевантности кандидатами без учёта ограничения (без этого при малом
    числе источников результат мог бы недобрать до `limit`).

    Args:
        ranked_candidates: Кандидаты, отсортированные по релевантности.
        limit: Максимальное число кандидатов в результате.

    Returns:
        До `limit` кандидатов: тот же набор, что и на входе, отобранный
        с приоритетом на релевантность и разнообразие источников.
    """
    if limit <= 0 or not ranked_candidates:
        return []

    distinct_sources = len({str(item.get("source") or "") for item in ranked_candidates})
    cap = max(1, math.ceil(limit / min(distinct_sources, limit)))

    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    per_source_count: dict[str, int] = {}

    for item in ranked_candidates:
        if len(selected) >= limit:
            break
        source = str(item.get("source") or "")
        if per_source_count.get(source, 0) < cap:
            selected.append(item)
            per_source_count[source] = per_source_count.get(source, 0) + 1
        else:
            deferred.append(item)

    if len(selected) < limit:
        selected.extend(deferred[: limit - len(selected)])
    return selected[:limit]


def candidate_text(item: dict[str, Any]) -> str:
    """Собирает текст RSS-кандидата для векторизации.

    Args:
        item: RSS-кандидат.

    Returns:
        Название тайтла, заголовок, аннотацию и tags, объединённые в один текст.
    """
    tags = item.get("tags", [])
    tags_text = " ".join(str(tag) for tag in tags) if isinstance(tags, list) else ""
    parts = [
        str(item.get("title_entity") or ""),
        str(item.get("title") or ""),
        str(item.get("summary") or ""),
        tags_text,
    ]
    return " ".join(part.strip() for part in parts if part.strip())
