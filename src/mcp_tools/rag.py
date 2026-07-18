import logging
import math
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langfuse import observe

from app.config import RuntimeConfig
from domain.models import Category, category_matches
from llm_core.prompts.recommendations import build_hyde_prompt
from mcp_tools.media import candidate_key
from mcp_tools.query_analysis import item_matches_medium

logger = logging.getLogger(__name__)

CachedSearch = Callable[..., list[dict[str, Any]]]
AskLLM = Callable[..., dict[str, Any]]


@observe(name="semantic_search", as_type="tool")
def semantic_search_data(
    query: str,
    config: RuntimeConfig,
    search_cached: CachedSearch,
    embeddings: Embeddings,
    ask_llm: AskLLM,
    category: Category = "all",
    limit: int | None = None,
    medium: str | None = None,
) -> list[dict[str, Any]]:
    """Ищет RSS-кандидатов семантически через embedding-похожесть.

    Функция спроектирована как fail-soft: любая ошибка (сеть, авторизация,
    неожиданная схема ответа провайдера) приводит к пустому результату,
    а не к исключению — вызывающий код откатывается на наивный retrieval.

    Args:
        query: Пользовательский запрос.
        config: Runtime-настройки приложения.
        search_cached: Инъецируемый поиск по RSS-кешу.
        embeddings: Настроенный embedding-адаптер.
        ask_llm: Инъецируемая функция LLM-вызова.
        category: Категория поиска.
        limit: Максимальное число кандидатов.
        medium: Медиум, явно названный в запросе (из `analyze_query_data`).

    Returns:
        Семантически близкие RSS-кандидаты либо пустой список,
        если фича выключена, запрос пустой, пул пуст или произошла ошибка.
    """
    tools = config.tools
    result_limit = limit if limit is not None else tools.recommendation_limit
    if not tools.rag_enabled:
        return []
    if not query.strip():
        return []

    try:
        # Пул не зависит от result_limit: семантический поиск должен видеть
        # весь свежий кеш, иначе при limit=5 искалось бы лишь по 15 новейшим
        # записям из сотен. Эмбеддинги кешируются, поэтому широкий пул дёшев.
        pool = search_cached(
            query="",
            category=category,
            days=tools.cache_search_days,
            limit=max(tools.rag_pool_limit, result_limit * tools.recommendation_fallback_candidate_multiplier),
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

        store = InMemoryVectorStore(embeddings)
        store.add_texts(texts, metadatas=metadatas)
        scored_documents = store.similarity_search_with_score(query, k=len(texts))
        scored_by_key = _score_documents(scored_documents, items_by_key, category, config, medium)

        if tools.hyde_enabled:
            hyde_text = _generate_hyde_document(query, category, config, ask_llm)
            if hyde_text is not None:
                hyde_documents = store.similarity_search_with_score(hyde_text, k=len(texts))
                for key, (item, score) in _score_documents(
                    hyde_documents, items_by_key, category, config, medium
                ).items():
                    if key not in scored_by_key or score > scored_by_key[key][1]:
                        scored_by_key[key] = (item, score)

        scored_candidates = sorted(scored_by_key.values(), key=lambda pair: pair[1], reverse=True)
        # semantic_score остаётся в кандидате: последующее персональное
        # ранжирование пересортирует список по вкусовым весам, и это поле —
        # единственный след близости кандидата к самому запросу (его видит LLM).
        # query_match_medium дополнительно бустится в персональном ранжировании.
        ranked_items = []
        for item, score in scored_candidates:
            enriched = {**item, "semantic_score": round(score, 4)}
            if medium and item_matches_medium(item, medium):
                enriched["query_match_medium"] = medium
            ranked_items.append(enriched)
        results = _diversify_by_source(ranked_items, limit=result_limit)
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


def _generate_hyde_document(
    query: str,
    category: Category,
    config: RuntimeConfig,
    ask_llm: AskLLM,
) -> str | None:
    """Генерирует гипотетический документ для cross-lingual semantic match.

    Fail-soft: любая ошибка LLM-генерации логируется и возвращает None,
    вызывающий код в этом случае откатывается на эмбеддинг сырого запроса.

    Args:
        query: Пользовательский запрос.
        category: Категория поиска.
        config: Runtime-настройки приложения.
        ask_llm: Инъецируемая функция LLM-вызова.

    Returns:
        Текст гипотетического документа либо None при сбое генерации.
    """
    try:
        return str(
            ask_llm(
                build_hyde_prompt(query, category),
                system=config.prompts.hyde_system,
            )["response"]
        )
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
    config: RuntimeConfig,
    medium: str | None = None,
) -> dict[str, tuple[dict[str, Any], float]]:
    """Сопоставляет scored-документы с кандидатами и применяет бонусы.

    Помимо exact-category бонуса к similarity-score добавляются поправка
    за тип записи (`kind`: бонус обзорам, штраф шуму) и бонус за совпадение
    с явно названным в запросе медиумом.

    Args:
        scored_documents: Пары (документ, score) из similarity search.
        items_by_key: Кандидаты пула, индексированные по candidate_key.
        category: Категория поиска для exact-match бонуса.
        config: Runtime-настройки scoring.
        medium: Медиум, явно названный в запросе, либо None.

    Returns:
        Словарь candidate_key -> (кандидат, score с учётом бонусов).
    """
    scored_by_key: dict[str, tuple[dict[str, Any], float]] = {}
    for document, score in scored_documents:
        key = str(document.metadata.get("candidate_key"))
        matched = items_by_key.get(key)
        if matched is None:
            continue
        kind_adjustments = {
            "review": config.tools.rag_review_score_bonus,
            "noise": config.tools.rag_noise_score_penalty,
        }
        adjusted = score + kind_adjustments.get(str(matched.get("kind") or ""), 0.0)
        item_category = str(matched.get("category") or "")
        if category != "all" and item_category != "mixed" and category_matches(item_category, category):
            adjusted += config.tools.rag_exact_category_bonus
        if medium and item_matches_medium(matched, medium):
            adjusted += config.tools.rag_medium_match_bonus
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
