"""LLM-классификация RSS-записей поверх regex-эвристики.

Regex-эвристика (`rss_feeds.classify`) ставит `kind` при загрузке фида —
дёшево, но грубо. Этот модуль уточняет вердикт дешёвой LLM (батчами) и
дополнительно извлекает `title_entity` — название тайтла, о котором запись.
Каждая запись классифицируется один раз в жизни: результат хранится в
персистентном кеше по хешу контента и модели (по образцу embeddings-кеша).
Fail-soft: при любом сбое LLM записи остаются с эвристическим `kind`.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langfuse import observe

from app.config import RuntimeConfig
from domain.models import FeedItem
from domain.storage.json_store import read_json, update_json
from llm_core.parsing import content_cache_key, parse_llm_json
from llm_core.prompts.classification import build_classification_prompt
from mcp_tools.llm import ask_llm_data

logger = logging.getLogger(__name__)

AskLLM = Callable[..., dict[str, Any]]

_VALID_KINDS = frozenset({"review", "release", "news", "noise"})
_VALID_MEDIUMS = frozenset({"anime", "game", "movie", "series", "other"})
# Версия схемы вердикта: смена инвалидирует кеш, и корпус переклассифицируется
# один раз с новыми полями (например, при добавлении `medium`).
_CACHE_SCHEMA_VERSION = "v2"


@observe(name="classify_feed_items", as_type="tool")
def classify_feed_items(
    items: list[FeedItem],
    config: RuntimeConfig,
    ask_llm: AskLLM = ask_llm_data,
) -> list[FeedItem]:
    """Уточняет `kind` и извлекает `title_entity` RSS-записей через LLM.

    Args:
        items: RSS-записи с эвристическим `kind`.
        config: Runtime-настройки приложения.
        ask_llm: Инъецируемая функция LLM-вызова.

    Returns:
        Записи с LLM-вердиктами там, где классификация удалась;
        остальные — без изменений (fail-soft).
    """
    if not config.tools.classifier_enabled or not items:
        return items

    cache: dict[str, Any] = read_json(config.backend.classification_cache_file, {})
    classifier_model = config.tools.normalized_classifier_model
    keys = [_cache_key(item, classifier_model) for item in items]
    pending = [index for index, key in enumerate(keys) if key not in cache]

    if pending:
        batch_size = max(1, config.tools.classifier_batch_size)
        batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
        worker_count = min(len(batches), max(1, config.tools.classifier_concurrency))

        def classify_batch(batch_indexes: list[int]) -> list[dict[str, Any]] | None:
            """Классифицирует один индексный батч без изменения общего кеша.

            Args:
                batch_indexes: Индексы RSS-элементов текущего батча.

            Returns:
                LLM-вердикты либо `None` при fail-soft ошибке.
            """
            return _classify_batch([items[index] for index in batch_indexes], config, ask_llm)

        if worker_count <= 1:
            batch_results = [classify_batch(batch_indexes) for batch_indexes in batches]
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="rss-classifier") as executor:
                batch_results = list(executor.map(classify_batch, batches))

        classified_count = 0
        classified: dict[str, Any] = {}
        for batch_indexes, verdicts in zip(batches, batch_results, strict=True):
            if verdicts is None:
                continue
            for index, verdict in zip(batch_indexes, verdicts, strict=True):
                classified[keys[index]] = verdict
            classified_count += len(batch_indexes)
        if classified_count:

            def merge(current: dict[str, Any]) -> dict[str, Any]:
                current.update(classified)
                return current

            cache = update_json(config.backend.classification_cache_file, {}, merge)
        logger.info(
            "LLM classification completed",
            extra={
                "item_count": len(items),
                "pending_count": len(pending),
                "classified_count": classified_count,
                "concurrency": worker_count,
            },
        )

    return [_apply_verdict(item, cache.get(key)) for item, key in zip(items, keys, strict=True)]


def _cache_key(item: FeedItem, classifier_model: str) -> str:
    """Строит ключ кеша классификации, привязанный к модели и контенту.

    Args:
        item: RSS-запись.
        classifier_model: Модель, для которой кешируется вердикт.

    Returns:
        Ключ вида `версия::model::sha256(title + summary)`.
    """
    return content_cache_key(f"{_CACHE_SCHEMA_VERSION}::{classifier_model}", f"{item.title}\n{item.summary}")


def _classify_batch(
    batch: list[FeedItem],
    config: RuntimeConfig,
    ask_llm: AskLLM,
) -> list[dict[str, Any]] | None:
    """Классифицирует один батч записей через LLM.

    Fail-soft: любая ошибка (сеть, невалидный JSON, неожиданная форма ответа)
    логируется и возвращает None — батч остаётся с эвристическими `kind`
    и не кешируется, чтобы повторная попытка случилась на следующем refresh.

    Args:
        batch: Записи одного батча.
        config: Runtime-настройки приложения.
        ask_llm: Инъецируемая функция LLM-вызова.

    Returns:
        Вердикты в порядке записей батча либо None при сбое.
    """
    tools = config.tools
    entries = [
        {"title": item.title, "summary": item.summary[: tools.classifier_summary_prompt_chars]} for item in batch
    ]
    try:
        llm = ask_llm(
            build_classification_prompt(entries),
            config.llm,
            config.prompts.recommendation_system,
            system=config.prompts.classifier_system,
            model=tools.normalized_classifier_model,
            max_tokens=tools.classifier_response_tokens_headroom + tools.classifier_verdict_tokens * len(batch),
            reasoning_effort="",
        )
        return _parse_verdicts(str(llm["response"]), expected_count=len(batch))
    except Exception:
        logger.warning(
            "LLM classification batch failed; keeping heuristic kinds",
            extra={"batch_size": len(batch), "model": config.tools.normalized_classifier_model},
            exc_info=True,
        )
        return None


def _parse_verdicts(response: str, expected_count: int) -> list[dict[str, Any]]:
    """Разбирает JSON-ответ классификатора в список вердиктов.

    Args:
        response: Сырой текст ответа модели.
        expected_count: Ожидаемое число вердиктов.

    Returns:
        Нормализованные вердикты `{"kind": ..., "title_entity": ...}`.

    Raises:
        ValueError: Если ответ не JSON-массив ожидаемой длины и формы.
    """
    data = parse_llm_json(response)
    if not isinstance(data, list) or len(data) != expected_count:
        raise ValueError(f"Expected JSON array of {expected_count} verdicts")

    verdicts: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict) or row.get("kind") not in _VALID_KINDS:
            raise ValueError("Invalid classification verdict")
        title_entity = row.get("title_entity")
        medium = row.get("medium")
        verdicts.append(
            {
                "kind": row["kind"],
                "medium": medium if medium in _VALID_MEDIUMS else "other",
                "title_entity": title_entity.strip() if isinstance(title_entity, str) else "",
            }
        )
    return verdicts


def _apply_verdict(item: FeedItem, verdict: Any) -> FeedItem:
    """Применяет закешированный вердикт к RSS-записи.

    Args:
        item: RSS-запись с эвристическим `kind`.
        verdict: Вердикт из кеша либо None/мусор.

    Returns:
        Копию записи с LLM-вердиктом либо исходную запись.
    """
    if not isinstance(verdict, dict) or verdict.get("kind") not in _VALID_KINDS:
        return item
    title_entity = verdict.get("title_entity")
    medium = verdict.get("medium")
    return item.model_copy(
        update={
            "kind": verdict["kind"],
            "medium": medium if isinstance(medium, str) and medium in _VALID_MEDIUMS else "",
            "title_entity": title_entity if isinstance(title_entity, str) else "",
        }
    )
