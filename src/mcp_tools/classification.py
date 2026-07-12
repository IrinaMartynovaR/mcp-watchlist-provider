"""LLM-классификация RSS-записей поверх regex-эвристики.

Regex-эвристика (`rss_feeds.classify`) ставит `kind` при загрузке фида —
дёшево, но грубо. Этот модуль уточняет вердикт дешёвой LLM (батчами) и
дополнительно извлекает `title_entity` — название тайтла, о котором запись.
Каждая запись классифицируется один раз в жизни: результат хранится в
персистентном кеше по хешу контента и модели (по образцу embeddings-кеша).
Fail-soft: при любом сбое LLM записи остаются с эвристическим `kind`.
"""

import hashlib
import json
import logging
from typing import Any

from langfuse import observe

from app.settings import backend_settings
from domain.models import FeedItem
from domain.storage.json_store import read_json, write_json
from llm_core.prompts.classification import CLASSIFIER_SYSTEM_PROMPT, build_classification_prompt
from mcp_tools.llm import ask_llm_data
from mcp_tools.settings import tool_settings

logger = logging.getLogger(__name__)

_VALID_KINDS = frozenset({"review", "release", "news", "noise"})
_SUMMARY_PROMPT_CHARS = 200
# Бюджет ответа на один вердикт: {"kind": ..., "title_entity": ...} с длинным
# названием — до ~100 токенов; без запаса ответ обрезается и батч теряется.
_VERDICT_TOKENS = 120
_RESPONSE_TOKENS_HEADROOM = 200


@observe(name="classify_feed_items", as_type="tool")
def classify_feed_items(items: list[FeedItem]) -> list[FeedItem]:
    """Уточняет `kind` и извлекает `title_entity` RSS-записей через LLM.

    Args:
        items: RSS-записи с эвристическим `kind`.

    Returns:
        Записи с LLM-вердиктами там, где классификация удалась;
        остальные — без изменений (fail-soft).
    """
    if not tool_settings.classifier_enabled or not items:
        return items

    cache: dict[str, Any] = read_json(backend_settings.classification_cache_file, {})
    keys = [_cache_key(item) for item in items]
    pending = [index for index, key in enumerate(keys) if key not in cache]

    if pending:
        batch_size = max(1, tool_settings.classifier_batch_size)
        classified_count = 0
        for start in range(0, len(pending), batch_size):
            batch_indexes = pending[start : start + batch_size]
            verdicts = _classify_batch([items[index] for index in batch_indexes])
            if verdicts is None:
                continue
            for index, verdict in zip(batch_indexes, verdicts, strict=True):
                cache[keys[index]] = verdict
            classified_count += len(batch_indexes)
        if classified_count:
            write_json(backend_settings.classification_cache_file, cache)
        logger.info(
            "LLM classification completed",
            extra={"item_count": len(items), "pending_count": len(pending), "classified_count": classified_count},
        )

    return [_apply_verdict(item, cache.get(key)) for item, key in zip(items, keys, strict=True)]


def _cache_key(item: FeedItem) -> str:
    """Строит ключ кеша классификации, привязанный к модели и контенту.

    Args:
        item: RSS-запись.

    Returns:
        Ключ вида `model::sha256(title + summary)`.
    """
    digest = hashlib.sha256(f"{item.title}\n{item.summary}".encode()).hexdigest()
    return f"{tool_settings.normalized_classifier_model}::{digest}"


def _classify_batch(batch: list[FeedItem]) -> list[dict[str, Any]] | None:
    """Классифицирует один батч записей через LLM.

    Fail-soft: любая ошибка (сеть, невалидный JSON, неожиданная форма ответа)
    логируется и возвращает None — батч остаётся с эвристическими `kind`
    и не кешируется, чтобы повторная попытка случилась на следующем refresh.

    Args:
        batch: Записи одного батча.

    Returns:
        Вердикты в порядке записей батча либо None при сбое.
    """
    entries = [{"title": item.title, "summary": item.summary[:_SUMMARY_PROMPT_CHARS]} for item in batch]
    try:
        llm = ask_llm_data(
            build_classification_prompt(entries),
            system=CLASSIFIER_SYSTEM_PROMPT,
            model=tool_settings.normalized_classifier_model,
            max_tokens=_RESPONSE_TOKENS_HEADROOM + _VERDICT_TOKENS * len(batch),
        )
        return _parse_verdicts(str(llm["response"]), expected_count=len(batch))
    except Exception:
        logger.warning(
            "LLM classification batch failed; keeping heuristic kinds",
            extra={"batch_size": len(batch), "model": tool_settings.normalized_classifier_model},
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
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:]

    data = json.loads(text)
    if not isinstance(data, list) or len(data) != expected_count:
        raise ValueError(f"Expected JSON array of {expected_count} verdicts")

    verdicts: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict) or row.get("kind") not in _VALID_KINDS:
            raise ValueError("Invalid classification verdict")
        title_entity = row.get("title_entity")
        verdicts.append(
            {
                "kind": row["kind"],
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
    return item.model_copy(
        update={
            "kind": verdict["kind"],
            "title_entity": title_entity if isinstance(title_entity, str) else "",
        }
    )
