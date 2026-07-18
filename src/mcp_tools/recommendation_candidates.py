"""Чистые функции отбора и персонального ранжирования кандидатов.

Recommendation-граф отвечает за оркестрацию и внешний I/O. Этот модуль
содержит детерминированные преобразования уже загруженных кандидатов, чтобы
их можно было развивать и тестировать независимо от LangGraph, RSS и LLM.
"""

from collections import deque
from typing import Any

from domain.models import Category, category_matches
from mcp_tools.media import candidate_key
from mcp_tools.settings import ToolSettings


def query_variants(query: str, settings: ToolSettings) -> list[str]:
    """Строит варианты поиска по кешу для пользовательского запроса.

    Args:
        query: Исходный пользовательский запрос.
        settings: Настройки recommendation-инструментов.

    Returns:
        Базовый запрос и дополнительные vibe-варианты.
    """
    variants = [query]
    normalized = query.lower()
    if settings.normalized_keyword_marker in normalized or "vibe" in normalized:
        variants.extend(settings.normalized_keyword_variants)
    return variants


def prefer_exact_category(
    candidates: list[dict[str, Any]],
    category: Category,
    limit: int,
) -> list[dict[str, Any]]:
    """Сортирует fallback-кандидатов по exact, mixed и другим категориям.

    Args:
        candidates: Кандидаты fallback-пула.
        category: Запрошенная категория.
        limit: Максимальный размер результата.

    Returns:
        Приоритетный список кандидатов.
    """
    if category == "all":
        return interleave_by_source(candidates)[:limit]

    def bucket(item: dict[str, Any]) -> str:
        item_category = str(item.get("category") or "")
        if item_category != "mixed" and category_matches(item_category, category):
            return "exact"
        return "mixed" if item_category == "mixed" else "other"

    exact = interleave_by_source([item for item in candidates if bucket(item) == "exact"])
    mixed = interleave_by_source([item for item in candidates if bucket(item) == "mixed"])
    other = interleave_by_source([item for item in candidates if bucket(item) == "other"])
    return [*exact, *mixed, *other][:limit]


def interleave_by_source(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Чередует источники, сохраняя внутренний порядок.

    Args:
        candidates: Кандидаты в исходном порядке.

    Returns:
        Тот же набор кандидатов в round-robin порядке источников.
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


def rank_candidates_by_learned_preferences(
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
    settings: ToolSettings,
    memory_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Ранжирует кандидатов по весам профиля и семантической памяти.

    Args:
        candidates: RSS-кандидаты.
        profile: Профиль пользовательских предпочтений.
        settings: Настройки ranking weights.
        memory_scores: Семантические memory-score.

    Returns:
        Кандидаты в порядке убывания preference score.
    """
    learned = profile.get("learned_preferences", {})
    if not isinstance(learned, dict):
        return candidates

    scores = memory_scores or {}
    scored_candidates = [
        candidate_with_preference_score(candidate, learned, settings, scores.get(candidate_key(candidate)))
        for candidate in candidates
    ]
    return sorted(scored_candidates, key=lambda item: int(item.get("preference_score", 0)), reverse=True)


def candidate_with_preference_score(
    candidate: dict[str, Any],
    learned: dict[str, Any],
    settings: ToolSettings,
    memory_score: float | None = None,
) -> dict[str, Any]:
    """Добавляет одному кандидату объяснимый score предпочтения.

    Args:
        candidate: RSS-кандидат.
        learned: Накопленные learned preferences.
        settings: Настройки ranking weights.
        memory_score: Семантический memory-score.

    Returns:
        Копия кандидата с preference score и объяснениями.
    """
    score = 0
    reasons: list[str] = []
    score += _score_field(learned, "categories", str(candidate.get("category") or ""), reasons)
    score += _score_field(learned, "sources", str(candidate.get("source") or ""), reasons)

    kind = str(candidate.get("kind") or "")
    kind_weights = {
        "review": settings.recommendation_review_kind_weight,
        "noise": settings.recommendation_noise_kind_weight,
    }
    kind_score = kind_weights.get(kind, 0)
    if kind_score:
        score += kind_score
        reasons.append(f"kind:{kind}:{kind_score:+d}")

    medium = str(candidate.get("query_match_medium") or "")
    if medium:
        medium_weight = settings.recommendation_medium_match_weight
        score += medium_weight
        reasons.append(f"medium:{medium}:{medium_weight:+d}")

    # Запись с распознанным тайтлом — это «что посмотреть/поиграть»;
    # интервью и индустриальная болтовня без тайтла — слабый материал.
    if str(candidate.get("title_entity") or "").strip():
        entity_weight = settings.recommendation_title_entity_weight
        score += entity_weight
        reasons.append(f"entity:{entity_weight:+d}")

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
    """Возвращает один learned-вес и сохраняет его объяснение."""
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
