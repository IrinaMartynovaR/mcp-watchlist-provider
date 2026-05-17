import logging
from typing import Any

from app.settings import backend_settings
from domain.models import (
    Category,
    RecommendationFeedback,
    RecommendationRecord,
    parse_feedback_action,
)
from domain.storage.json_store import read_json, write_json
from mcp_tools.media import add_to_watchlist_data
from mcp_tools.settings import tool_settings

logger = logging.getLogger(__name__)


def save_recommendation_result(result: dict[str, Any]) -> RecommendationRecord:
    """Сохраняет результат рекомендации в историю.

    Args:
        result: Нормализованный результат `recommend_media_data`.

    Returns:
        Сохранённую recommendation-запись.
    """
    record = RecommendationRecord.model_validate(result)
    data: dict[str, Any] = read_json(backend_settings.recommendations_file, {"items": []})
    items = [item for item in data.get("items", []) if item.get("id") != record.id]
    items.append(record.model_dump(mode="json"))
    data["items"] = items
    write_json(backend_settings.recommendations_file, data)
    logger.info(
        "Recommendation saved",
        extra={"recommendation_id": record.id, "candidate_count": len(record.candidates)},
    )
    return record


def get_recommendation_record(recommendation_id: str) -> RecommendationRecord | None:
    """Возвращает сохранённую рекомендацию по id.

    Args:
        recommendation_id: Идентификатор recommendation-записи.

    Returns:
        Recommendation-запись либо `None`, если она не найдена.
    """
    data: dict[str, Any] = read_json(backend_settings.recommendations_file, {"items": []})
    for item in data.get("items", []):
        if item.get("id") == recommendation_id:
            return RecommendationRecord.model_validate(item)
    return None


def apply_recommendation_feedback(recommendation_id: str, action: str) -> dict[str, Any]:
    """Сохраняет feedback и обновляет learned-предпочтения пользователя.

    Args:
        recommendation_id: Идентификатор рекомендации из Telegram callback.
        action: Feedback-действие пользователя.

    Returns:
        Сводку результата обработки feedback.

    Raises:
        ValueError: Если recommendation id или action некорректны.
    """
    parsed_action = parse_feedback_action(action)
    record = get_recommendation_record(recommendation_id)
    if record is None:
        raise ValueError(f"Recommendation not found: {recommendation_id}")

    top_candidate = _top_candidate(record)
    feedback = RecommendationFeedback(
        recommendation_id=record.id,
        action=parsed_action,
        query=record.query,
        category=record.category,
        title=_candidate_text(top_candidate, "title"),
        source=_candidate_text(top_candidate, "source"),
    )
    _save_feedback(feedback)
    _apply_feedback_to_profile(feedback=feedback, candidate=top_candidate)

    watchlist_item: dict[str, Any] | None = None
    if parsed_action == "watchlist" and top_candidate is not None:
        watchlist_item = add_to_watchlist_data(
            title=_candidate_text(top_candidate, "title") or "Untitled recommendation",
            media_type=_media_type_from_category(record.category),
            url=_candidate_text(top_candidate, "url"),
            reason=f"Saved from recommendation: {record.query}",
            source=_candidate_text(top_candidate, "source"),
        )

    logger.info(
        "Recommendation feedback applied",
        extra={"recommendation_id": record.id, "action": parsed_action, "title": feedback.title},
    )
    return {
        "recommendation_id": record.id,
        "action": parsed_action,
        "title": feedback.title,
        "watchlist_item": watchlist_item,
    }


def _save_feedback(feedback: RecommendationFeedback) -> None:
    """Добавляет feedback-событие в хранилище."""
    data: dict[str, Any] = read_json(backend_settings.feedback_file, {"items": []})
    data["items"].append(feedback.model_dump(mode="json"))
    write_json(backend_settings.feedback_file, data)


def _apply_feedback_to_profile(feedback: RecommendationFeedback, candidate: dict[str, Any] | None) -> None:
    """Обновляет learned_preferences на основе feedback."""
    profile: dict[str, Any] = read_json(backend_settings.profile_file, {})
    learned = profile.setdefault("learned_preferences", {})
    if not isinstance(learned, dict):
        learned = {}
        profile["learned_preferences"] = learned

    weight = tool_settings.feedback_weights[feedback.action]
    _bump_weight(learned, "categories", feedback.category, weight)
    if feedback.source:
        _bump_weight(learned, "sources", feedback.source, weight)

    for tag in _candidate_tags(candidate):
        _bump_weight(learned, "tags", tag, weight)

    write_json(backend_settings.profile_file, profile)


def _bump_weight(learned: dict[str, Any], section: str, key: str, delta: int) -> None:
    """Изменяет числовой вес preference-секции."""
    values = learned.setdefault(section, {})
    if not isinstance(values, dict):
        values = {}
        learned[section] = values
    current = values.get(key, 0)
    values[key] = int(current if isinstance(current, int) else 0) + delta


def _top_candidate(record: RecommendationRecord) -> dict[str, Any] | None:
    """Возвращает первого кандидата рекомендации."""
    if not record.candidates:
        return None
    candidate = record.candidates[0]
    return candidate if isinstance(candidate, dict) else None


def _candidate_text(candidate: dict[str, Any] | None, key: str) -> str | None:
    """Достаёт непустую строку из candidate."""
    if candidate is None:
        return None
    value = candidate.get(key)
    return str(value).strip() if value else None


def _candidate_tags(candidate: dict[str, Any] | None) -> list[str]:
    """Возвращает строковые tags кандидата."""
    if candidate is None:
        return []
    raw_tags = candidate.get("tags", [])
    if not isinstance(raw_tags, list):
        return []
    return [str(tag).strip().lower() for tag in raw_tags if str(tag).strip()]


def _media_type_from_category(category: Category) -> str:
    """Преобразует recommendation-категорию в watchlist media type."""
    if category == "games":
        return "game"
    if category == "series":
        return "series"
    if category in {"movies", "movies_series"}:
        return "movie"
    return "unknown"
