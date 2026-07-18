import logging
from collections.abc import Callable
from typing import Any

from app.config import RuntimeConfig
from domain.models import (
    Category,
    RecommendationFeedback,
    RecommendationRecord,
    parse_feedback_action,
)
from domain.repositories.feedback import FeedbackRepository
from domain.repositories.profile import ProfileRepository
from domain.repositories.recommendations import RecommendationRepository
from mcp_tools.media import add_to_watchlist_data

logger = logging.getLogger(__name__)

PreferenceRecorder = Callable[[RecommendationFeedback, dict[str, Any] | None], None]


def save_recommendation_result(config: RuntimeConfig, result: dict[str, Any]) -> RecommendationRecord:
    """Сохраняет результат рекомендации в историю.

    Args:
        config: Runtime-настройки приложения.
        result: Нормализованный результат `recommend_media_data`.

    Returns:
        Сохранённую recommendation-запись.
    """
    record = RecommendationRecord.model_validate(result)
    RecommendationRepository(config.backend.recommendations_file).save(record)
    logger.info(
        "Recommendation saved",
        extra={"recommendation_id": record.id, "candidate_count": len(record.candidates)},
    )
    return record


def get_recommendation_record(config: RuntimeConfig, recommendation_id: str) -> RecommendationRecord | None:
    """Возвращает сохранённую рекомендацию по id.

    Args:
        config: Runtime-настройки приложения.
        recommendation_id: Идентификатор recommendation-записи.

    Returns:
        Recommendation-запись либо `None`, если она не найдена.
    """
    return RecommendationRepository(config.backend.recommendations_file).get(recommendation_id)


def apply_recommendation_feedback(
    config: RuntimeConfig,
    recommendation_id: str,
    action: str,
    record_preference: PreferenceRecorder,
) -> dict[str, Any]:
    """Сохраняет feedback и обновляет learned-предпочтения пользователя.

    Args:
        config: Runtime-настройки приложения.
        recommendation_id: Идентификатор рекомендации из Telegram callback.
        action: Feedback-действие пользователя.
        record_preference: Инъецируемая функция записи семантической памяти.

    Returns:
        Сводку результата обработки feedback.

    Raises:
        ValueError: Если recommendation id или action некорректны.
    """
    parsed_action = parse_feedback_action(action)
    record = get_recommendation_record(config, recommendation_id)
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
    _save_feedback(config, feedback)
    _apply_feedback_to_profile(config, feedback=feedback, candidate=top_candidate, record_preference=record_preference)

    watchlist_item: dict[str, Any] | None = None
    if parsed_action == "watchlist" and top_candidate is not None:
        watchlist_item = add_to_watchlist_data(
            config,
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


def _save_feedback(config: RuntimeConfig, feedback: RecommendationFeedback) -> None:
    """Добавляет feedback-событие в хранилище."""
    FeedbackRepository(config.backend.feedback_file).append(feedback)


def _apply_feedback_to_profile(
    config: RuntimeConfig,
    feedback: RecommendationFeedback,
    candidate: dict[str, Any] | None,
    record_preference: PreferenceRecorder,
) -> None:
    """Обновляет learned_preferences на основе feedback."""
    weight = config.tools.feedback_weights[feedback.action]
    ProfileRepository(config.backend.profile_file).learn(
        category=feedback.category,
        source=feedback.source,
        tags=_candidate_tags(candidate),
        weight=weight,
    )
    try:
        record_preference(feedback, candidate)
    except Exception:
        logger.warning(
            "Failed to record preference note",
            extra={"recommendation_id": feedback.recommendation_id, "action": feedback.action},
            exc_info=True,
        )


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
