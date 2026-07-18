from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

Category = Literal["games", "movies", "series", "movies_series", "mixed", "all"]
MediaType = Literal["game", "movie", "series", "article", "unknown"]
FeedbackAction = Literal["like", "dislike", "watchlist", "block_similar"]
ItemKind = Literal["review", "release", "news", "noise"]

CATEGORIES: tuple[Category, ...] = ("games", "movies", "series", "movies_series", "mixed", "all")
MEDIA_TYPES: tuple[MediaType, ...] = ("game", "movie", "series", "article", "unknown")
FEEDBACK_ACTIONS: tuple[FeedbackAction, ...] = ("like", "dislike", "watchlist", "block_similar")


def parse_category(value: str) -> Category:
    """Преобразует строку в поддерживаемую категорию.

    Args:
        value: Значение категории из внешнего ввода.

    Returns:
        Валидированную категорию WatchQuest.

    Raises:
        ValueError: Если категория не поддерживается.
    """
    if value in CATEGORIES:
        return value
    raise ValueError(f"Unsupported category: {value}")


def category_matches(value: str, requested: Category) -> bool:
    """Проверяет, попадает ли категория записи в запрошенную категорию.

    Категория `movies_series` — общая для кино и сериалов: такие записи
    подходят под запросы `movies` и `series` (и наоборот), иначе источники
    вроде Variety были бы невидимы для узких запросов.

    Args:
        value: Категория записи или источника (строка из данных).
        requested: Запрошенная пользователем категория.

    Returns:
        True, если запись подходит под запрошенную категорию.
    """
    if requested == "all" or value in {requested, "mixed"}:
        return True
    if value == "movies_series":
        return requested in {"movies", "series"}
    if requested == "movies_series":
        return value in {"movies", "series"}
    return False


def parse_media_type(value: str) -> MediaType:
    """Преобразует строку в поддерживаемый тип медиа.

    Args:
        value: Значение типа медиа из внешнего ввода.

    Returns:
        Валидированный тип медиа.

    Raises:
        ValueError: Если тип медиа не поддерживается.
    """
    if value in MEDIA_TYPES:
        return value
    raise ValueError(f"Unsupported media type: {value}")


def parse_feedback_action(value: str) -> FeedbackAction:
    """Преобразует строку в поддерживаемое feedback-действие.

    Args:
        value: Значение действия из callback data.

    Returns:
        Валидированное feedback-действие.

    Raises:
        ValueError: Если действие не поддерживается.
    """
    if value in FEEDBACK_ACTIONS:
        return value
    raise ValueError(f"Unsupported feedback action: {value}")


class Source(BaseModel):
    """Описывает один RSS-источник WatchQuest."""

    name: str
    category: Category = "mixed"
    language: str = "unknown"
    url: HttpUrl


class FeedItem(BaseModel):
    """Описывает нормализованную запись из RSS-ленты."""

    title: str
    url: str
    source: str
    source_language: str = "unknown"
    category: Category = "mixed"
    kind: ItemKind = "news"
    medium: str = ""
    title_entity: str = ""
    summary: str = ""
    published_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)


class WatchlistItem(BaseModel):
    """Описывает элемент пользовательского watchlist."""

    title: str
    type: MediaType = "unknown"
    url: str | None = None
    reason: str = ""
    source: str | None = None
    status: Literal["planned", "watched", "played", "dropped", "not_interested"] = "planned"
    rating: int | None = Field(default=None, ge=1, le=10)
    comment: str = ""
    added_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RecommendationRecord(BaseModel):
    """Описывает сохранённую рекомендацию и её контекст."""

    id: str
    query: str
    category: Category = "all"
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str = ""
    model: str = ""
    provider: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RecommendationFeedback(BaseModel):
    """Описывает пользовательскую оценку сохранённой рекомендации."""

    recommendation_id: str
    action: FeedbackAction
    query: str = ""
    category: Category = "all"
    title: str | None = None
    source: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
