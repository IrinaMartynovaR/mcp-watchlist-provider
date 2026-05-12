from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

Category = Literal["games", "movies", "series", "movies_series", "mixed", "all"]
MediaType = Literal["game", "movie", "series", "article", "unknown"]

CATEGORIES: tuple[Category, ...] = ("games", "movies", "series", "movies_series", "mixed", "all")
MEDIA_TYPES: tuple[MediaType, ...] = ("game", "movie", "series", "article", "unknown")


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

