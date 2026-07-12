from typing import Any

from mcp.server.fastmcp import FastMCP

from app.logging_config import configure_logging
from domain.models import parse_category
from mcp_tools.media import (
    add_to_watchlist_data,
    get_profile_data,
    list_sources_data,
    list_watchlist_data,
    rate_watchlist_item_data,
    refresh_feeds_data,
    search_cached_items_data,
    update_profile_data,
    validate_sources_data,
)
from mcp_tools.myshows_import import import_myshows_history_data
from mcp_tools.recommendation import recommend_media_data
from mcp_tools.settings import tool_settings

mcp = FastMCP("watchquest")


@mcp.tool()
def get_profile() -> dict[str, Any]:
    """Возвращает профиль медиапредпочтений пользователя.

    Returns:
        Профиль с предпочтениями, антипредпочтениями, платформами и языковыми настройками.
    """
    return get_profile_data()


@mcp.tool()
def update_profile(likes: list[str] | None = None, dislikes: list[str] | None = None) -> dict[str, Any]:
    """Добавляет новые предпочтения или антипредпочтения в профиль.

    Args:
        likes: Новые предпочтения пользователя.
        dislikes: Новые антипредпочтения пользователя.

    Returns:
        Обновлённый профиль пользователя.
    """
    return update_profile_data(likes=likes, dislikes=dislikes)


@mcp.tool()
def list_sources() -> list[dict[str, Any]]:
    """Возвращает настроенные RSS-источники.

    Returns:
        Список RSS-источников для игр, фильмов, сериалов и смешанного контента.
    """
    return list_sources_data()


@mcp.tool()
def validate_sources(
    category: str = "all",
    limit_per_source: int = tool_settings.source_validation_limit,
) -> dict[str, Any]:
    """Проверяет доступность настроенных RSS-источников.

    Args:
        category: Категория источников: games, movies, series,
            movies_series, mixed или all.
        limit_per_source: Максимальное число RSS-записей на источник.

    Returns:
        Диагностику по каждому источнику и общий статус проверки.

    Raises:
        ValueError: Если передана неподдерживаемая категория.
    """
    return validate_sources_data(category=parse_category(category), limit_per_source=limit_per_source)


@mcp.tool()
def refresh_feeds(
    category: str = "all",
    limit_per_source: int = tool_settings.feed_refresh_limit,
) -> dict[str, Any]:
    """Обновляет RSS-элементы и записывает локальный кеш.

    Args:
        category: Категория источников: games, movies, series,
            movies_series, mixed или all.
        limit_per_source: Максимальное число RSS-записей на источник.

    Returns:
        Сводку обновления, список элементов, диагностику источников и ошибки.

    Raises:
        ValueError: Если передана неподдерживаемая категория.
    """
    return refresh_feeds_data(category=parse_category(category), limit_per_source=limit_per_source)


@mcp.tool()
def search_cached_items(
    query: str,
    category: str = "all",
    days: int = tool_settings.cache_search_days,
    limit: int = tool_settings.cache_search_limit,
) -> list[dict[str, Any]]:
    """Ищет элементы в недавно обновлённом RSS-кеше.

    Args:
        query: Поисковая строка. Для русских запросов можно передавать
            двуязычные формулировки, если часть источников на английском.
        category: Категория поиска: games, movies, series,
            movies_series, mixed или all.
        days: Глубина поиска по давности публикации.
        limit: Максимальное число результатов.

    Returns:
        Список найденных RSS-элементов.

    Raises:
        ValueError: Если передана неподдерживаемая категория.
    """
    return search_cached_items_data(query=query, category=parse_category(category), days=days, limit=limit)


@mcp.tool()
def add_to_watchlist(
    title: str,
    media_type: str = "unknown",
    url: str | None = None,
    reason: str = "",
    source: str | None = None,
) -> dict[str, Any]:
    """Добавляет элемент в пользовательский watchlist.

    Args:
        title: Название игры, фильма, сериала или статьи.
        media_type: Тип медиа: game, movie, series, article или unknown.
        url: Необязательная ссылка на источник.
        reason: Причина добавления.
        source: Название источника рекомендации.

    Returns:
        Сохранённый watchlist-элемент.

    Raises:
        ValueError: Если передан неподдерживаемый тип медиа.
    """
    return add_to_watchlist_data(title=title, media_type=media_type, url=url, reason=reason, source=source)


@mcp.tool()
def list_watchlist(media_type: str = "all", status: str = "planned") -> list[dict[str, Any]]:
    """Возвращает элементы пользовательского watchlist.

    Args:
        media_type: Тип медиа для фильтрации или all.
        status: Статус элемента для фильтрации или all.

    Returns:
        Отфильтрованный список watchlist-элементов.
    """
    return list_watchlist_data(media_type=media_type, status=status)


@mcp.tool()
def rate_watchlist_item(title: str, rating: int, comment: str = "") -> dict[str, Any]:
    """Сохраняет оценку и комментарий для watchlist-элемента.

    Args:
        title: Название элемента для поиска.
        rating: Оценка пользователя от 1 до 10.
        comment: Необязательный комментарий.

    Returns:
        Обновлённый watchlist-элемент.

    Raises:
        ValueError: Если элемент с таким названием не найден.
    """
    return rate_watchlist_item_data(title=title, rating=rating, comment=comment)


@mcp.tool()
def recommend_media(
    query: str,
    category: str = "all",
    refresh: bool = True,
    limit: int = tool_settings.recommendation_limit,
    limit_per_source: int = tool_settings.recommendation_source_limit,
) -> dict[str, Any]:
    """Собирает практическую рекомендацию через RSS-кандидатов и LLM.

    Args:
        query: Пользовательский запрос.
        category: Категория поиска: games, movies, series,
            movies_series, mixed или all.
        refresh: Нужно ли обновить RSS-кеш перед поиском кандидатов.
        limit: Максимальное число кандидатов для LLM.
        limit_per_source: Максимальное число RSS-записей на источник.

    Returns:
        Результат рекомендации с кандидатами, LLM-ответом и диагностикой использованных инструментов.

    Raises:
        ValueError: Если передана неподдерживаемая категория.
    """
    return recommend_media_data(
        query=query,
        category=parse_category(category),
        refresh=refresh,
        limit=limit,
        limit_per_source=limit_per_source,
    )


@mcp.tool()
def import_myshows_history(dry_run: bool = True) -> dict[str, Any]:
    """Импортирует историю просмотров MyShows.me в графовую память предпочтений.

    Требует MYSHOWS_LOGIN и MYSHOWS_PASSWORD в окружении. По умолчанию dry_run=True:
    возвращается только сводка, в память ничего не записывается — так можно проверить
    маппинг полей на реальном ответе API перед первым настоящим импортом.

    Args:
        dry_run: Если True — только посчитать сводку, ничего не записывать в память.

    Returns:
        Сводку импорта: сколько фильмов и сериалов обработано, сколько заметок
        записано (или было бы записано при dry_run) и сколько пропущено.

    Raises:
        RuntimeError: Если логин в MyShows или вызовы его API завершились ошибкой.
    """
    return import_myshows_history_data(dry_run=dry_run)


@mcp.resource("watchquest://profile")
def profile_resource() -> str:
    """Возвращает профиль пользователя как MCP resource.

    Returns:
        Строковое представление профиля пользователя.
    """
    profile = get_profile_data()
    return str(profile)


@mcp.resource("watchquest://watchlist")
def watchlist_resource() -> str:
    """Возвращает watchlist пользователя как MCP resource.

    Returns:
        Строковое представление полного watchlist.
    """
    return str(list_watchlist_data(media_type="all", status="all"))


def main() -> None:
    """Запускает MCP-сервер WatchQuest."""
    configure_logging()
    mcp.run()


if __name__ == "__main__":
    main()

