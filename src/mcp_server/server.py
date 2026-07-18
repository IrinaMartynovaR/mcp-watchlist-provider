from typing import Any

from mcp.server.fastmcp import FastMCP

from app.application import WatchQuestApplication
from app.config import RuntimeConfig
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


def create_mcp_server(application: WatchQuestApplication) -> FastMCP:
    """Создаёт MCP-сервер с зависимостями текущего приложения.

    Args:
        application: Собранный composition root WatchQuest.

    Returns:
        Настроенный FastMCP server.
    """
    server = FastMCP("watchquest")
    config = application.config

    @server.tool()
    def get_profile() -> dict[str, Any]:
        """Возвращает профиль медиапредпочтений пользователя."""
        return get_profile_data(config)

    @server.tool()
    def update_profile(likes: list[str] | None = None, dislikes: list[str] | None = None) -> dict[str, Any]:
        """Добавляет новые предпочтения или антипредпочтения.

        Args:
            likes: Новые предпочтения пользователя.
            dislikes: Новые антипредпочтения пользователя.

        Returns:
            Обновлённый профиль.
        """
        return update_profile_data(config, likes=likes, dislikes=dislikes)

    @server.tool()
    def list_sources() -> list[dict[str, Any]]:
        """Возвращает настроенные RSS-источники."""
        return list_sources_data(config)

    @server.tool()
    def validate_sources(
        category: str = "all",
        limit_per_source: int = config.tools.source_validation_limit,
    ) -> dict[str, Any]:
        """Проверяет доступность настроенных RSS-источников.

        Args:
            category: Категория источников.
            limit_per_source: Максимальное число записей на источник.

        Returns:
            Диагностика по каждому источнику.
        """
        return validate_sources_data(
            config,
            category=parse_category(category),
            limit_per_source=limit_per_source,
        )

    @server.tool()
    def refresh_feeds(
        category: str = "all",
        limit_per_source: int = config.tools.feed_refresh_limit,
    ) -> dict[str, Any]:
        """Обновляет RSS-кеш.

        Args:
            category: Категория источников.
            limit_per_source: Максимальное число записей на источник.

        Returns:
            Сводка обновления, записи и ошибки источников.
        """
        return refresh_feeds_data(
            config,
            category=parse_category(category),
            limit_per_source=limit_per_source,
        )

    @server.tool()
    def search_cached_items(
        query: str,
        category: str = "all",
        days: int = config.tools.cache_search_days,
        limit: int = config.tools.cache_search_limit,
    ) -> list[dict[str, Any]]:
        """Ищет записи в локальном RSS-кеше.

        Args:
            query: Поисковая строка.
            category: Категория поиска.
            days: Глубина поиска по давности.
            limit: Максимальное число результатов.

        Returns:
            Найденные RSS-записи.
        """
        return search_cached_items_data(
            config,
            query=query,
            category=parse_category(category),
            days=days,
            limit=limit,
        )

    @server.tool()
    def add_to_watchlist(
        title: str,
        media_type: str = "unknown",
        url: str | None = None,
        reason: str = "",
        source: str | None = None,
    ) -> dict[str, Any]:
        """Добавляет элемент в watchlist.

        Args:
            title: Название элемента.
            media_type: Тип медиа.
            url: Ссылка на источник.
            reason: Причина добавления.
            source: Название источника.

        Returns:
            Сохранённый watchlist item.
        """
        return add_to_watchlist_data(
            config,
            title=title,
            media_type=media_type,
            url=url,
            reason=reason,
            source=source,
        )

    @server.tool()
    def list_watchlist(media_type: str = "all", status: str = "planned") -> list[dict[str, Any]]:
        """Возвращает отфильтрованный watchlist.

        Args:
            media_type: Тип медиа или ``all``.
            status: Статус элемента или ``all``.

        Returns:
            Watchlist items.
        """
        return list_watchlist_data(config, media_type=media_type, status=status)

    @server.tool()
    def rate_watchlist_item(title: str, rating: int, comment: str = "") -> dict[str, Any]:
        """Сохраняет оценку watchlist-элемента.

        Args:
            title: Название элемента.
            rating: Оценка от 1 до 10.
            comment: Пользовательский комментарий.

        Returns:
            Обновлённый watchlist item.
        """
        return rate_watchlist_item_data(config, title=title, rating=rating, comment=comment)

    @server.tool()
    def recommend_media(
        query: str,
        category: str = "all",
        refresh: bool = True,
        limit: int = config.tools.recommendation_limit,
        limit_per_source: int = config.tools.recommendation_source_limit,
    ) -> dict[str, Any]:
        """Собирает рекомендацию через RSS, память и LLM.

        Args:
            query: Пользовательский запрос.
            category: Категория поиска.
            refresh: Нужно ли обновить RSS-кеш.
            limit: Максимальное число кандидатов.
            limit_per_source: Лимит записей на источник.

        Returns:
            Рекомендация, кандидаты и диагностика pipeline.
        """
        return application.recommendation.recommend(
            query=query,
            category=parse_category(category),
            refresh=refresh,
            limit=limit,
            limit_per_source=limit_per_source,
        )

    @server.tool()
    def import_myshows_history(dry_run: bool = True) -> dict[str, Any]:
        """Импортирует историю MyShows в графовую память.

        Args:
            dry_run: Если True, только рассчитывает сводку.

        Returns:
            Сводка импорта.
        """
        return application.import_myshows_history(dry_run=dry_run)

    @server.resource("watchquest://profile")
    def profile_resource() -> str:
        """Возвращает профиль как MCP resource."""
        return str(get_profile_data(config))

    @server.resource("watchquest://watchlist")
    def watchlist_resource() -> str:
        """Возвращает watchlist как MCP resource."""
        return str(list_watchlist_data(config, media_type="all", status="all"))

    return server


def main() -> None:
    """Собирает зависимости и запускает MCP-сервер WatchQuest."""
    config = RuntimeConfig.from_env()
    configure_logging(config.backend, stream="stderr")
    create_mcp_server(WatchQuestApplication(config)).run()


if __name__ == "__main__":
    main()
