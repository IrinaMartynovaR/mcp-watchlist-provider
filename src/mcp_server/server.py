from typing import Any

from mcp.server.fastmcp import FastMCP

from app.logging_config import configure_logging
from app.observability import get_langfuse_client
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
from mcp_tools.recommendation import recommend_media_data
from mcp_tools.settings import (
    CACHE_SEARCH_DAYS,
    CACHE_SEARCH_LIMIT,
    FEED_REFRESH_LIMIT,
    RECOMMENDATION_LIMIT,
    RECOMMENDATION_SOURCE_LIMIT,
    SOURCE_VALIDATION_LIMIT,
)

mcp = FastMCP("watchquest")


@mcp.tool()
def get_profile() -> dict[str, Any]:
    """Return the user's media taste profile: likes, dislikes, platforms and language preferences."""
    return get_profile_data()


@mcp.tool()
def update_profile(likes: list[str] | None = None, dislikes: list[str] | None = None) -> dict[str, Any]:
    """Add new likes or dislikes to the user's profile."""
    return update_profile_data(likes=likes, dislikes=dislikes)


@mcp.tool()
def list_sources() -> list[dict[str, Any]]:
    """Return configured RSS sources for games, movies and series."""
    return list_sources_data()


@mcp.tool()
def validate_sources(category: str = "all", limit_per_source: int = SOURCE_VALIDATION_LIMIT) -> dict[str, Any]:
    """
    Check configured RSS sources and return per-source diagnostics.

    category: games, movies, series, movies_series, mixed, or all.
    """
    return validate_sources_data(category=parse_category(category), limit_per_source=limit_per_source)


@mcp.tool()
def refresh_feeds(category: str = "all", limit_per_source: int = FEED_REFRESH_LIMIT) -> dict[str, Any]:
    """
    Refresh RSS items, write the local cache, and return diagnostics.

    category: games, movies, series, movies_series, mixed, or all.
    """
    return refresh_feeds_data(category=parse_category(category), limit_per_source=limit_per_source)


@mcp.tool()
def search_cached_items(
    query: str,
    category: str = "all",
    days: int = CACHE_SEARCH_DAYS,
    limit: int = CACHE_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    """
    Search recently fetched cached items.

    Use bilingual queries when the user asks in Russian but sources may be English.
    Example query: "cozy RPG story rich".
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
    """Add a game, movie, series, or article to the user's watchlist."""
    return add_to_watchlist_data(title=title, media_type=media_type, url=url, reason=reason, source=source)


@mcp.tool()
def list_watchlist(media_type: str = "all", status: str = "planned") -> list[dict[str, Any]]:
    """Return items from the user's watchlist."""
    return list_watchlist_data(media_type=media_type, status=status)


@mcp.tool()
def rate_watchlist_item(title: str, rating: int, comment: str = "") -> dict[str, Any]:
    """Rate an item from the watchlist from 1 to 10 and add a comment."""
    return rate_watchlist_item_data(title=title, rating=rating, comment=comment)


@mcp.tool()
def recommend_media(
    query: str,
    category: str = "all",
    refresh: bool = True,
    limit: int = RECOMMENDATION_LIMIT,
    limit_per_source: int = RECOMMENDATION_SOURCE_LIMIT,
) -> dict[str, Any]:
    """Refresh feeds, select candidates, and ask the configured LLM for practical recommendations."""
    return recommend_media_data(
        query=query,
        category=parse_category(category),
        refresh=refresh,
        limit=limit,
        limit_per_source=limit_per_source,
    )


@mcp.resource("watchquest://profile")
def profile_resource() -> str:
    """Readable user profile resource."""
    profile = get_profile_data()
    return str(profile)


@mcp.resource("watchquest://watchlist")
def watchlist_resource() -> str:
    """Readable watchlist resource."""
    return str(list_watchlist_data(media_type="all", status="all"))


def main() -> None:
    """Запускает MCP-сервер WatchQuest."""
    configure_logging()
    get_langfuse_client()
    mcp.run()


if __name__ == "__main__":
    main()

