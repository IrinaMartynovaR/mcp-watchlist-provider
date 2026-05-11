from typing import Any

from mcp.server.fastmcp import FastMCP

from app.logging_config import configure_logging
from app.observability import get_langfuse_client
from domain.models import parse_category
from mcp_tools.llm import ask_llm_data, recommend_with_llm_data
from mcp_tools.media import (
    add_to_watchlist_data,
    fetch_latest_items_data,
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
from mcp_tools.rss_bridge import (
    add_rss_bridge_source_data,
    build_rss_bridge_feed_url_data,
    list_rss_bridges_data,
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
def fetch_latest_items(category: str = "all", limit_per_source: int = 20) -> list[dict[str, Any]]:
    """
    Fetch latest items from configured RSS sources.

    category: games, movies, series, movies_series, mixed, or all.
    """
    return fetch_latest_items_data(category=parse_category(category), limit_per_source=limit_per_source)


@mcp.tool()
def validate_sources(category: str = "all", limit_per_source: int = 3) -> dict[str, Any]:
    """
    Check configured RSS sources and return per-source diagnostics.

    category: games, movies, series, movies_series, mixed, or all.
    """
    return validate_sources_data(category=parse_category(category), limit_per_source=limit_per_source)


@mcp.tool()
def refresh_feeds(category: str = "all", limit_per_source: int = 20) -> dict[str, Any]:
    """
    Refresh RSS items, write the local cache, and return diagnostics.

    category: games, movies, series, movies_series, mixed, or all.
    """
    return refresh_feeds_data(category=parse_category(category), limit_per_source=limit_per_source)


@mcp.tool()
def search_cached_items(query: str, category: str = "all", days: int = 30, limit: int = 20) -> list[dict[str, Any]]:
    """
    Search recently fetched cached items.

    Use bilingual queries when the user asks in Russian but sources may be English.
    Example query: "cozy RPG story rich".
    """
    return search_cached_items_data(query=query, category=parse_category(category), days=days, limit=limit)


@mcp.tool()
def add_to_watchlist(
    title: str,
    type: str = "unknown",  # noqa: A002
    url: str | None = None,
    reason: str = "",
    source: str | None = None,
) -> dict[str, Any]:
    """Add a game, movie, series, or article to the user's watchlist."""
    return add_to_watchlist_data(title=title, type=type, url=url, reason=reason, source=source)


@mcp.tool()
def list_watchlist(type: str = "all", status: str = "planned") -> list[dict[str, Any]]:  # noqa: A002
    """Return items from the user's watchlist."""
    return list_watchlist_data(type=type, status=status)


@mcp.tool()
def rate_watchlist_item(title: str, rating: int, comment: str = "") -> dict[str, Any]:
    """Rate an item from the watchlist from 1 to 10 and add a comment."""
    return rate_watchlist_item_data(title=title, rating=rating, comment=comment)


@mcp.tool()
def ask_llm(prompt: str, system: str | None = None) -> dict[str, Any]:
    """Ask the configured LLM provider directly."""
    return ask_llm_data(prompt=prompt, system=system)


@mcp.tool()
def recommend_with_llm(query: str = "", category: str = "all", limit: int = 8) -> dict[str, Any]:
    """Generate recommendations with the configured LLM provider using profile, watchlist and cached feed items."""
    return recommend_with_llm_data(query=query, category=parse_category(category), limit=limit)


@mcp.tool()
def list_rss_bridges(query: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """Search available RSS-Bridge bridge definitions."""
    return list_rss_bridges_data(query=query, limit=limit)


@mcp.tool()
def build_rss_bridge_feed_url(
    bridge: str,
    params: dict[str, str] | None = None,
    format: str = "Atom",  # noqa: A002
) -> dict[str, str]:
    """Build an RSS-Bridge feed URL from a bridge id and parameters."""
    return build_rss_bridge_feed_url_data(bridge=bridge, params=params, format=format)


@mcp.tool()
def add_rss_bridge_source(
    name: str,
    bridge: str,
    params: dict[str, str] | None = None,
    category: str = "mixed",
    language: str = "unknown",
    format: str = "Atom",  # noqa: A002
) -> dict[str, Any]:
    """Build an RSS-Bridge feed URL and add it to data/sources.json."""
    return add_rss_bridge_source_data(
        name=name,
        bridge=bridge,
        params=params,
        category=parse_category(category),
        language=language,
        format=format,
    )


@mcp.tool()
def recommend_media(
    query: str,
    category: str = "all",
    refresh: bool = True,
    limit: int = 8,
    limit_per_source: int = 10,
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
    return str(list_watchlist_data(type="all", status="all"))


def main() -> None:
    configure_logging()
    get_langfuse_client()
    mcp.run()


if __name__ == "__main__":
    main()

