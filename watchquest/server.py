from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from watchquest.tools.media import (
    add_to_watchlist_data,
    fetch_latest_items_data,
    get_profile_data,
    list_sources_data,
    list_watchlist_data,
    rate_watchlist_item_data,
    search_cached_items_data,
    update_profile_data,
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
    Fetch latest items from configured RSS sources and optional Feedly streams.

    category: games, movies, series, movies_series, mixed, or all.
    """
    return fetch_latest_items_data(category=category, limit_per_source=limit_per_source)


@mcp.tool()
def search_cached_items(query: str, category: str = "all", days: int = 30, limit: int = 20) -> list[dict[str, Any]]:
    """
    Search recently fetched cached items.

    Use bilingual queries when the user asks in Russian but sources may be English.
    Example query: "cozy RPG story rich уютная RPG сюжетная".
    """
    return search_cached_items_data(query=query, category=category, days=days, limit=limit)


@mcp.tool()
def add_to_watchlist(title: str, type: str = "unknown", url: str | None = None, reason: str = "", source: str | None = None) -> dict[str, Any]:
    """Add a game, movie, series, or article to the user's watchlist."""
    return add_to_watchlist_data(title=title, type=type, url=url, reason=reason, source=source)


@mcp.tool()
def list_watchlist(type: str = "all", status: str = "planned") -> list[dict[str, Any]]:
    """Return items from the user's watchlist."""
    return list_watchlist_data(type=type, status=status)


@mcp.tool()
def rate_watchlist_item(title: str, rating: int, comment: str = "") -> dict[str, Any]:
    """Rate an item from the watchlist from 1 to 10 and add a comment."""
    return rate_watchlist_item_data(title=title, rating=rating, comment=comment)


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
    mcp.run()


if __name__ == "__main__":
    main()
