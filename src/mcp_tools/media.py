import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from langfuse import observe

from app.settings import backend_settings
from domain.models import Category, FeedItem, Source, WatchlistItem, category_matches, parse_media_type
from domain.storage.json_store import read_json, write_json
from mcp_tools.classification import classify_feed_items
from mcp_tools.settings import tool_settings
from rss_feeds.client import RSSFetchResult, fetch_rss_source_result
from rss_feeds.settings import rss_settings

logger = logging.getLogger(__name__)


RSS_BRIDGE_PLACEHOLDER = "{rss_bridge}"


def _load_sources() -> list[Source]:
    """Загружает RSS-источники из хранилища.

    Плейсхолдер `{rss_bridge}` в URL источника заменяется на адрес RSS-Bridge
    из настроек (`RSS_BRIDGE_URL`): локально это `http://localhost:3001`,
    а внутри Docker-сети — имя сервиса, которое прокидывает compose.

    Returns:
        Валидированный список источников.
    """
    data: dict[str, Any] = read_json(backend_settings.sources_file, {"feeds": []})
    bridge_url = rss_settings.normalized_rss_bridge_url
    sources = []
    for item in data.get("feeds", []):
        if isinstance(item.get("url"), str):
            item = {**item, "url": item["url"].replace(RSS_BRIDGE_PLACEHOLDER, bridge_url)}
        sources.append(Source.model_validate(item))
    logger.debug("RSS sources loaded", extra={"source_count": len(sources)})
    return sources


def _as_dicts(items: list[FeedItem]) -> list[dict[str, Any]]:
    """Преобразует RSS-элементы в JSON-совместимые словари.

    Args:
        items: Доменные RSS-элементы.

    Returns:
        Список сериализованных словарей.
    """
    return [item.model_dump(mode="json") for item in items]


def _dedupe(items: list[FeedItem]) -> list[FeedItem]:
    """Удаляет дубликаты RSS-элементов по URL.

    Args:
        items: Исходные RSS-элементы.

    Returns:
        Список без повторяющихся URL.
    """
    seen: set[str] = set()
    result: list[FeedItem] = []
    for item in items:
        key = item.url.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _matches_category(item: FeedItem, category: Category) -> bool:
    """Проверяет соответствие элемента заданной категории."""
    return category_matches(item.category, category)


def _matches_query(item: FeedItem, query: str) -> bool:
    """Проверяет наличие всех query-термов в содержимом RSS-элемента."""
    haystack = " ".join([item.title, item.summary, " ".join(item.tags)]).lower()
    terms = [term.strip().lower() for term in query.split() if term.strip()]
    return all(term in haystack for term in terms) if terms else True


def _source_matches_category(source: Source, category: Category) -> bool:
    """Проверяет соответствие RSS-источника заданной категории."""
    return category_matches(source.category, category)


def _rss_result_as_dict(result: RSSFetchResult) -> dict[str, Any]:
    """Преобразует результат RSS-fetch в диагностический словарь."""
    return {
        "name": result.source.name,
        "category": result.source.category,
        "language": result.source.language,
        "url": result.url,
        "ok": result.ok,
        "status_code": result.status_code,
        "item_count": len(result.items),
        "error": result.error,
    }


def _collect_rss_sources(category: Category, limit_per_source: int) -> tuple[list[FeedItem], list[dict[str, Any]]]:
    """Собирает RSS-элементы и диагностику по подходящим источникам.

    Args:
        category: Категория источников для выборки.
        limit_per_source: Максимальное число записей на источник.

    Returns:
        Кортеж из RSS-элементов и диагностик по источникам.
    """
    items: list[FeedItem] = []
    source_results: list[dict[str, Any]] = []

    for source in _load_sources():
        if not _source_matches_category(source, category):
            continue

        result = fetch_rss_source_result(source, limit=limit_per_source)
        items.extend(result.items)
        source_results.append(_rss_result_as_dict(result))

    logger.info(
        "RSS sources collected",
        extra={
            "category": category,
            "source_count": len(source_results),
            "item_count": len(items),
            "limit_per_source": limit_per_source,
        },
    )
    return items, source_results


def _preserved_cache_items(fetched_sources: set[str]) -> list[FeedItem]:
    """Возвращает кешированные RSS-элементы источников, не попавших в текущий refresh.

    Args:
        fetched_sources: Имена источников, обновлённых в текущем refresh.

    Returns:
        Элементы кеша от всех остальных источников.
    """
    cached: dict[str, Any] = read_json(backend_settings.cache_file, {"items": []})
    preserved = [FeedItem.model_validate(raw) for raw in cached.get("items", [])]
    return [item for item in preserved if item.source not in fetched_sources]


def candidate_key(item: dict[str, Any]) -> str:
    """Строит стабильный ключ RSS-кандидата для дедупликации.

    Args:
        item: RSS-кандидат.

    Returns:
        Ключ по URL, если он есть, иначе ключ по названию.
    """
    url = str(item.get("url") or "").strip().lower()
    if url:
        return f"url:{url}"
    return f"title:{str(item.get('title') or '').strip().lower()}"


@observe(name="get_profile", as_type="tool")
def get_profile_data() -> dict[str, Any]:
    """Возвращает профиль пользовательских предпочтений."""
    return read_json(backend_settings.profile_file, {})


def update_profile_data(likes: list[str] | None = None, dislikes: list[str] | None = None) -> dict[str, Any]:
    """Обновляет likes и dislikes профиля.

    Args:
        likes: Новые предпочтения пользователя.
        dislikes: Новые антипредпочтения пользователя.

    Returns:
        Обновлённый профиль.
    """
    profile: dict[str, Any] = read_json(backend_settings.profile_file, {})
    if likes:
        profile["likes"] = sorted(set(profile.get("likes", []) + likes))
    if dislikes:
        profile["dislikes"] = sorted(set(profile.get("dislikes", []) + dislikes))
    write_json(backend_settings.profile_file, profile)
    logger.info(
        "Profile updated",
        extra={"likes_added": len(likes or []), "dislikes_added": len(dislikes or [])},
    )
    return profile


def list_sources_data() -> list[dict[str, Any]]:
    """Возвращает сериализованный список RSS-источников."""
    return [source.model_dump(mode="json") for source in _load_sources()]


def validate_sources_data(
    category: Category = "all",
    limit_per_source: int = tool_settings.source_validation_limit,
) -> dict[str, Any]:
    """Проверяет доступность RSS-источников.

    Args:
        category: Категория источников.
        limit_per_source: Максимальное число записей на источник.

    Returns:
        Диагностику проверки по каждому источнику.
    """
    _, source_results = _collect_rss_sources(category=category, limit_per_source=limit_per_source)
    errors = [result for result in source_results if not result["ok"]]
    logger.info(
        "RSS sources validated",
        extra={"category": category, "checked_count": len(source_results), "error_count": len(errors)},
    )

    return {
        "ok": not errors and bool(source_results),
        "checked_count": len(source_results),
        "errors": errors,
        "sources": source_results,
    }


@observe(name="refresh_feeds", as_type="tool")
def refresh_feeds_data(
    category: Category = "all",
    limit_per_source: int = tool_settings.feed_refresh_limit,
) -> dict[str, Any]:
    """Обновляет RSS-кеш и сохраняет его на диск.

    Category-scoped refresh обновляет только источники своей категории;
    кешированные элементы остальных источников сохраняются, чтобы узкий
    refresh не стирал кандидатов других категорий.

    Args:
        category: Категория источников.
        limit_per_source: Максимальное число записей на источник.

    Returns:
        Результат refresh-операции вместе с ошибками источников.
    """
    items, source_results = _collect_rss_sources(category=category, limit_per_source=limit_per_source)
    if category != "all":
        fetched_sources = {result["name"] for result in source_results}
        items.extend(_preserved_cache_items(fetched_sources))
    items = _dedupe(items)
    items.sort(key=lambda item: item.published_at, reverse=True)
    items = classify_feed_items(items)

    payload = {
        "items": _as_dicts(items),
        "refreshed_at": datetime.now(UTC).isoformat(),
        "category": category,
        "sources": source_results,
    }
    write_json(backend_settings.cache_file, payload)

    errors = [result for result in source_results if not result["ok"]]
    logger.info(
        "RSS cache refreshed",
        extra={
            "category": category,
            "item_count": len(items),
            "source_count": len(source_results),
            "error_count": len(errors),
        },
    )

    return {
        "ok": bool(items),
        "total_count": len(items),
        "items": payload["items"],
        "sources": source_results,
        "errors": errors,
        "cache_file": str(backend_settings.cache_file),
    }


@observe(name="fetch_latest_items", as_type="tool")
def fetch_latest_items_data(
    category: Category = "all",
    limit_per_source: int = tool_settings.feed_refresh_limit,
) -> list[dict[str, Any]]:
    """Возвращает свежие RSS-элементы после обновления кеша.

    Args:
        category: Категория источников.
        limit_per_source: Максимальное число записей на источник.

    Returns:
        Список сериализованных RSS-элементов.
    """
    refreshed = refresh_feeds_data(category=category, limit_per_source=limit_per_source)
    items = refreshed["items"]
    if not isinstance(items, list):
        raise TypeError("Expected refresh_feeds_data to return a list of items")
    return [dict(item) for item in items]


@observe(name="search_cached_items", as_type="tool")
def search_cached_items_data(
    query: str,
    category: Category = "all",
    days: int = tool_settings.cache_search_days,
    limit: int = tool_settings.cache_search_limit,
) -> list[dict[str, Any]]:
    """Ищет элементы в локальном RSS-кеше.

    Args:
        query: Поисковая строка.
        category: Категория поиска.
        days: Глубина поиска по давности публикации.
        limit: Максимальное число результатов.

    Returns:
        Найденные RSS-элементы.
    """
    cached: dict[str, Any] = read_json(backend_settings.cache_file, {"items": []})
    cutoff = datetime.now(UTC) - timedelta(days=days)
    results: list[FeedItem] = []

    for raw in cached.get("items", []):
        item = FeedItem.model_validate(raw)
        if item.published_at < cutoff:
            continue
        if not _matches_category(item, category):
            continue
        if not _matches_query(item, query):
            continue
        results.append(item)

    logger.debug(
        "Cached RSS search completed",
        extra={"query": query, "category": category, "result_count": min(len(results), limit), "days": days},
    )
    return _as_dicts(results[:limit])


def add_to_watchlist_data(
    title: str,
    media_type: str = "unknown",
    url: str | None = None,
    reason: str = "",
    source: str | None = None,
) -> dict[str, Any]:
    """Добавляет элемент в watchlist пользователя.

    Args:
        title: Название сущности.
        media_type: Тип медиа.
        url: Необязательная ссылка.
        reason: Причина добавления.
        source: Источник рекомендации.

    Returns:
        Сериализованный watchlist item.
    """
    data: dict[str, Any] = read_json(backend_settings.watchlist_file, {"items": []})
    item = WatchlistItem(title=title, type=parse_media_type(media_type), url=url, reason=reason, source=source)
    data["items"].append(item.model_dump(mode="json"))
    write_json(backend_settings.watchlist_file, data)
    logger.info("Watchlist item added", extra={"title": title, "media_type": item.type})
    return item.model_dump(mode="json")


@observe(name="list_watchlist", as_type="tool")
def list_watchlist_data(media_type: str = "all", status: str = "planned") -> list[dict[str, Any]]:
    """Возвращает watchlist с фильтрацией по типу и статусу.

    Args:
        media_type: Тип медиа или `all`.
        status: Статус watchlist-элемента или `all`.

    Returns:
        Отфильтрованные watchlist items.
    """
    data: dict[str, Any] = read_json(backend_settings.watchlist_file, {"items": []})
    items = data.get("items", [])
    if media_type != "all":
        items = [item for item in items if item.get("type") == media_type]
    if status != "all":
        items = [item for item in items if item.get("status") == status]
    return [dict(item) for item in items]


def rate_watchlist_item_data(title: str, rating: int, comment: str = "") -> dict[str, Any]:
    """Сохраняет оценку и комментарий для элемента watchlist.

    Args:
        title: Название элемента для поиска.
        rating: Пользовательская оценка от 1 до 10.
        comment: Необязательный комментарий.

    Returns:
        Обновлённый watchlist item.

    Raises:
        ValueError: Если элемент с таким названием не найден.
    """
    data: dict[str, Any] = read_json(backend_settings.watchlist_file, {"items": []})
    for item in data.get("items", []):
        if item.get("title", "").lower() == title.lower():
            item["rating"] = rating
            item["comment"] = comment
            write_json(backend_settings.watchlist_file, data)
            logger.info("Watchlist item rated", extra={"title": item.get("title", title), "rating": rating})
            return dict(item)
    logger.warning("Watchlist item not found for rating", extra={"title": title})
    raise ValueError(f"Item not found in watchlist: {title}")

