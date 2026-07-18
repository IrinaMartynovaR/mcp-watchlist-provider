import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

from langfuse import observe

from app.config import RuntimeConfig
from domain.models import Category, FeedItem, Source, WatchlistItem, category_matches, parse_media_type
from domain.repositories.profile import ProfileRepository
from domain.repositories.watchlist import WatchlistRepository
from domain.storage.json_store import read_json, update_json
from mcp_tools.classification import classify_feed_items
from rss_feeds.client import RSSFetchResult, fetch_rss_source_result

logger = logging.getLogger(__name__)


RSSFetcher = Callable[..., RSSFetchResult]
FeedClassifier = Callable[..., list[FeedItem]]


def _load_sources(config: RuntimeConfig) -> list[Source]:
    """Загружает RSS-источники из хранилища.

    Плейсхолдер `{rss_bridge}` в URL источника заменяется на адрес RSS-Bridge
    из настроек (`RSS_BRIDGE_URL`): локально это `http://localhost:3001`,
    а внутри Docker-сети — имя сервиса, которое прокидывает compose.

    Returns:
        Валидированный список источников.
    """
    data: dict[str, Any] = read_json(config.backend.sources_file, {"feeds": []})
    bridge_url = config.rss.normalized_rss_bridge_url
    sources = []
    for item in data.get("feeds", []):
        if isinstance(item.get("url"), str):
            item = {**item, "url": item["url"].replace(config.rss.rss_bridge_placeholder, bridge_url)}
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


def _collect_rss_sources(
    config: RuntimeConfig,
    category: Category,
    limit_per_source: int,
    fetch_source: RSSFetcher,
) -> tuple[list[FeedItem], list[dict[str, Any]]]:
    """Собирает RSS-элементы и диагностику по подходящим источникам.

    Args:
        config: Runtime-настройки приложения.
        category: Категория источников для выборки.
        limit_per_source: Максимальное число записей на источник.
        fetch_source: Инъецируемая функция загрузки RSS.

    Returns:
        Кортеж из RSS-элементов и диагностик по источникам.
    """
    sources = [source for source in _load_sources(config) if _source_matches_category(source, category)]
    worker_count = min(len(sources), max(1, config.rss.rss_fetch_concurrency))

    def fetch(source: Source) -> RSSFetchResult:
        """Загружает один источник с общими runtime-настройками.

        Args:
            source: RSS-источник текущей категории.

        Returns:
            Нормализованный результат загрузки источника.
        """
        return fetch_source(source, config.rss, limit=limit_per_source)

    if worker_count <= 1:
        results = [fetch(source) for source in sources]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="rss-fetch") as executor:
            results = list(executor.map(fetch, sources))

    items: list[FeedItem] = []
    source_results: list[dict[str, Any]] = []
    for result in results:
        items.extend(result.items)
        source_results.append(_rss_result_as_dict(result))

    logger.info(
        "RSS sources collected",
        extra={
            "category": category,
            "source_count": len(source_results),
            "item_count": len(items),
            "limit_per_source": limit_per_source,
            "concurrency": worker_count,
        },
    )
    return items, source_results


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
def get_profile_data(config: RuntimeConfig) -> dict[str, Any]:
    """Возвращает профиль пользовательских предпочтений.

    Args:
        config: Runtime-настройки приложения.

    Returns:
        Словарь профиля.
    """
    return ProfileRepository(config.backend.profile_file).get()


def update_profile_data(
    config: RuntimeConfig,
    likes: list[str] | None = None,
    dislikes: list[str] | None = None,
) -> dict[str, Any]:
    """Обновляет likes и dislikes профиля.

    Args:
        config: Runtime-настройки приложения.
        likes: Новые предпочтения пользователя.
        dislikes: Новые антипредпочтения пользователя.

    Returns:
        Обновлённый профиль.
    """
    profile = ProfileRepository(config.backend.profile_file).add_preferences(likes=likes, dislikes=dislikes)
    logger.info(
        "Profile updated",
        extra={"likes_added": len(likes or []), "dislikes_added": len(dislikes or [])},
    )
    return profile


def list_sources_data(config: RuntimeConfig) -> list[dict[str, Any]]:
    """Возвращает сериализованный список RSS-источников.

    Args:
        config: Runtime-настройки приложения.

    Returns:
        Настроенные RSS-источники.
    """
    return [source.model_dump(mode="json") for source in _load_sources(config)]


def validate_sources_data(
    config: RuntimeConfig,
    category: Category = "all",
    limit_per_source: int | None = None,
    fetch_source: RSSFetcher = fetch_rss_source_result,
) -> dict[str, Any]:
    """Проверяет доступность RSS-источников.

    Args:
        config: Runtime-настройки приложения.
        category: Категория источников.
        limit_per_source: Максимальное число записей на источник.
        fetch_source: Инъецируемая функция загрузки RSS.

    Returns:
        Диагностику проверки по каждому источнику.
    """
    source_limit = limit_per_source if limit_per_source is not None else config.tools.source_validation_limit
    _, source_results = _collect_rss_sources(config, category, source_limit, fetch_source)
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
    config: RuntimeConfig,
    category: Category = "all",
    limit_per_source: int | None = None,
    fetch_source: RSSFetcher = fetch_rss_source_result,
    classify_items: FeedClassifier = classify_feed_items,
) -> dict[str, Any]:
    """Обновляет RSS-кеш и сохраняет его на диск.

    Category-scoped refresh обновляет только источники своей категории;
    кешированные элементы остальных источников сохраняются, чтобы узкий
    refresh не стирал кандидатов других категорий.

    Args:
        config: Runtime-настройки приложения.
        category: Категория источников.
        limit_per_source: Максимальное число записей на источник.
        fetch_source: Инъецируемая функция загрузки RSS.
        classify_items: Инъецируемая функция классификации записей.

    Returns:
        Результат refresh-операции вместе с ошибками источников.
    """
    source_limit = limit_per_source if limit_per_source is not None else config.tools.feed_refresh_limit
    fetched_items, source_results = _collect_rss_sources(config, category, source_limit, fetch_source)
    fetched_items = classify_items(_dedupe(fetched_items), config)
    fetched_sources = {result["name"] for result in source_results}
    refreshed_at = datetime.now(UTC).isoformat()

    def commit(cached: dict[str, Any]) -> dict[str, Any]:
        items = list(fetched_items)
        if category != "all":
            preserved = [FeedItem.model_validate(raw) for raw in cached.get("items", [])]
            items.extend(item for item in preserved if item.source not in fetched_sources)
        items = _dedupe(items)
        items.sort(key=lambda item: item.published_at, reverse=True)
        raw_refresh_times = cached.get("refreshed_at_by_category", {})
        refresh_times = dict(raw_refresh_times) if isinstance(raw_refresh_times, dict) else {}
        if category == "all":
            refresh_times = {"all": refreshed_at}
        else:
            refresh_times[category] = refreshed_at
        return {
            "items": _as_dicts(items),
            "refreshed_at": refreshed_at,
            "refreshed_at_by_category": refresh_times,
            "category": category,
            "sources": source_results,
        }

    payload: dict[str, Any] = update_json(config.backend.cache_file, {"items": []}, commit)

    errors = [result for result in source_results if not result["ok"]]
    logger.info(
        "RSS cache refreshed",
        extra={
            "category": category,
            "item_count": len(payload["items"]),
            "source_count": len(source_results),
            "error_count": len(errors),
        },
    )

    return {
        "ok": bool(payload["items"]),
        "total_count": len(payload["items"]),
        "items": payload["items"],
        "sources": source_results,
        "errors": errors,
        "cache_file": str(config.backend.cache_file),
    }


@observe(name="fetch_latest_items", as_type="tool")
def fetch_latest_items_data(
    config: RuntimeConfig,
    category: Category = "all",
    limit_per_source: int | None = None,
    fetch_source: RSSFetcher = fetch_rss_source_result,
    classify_items: FeedClassifier = classify_feed_items,
) -> list[dict[str, Any]]:
    """Возвращает свежие RSS-элементы после обновления кеша.

    Args:
        config: Runtime-настройки приложения.
        category: Категория источников.
        limit_per_source: Максимальное число записей на источник.
        fetch_source: Инъецируемая функция загрузки RSS.
        classify_items: Инъецируемая функция классификации записей.

    Returns:
        Список сериализованных RSS-элементов.
    """
    refreshed = refresh_feeds_data(
        config,
        category=category,
        limit_per_source=limit_per_source,
        fetch_source=fetch_source,
        classify_items=classify_items,
    )
    items = refreshed["items"]
    if not isinstance(items, list):
        raise TypeError("Expected refresh_feeds_data to return a list of items")
    return [dict(item) for item in items]


@observe(name="search_cached_items", as_type="tool")
def search_cached_items_data(
    config: RuntimeConfig,
    query: str,
    category: Category = "all",
    days: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Ищет элементы в локальном RSS-кеше.

    Args:
        config: Runtime-настройки приложения.
        query: Поисковая строка.
        category: Категория поиска.
        days: Глубина поиска по давности публикации.
        limit: Максимальное число результатов.

    Returns:
        Найденные RSS-элементы.
    """
    search_days = days if days is not None else config.tools.cache_search_days
    search_limit = limit if limit is not None else config.tools.cache_search_limit
    cached: dict[str, Any] = read_json(config.backend.cache_file, {"items": []})
    cutoff = datetime.now(UTC) - timedelta(days=search_days)
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
        extra={
            "query": query,
            "category": category,
            "result_count": min(len(results), search_limit),
            "days": search_days,
        },
    )
    return _as_dicts(results[:search_limit])


def add_to_watchlist_data(
    config: RuntimeConfig,
    title: str,
    media_type: str = "unknown",
    url: str | None = None,
    reason: str = "",
    source: str | None = None,
) -> dict[str, Any]:
    """Добавляет элемент в watchlist пользователя.

    Args:
        config: Runtime-настройки приложения.
        title: Название сущности.
        media_type: Тип медиа.
        url: Необязательная ссылка.
        reason: Причина добавления.
        source: Источник рекомендации.

    Returns:
        Сериализованный watchlist item.
    """
    item = WatchlistItem(title=title, type=parse_media_type(media_type), url=url, reason=reason, source=source)
    serialized = WatchlistRepository(config.backend.watchlist_file).add(item)
    logger.info("Watchlist item added", extra={"title": title, "media_type": item.type})
    return serialized


@observe(name="list_watchlist", as_type="tool")
def list_watchlist_data(
    config: RuntimeConfig,
    media_type: str = "all",
    status: str = "planned",
) -> list[dict[str, Any]]:
    """Возвращает watchlist с фильтрацией по типу и статусу.

    Args:
        config: Runtime-настройки приложения.
        media_type: Тип медиа или `all`.
        status: Статус watchlist-элемента или `all`.

    Returns:
        Отфильтрованные watchlist items.
    """
    return WatchlistRepository(config.backend.watchlist_file).list(media_type=media_type, status=status)


def rate_watchlist_item_data(
    config: RuntimeConfig,
    title: str,
    rating: int,
    comment: str = "",
) -> dict[str, Any]:
    """Сохраняет оценку и комментарий для элемента watchlist.

    Args:
        config: Runtime-настройки приложения.
        title: Название элемента для поиска.
        rating: Пользовательская оценка от 1 до 10.
        comment: Необязательный комментарий.

    Returns:
        Обновлённый watchlist item.

    Raises:
        ValueError: Если элемент с таким названием не найден.
    """
    try:
        item = WatchlistRepository(config.backend.watchlist_file).rate(
            title=title,
            rating=rating,
            comment=comment,
        )
    except ValueError:
        logger.warning("Watchlist item not found for rating", extra={"title": title})
        raise
    logger.info("Watchlist item rated", extra={"title": item.get("title", title), "rating": rating})
    return item
