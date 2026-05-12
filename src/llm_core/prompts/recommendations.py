from typing import Any

from domain.models import Category

SYSTEM_PROMPT = (
    "You are WatchQuest, a concise bilingual media recommendation assistant. "
    "Recommend games, movies, series, and articles using the user's taste profile, watchlist, and recent feed items. "
    "Prefer concrete titles and explain why each recommendation fits. Answer in the user's language."
)


def build_feed_recommendation_prompt(
    query: str,
    category: Category,
    profile: dict[str, Any],
    watchlist: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    """Собирает prompt для рекомендации на базе RSS-контекста.

    Args:
        query: Исходный пользовательский запрос.
        category: Нормализованная категория поиска.
        profile: Профиль вкусов пользователя.
        watchlist: Уже сохранённые элементы watchlist.
        candidates: Кандидаты, найденные в RSS-кеше.

    Returns:
        Финальный текст prompt для LLM.
    """
    return (
        "You are creating a practical WatchQuest recommendation. "
        "Return 1-5 recommendations in the user's language. "
        "Use only the provided candidates. Do not invent titles that are not present in candidates. "
        "If the user asks for games, recommend games mentioned in the candidates, not generic industry articles. "
        "If there are fewer than 3 solid matches, recommend fewer and say the feed context is limited. "
        "For each recommendation include title, type/category, why it fits, and a concrete next action.\n\n"
        f"User query: {query}\n"
        f"Category filter: {category}\n"
        f"Profile: {profile}\n"
        f"Watchlist: {watchlist[:20]}\n"
        f"Candidates: {candidates}"
    )
