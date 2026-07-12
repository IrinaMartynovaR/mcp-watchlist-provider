from typing import Any

from domain.models import Category

HYDE_SYSTEM_PROMPT = (
    "You write short hypothetical article snippets used only for semantic search matching, "
    "never shown to the user. Write in English, 2-4 sentences, in the style of a gaming/movie "
    "news or review article. Do not mention that it's hypothetical."
)

SYSTEM_PROMPT = (
    "You are WatchQuest, a concise bilingual media recommendation assistant. "
    "Recommend games, movies, series, and articles using the user's taste profile, watchlist, and recent feed items. "
    "Prefer concrete titles and explain why each recommendation fits. Answer in the user's language. "
    "Format your answer as Telegram-safe HTML: the only allowed tags are <b>, <i>, "
    'and <a href="...">. Never use Markdown syntax (**bold**, [text](url), headings) or any other HTML tags.'
)


def build_hyde_prompt(query: str, category: Category) -> str:
    """Собирает prompt для генерации гипотетического документа (HyDE).

    Args:
        query: Пользовательский запрос.
        category: Категория поиска.

    Returns:
        Prompt для служебной генерации, не показываемой пользователю.
    """
    return (
        "Write a short hypothetical article passage (2-4 sentences, English) that would perfectly "
        f'match this request: "{query}" (category: {category}). Focus on genre, mood, and vibe '
        "so the passage reads like a real article about a matching game/movie/series."
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
        "Treat profile.learned_preferences as weighted user taste signals: positive values mean prefer, "
        "negative values mean avoid or down-rank. "
        "If the user asks for games, recommend games mentioned in the candidates, not generic industry articles. "
        "If there are fewer than 3 solid matches, recommend fewer and say the feed context is limited. "
        "For each recommendation include title, type/category, why it fits, and a concrete next action. "
        "Explain why it fits in 2-3 full sentences per recommendation: how it matches the query, mood, or vibe, "
        "and who will enjoy it — do not compress the reasoning into one short clipped phrase.\n"
        "Formatting rules (Telegram HTML): render each recommended title as a clickable link using that "
        'candidate\'s own `url` field, exactly like <a href="URL">Title</a>. '
        'Use only Telegram-supported HTML tags: <b>, <i>, and <a href="...">. '
        "Do not use any other HTML tags and no Markdown syntax (**bold**, [text](url), etc.) "
        "anywhere in the answer.\n\n"
        f"User query: {query}\n"
        f"Category filter: {category}\n"
        f"Profile: {profile}\n"
        f"Watchlist: {watchlist[:20]}\n"
        f"Candidates: {candidates}"
    )
