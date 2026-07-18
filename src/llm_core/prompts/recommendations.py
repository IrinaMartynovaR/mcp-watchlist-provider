from typing import Any

from domain.models import Category


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
        "You are a single-shot recommender working ONLY with the candidates listed below: you cannot search "
        "the web or any other feeds, cannot access anything beyond these candidates, and cannot perform "
        "follow-up actions. Never offer to expand the search, never ask the user for permission to look "
        "elsewhere, never promise to do anything after this answer — if the feeds lack a match, say so plainly. "
        "Return 3-5 recommendations in the user's language whenever the candidates contain at least "
        "3 distinct relevant titles — do not stop at 1-2 out of caution. "
        "Use only the provided candidates. Do not invent titles that are not present in candidates. "
        "Candidates are pre-ranked by the user's overall taste profile, NOT by relevance to this query: "
        "scan the FULL candidate list and put query relevance first; `semantic_score` (when present) shows "
        "how close a candidate is to the query, and `query_match_medium` marks candidates that belong to "
        "the medium the user explicitly asked for — prefer those. "
        "A news, trailer, or interview item from a specialist source (e.g. an anime site) still points at a real "
        "title of that medium — recommend that title itself when it matches the request. "
        "Prefer candidates with a non-empty `title_entity`: that is the concrete recommendable title. "
        "Interviews and industry chatter without a title_entity are weak material — use them only "
        "when nothing better matches. "
        "Never misrepresent a candidate to satisfy the request: if the user asks for a specific medium "
        "or genre (e.g. anime) and, after checking every candidate, none actually is one, say honestly that "
        "the feeds have no real match right now and present the closest alternatives as alternatives, "
        "not as the thing asked for. "
        "Treat profile.learned_preferences as weighted user taste signals: positive values mean prefer, "
        "negative values mean avoid or down-rank. "
        "Each candidate has a `kind` field: prefer `review` (a critic covered this exact title) and "
        "`release` items as recommendation material; never recommend `noise` items (deals, sales, promos). "
        "If the user asks for games, recommend games mentioned in the candidates, not generic industry articles. "
        "Only if the candidates genuinely lack relevant material, recommend fewer than 3 "
        "and say the feed context is limited. "
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
