def build_classification_prompt(entries: list[dict[str, str]]) -> str:
    """Собирает prompt батч-классификации RSS-записей.

    Args:
        entries: Записи батча, каждая с ключами `title` и `summary`.

    Returns:
        Prompt, требующий JSON-массив вердиктов в порядке записей.
    """
    lines = []
    for index, entry in enumerate(entries, start=1):
        line = f"{index}. title: {entry['title']}"
        if entry["summary"]:
            line += f" | summary: {entry['summary']}"
        lines.append(line)

    return (
        "Classify each RSS entry below.\n"
        "Return a JSON array with exactly one object per entry, in the same order:\n"
        '[{"kind": "...", "medium": "...", "title_entity": "..."}, ...]\n\n'
        "kind is one of:\n"
        "- review: a critic reviews or gives a verdict on one specific game, movie or series\n"
        "- release: a release, release date announcement or premiere of a specific title\n"
        "- noise: deals, sales, discounts, promo codes, shopping guides, hardware offers, giveaways\n"
        "- news: everything else (industry news, interviews, festivals, patches, rumors, esports)\n\n"
        "medium is the medium the entry is about, one of: anime, game, movie, series, other. "
        "Use anime for anime/manga adaptations, game for video games, movie for films, "
        "series for live-action TV shows; other when none clearly applies.\n\n"
        "title_entity is the exact name of the game/movie/series the entry is mainly about "
        '(keep the original language and spelling), or "" if the entry is not about one specific title.\n\n'
        "Entries:\n" + "\n".join(lines)
    )


def build_query_analysis_prompt(query: str) -> str:
    """Собирает prompt извлечения ограничений из запроса пользователя.

    Args:
        query: Пользовательский запрос рекомендации.

    Returns:
        Prompt, требующий JSON-объект с явно названным медиумом.
    """
    return (
        "Extract explicit constraints from this media recommendation request (Russian or English).\n"
        'Return a JSON object only: {"medium": "anime" | "game" | "movie" | "series" | null}\n'
        "Set medium ONLY if the user explicitly names the medium they want "
        "(anime, video game, film/movie, TV series); otherwise use null.\n\n"
        f'Request: "{query}"'
    )
