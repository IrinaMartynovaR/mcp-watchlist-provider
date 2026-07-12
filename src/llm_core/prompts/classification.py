CLASSIFIER_SYSTEM_PROMPT = (
    "You classify RSS entries from gaming and movie/series media (English and Russian). "
    "Respond with a JSON array only — no prose, no markdown code fences."
)


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
        '[{"kind": "...", "title_entity": "..."}, ...]\n\n'
        "kind is one of:\n"
        "- review: a critic reviews or gives a verdict on one specific game, movie or series\n"
        "- release: a release, release date announcement or premiere of a specific title\n"
        "- noise: deals, sales, discounts, promo codes, shopping guides, hardware offers, giveaways\n"
        "- news: everything else (industry news, interviews, festivals, patches, rumors, esports)\n\n"
        "title_entity is the exact name of the game/movie/series the entry is mainly about "
        '(keep the original language and spelling), or "" if the entry is not about one specific title.\n\n'
        "Entries:\n" + "\n".join(lines)
    )
