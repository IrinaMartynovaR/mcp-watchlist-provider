"""Эвристическая классификация RSS-записей по типу контента.

RSS-фиды СМИ смешивают обзоры конкретных тайтлов с индустриальными новостями
и рекламой скидок. Для рекомендаций эти типы неравноценны: обзор — сильный
сигнал «есть что порекомендовать», скидочный пост — шум. Классификация
дешёвая (ключевые слова, без LLM) и выполняется один раз при загрузке фида.
"""

import re

from domain.models import ItemKind

# Английские однословные маркеры — только по границам слов ("deal" не должен
# ловить "ideal"/"ordeal"); русские — по основе слова без границ, чтобы
# покрыть словоформы ("скидк" -> скидка/скидки/скидок).
_NOISE_RE = re.compile(
    # "sale" строго в единственном числе: "sales" — это новости о продажах
    # ("2 million sales"), а не распродажа.
    r"\b(?:deals?|sale|discounts?|coupons?|restocks?|pre-?orders?)\b"
    r"|% off|black friday|cyber monday|prime day|gift guide|promo code"
    r"|скидк|распродаж|промокод|предзаказ",
    re.IGNORECASE,
)

_REVIEW_TITLE_RE = re.compile(r"\b(?:reviews?|impressions|verdict)\b|обзор|рецензи", re.IGNORECASE)
# Ловит и сегмент пути (/reviews/...), и slug-стиль (...-tv-review-2026).
_REVIEW_URL_RE = re.compile(r"[/-]reviews?\b", re.IGNORECASE)
_REVIEW_TAGS = frozenset({"review", "reviews", "обзор", "обзоры", "рецензия", "рецензии"})

_RELEASE_RE = re.compile(
    r"\b(?:release date|out now|now available|launches)\b"
    r"|премьер|релиз|дата выхода|вышел|вышла|вышло",
    re.IGNORECASE,
)


def classify_item_kind(title: str, url: str = "", summary: str = "", tags: list[str] | None = None) -> ItemKind:
    """Определяет тип RSS-записи по ключевым словам заголовка, URL и тегов.

    Порядок проверок важен: шум (скидки/промо) распознаётся первым, потому что
    заголовки вида "Best deals on..." часто содержат и название игры; затем
    обзоры, затем релизы, всё остальное — новости.

    Args:
        title: Заголовок записи.
        url: Ссылка записи.
        summary: Аннотация записи (уже очищенная от HTML).
        tags: Теги записи.

    Returns:
        Тип записи: `review`, `release`, `news` или `noise`.
    """
    if _NOISE_RE.search(title):
        return "noise"

    if (
        _REVIEW_TITLE_RE.search(title)
        or _REVIEW_URL_RE.search(url)
        or any(tag.lower() in _REVIEW_TAGS for tag in tags or [])
    ):
        return "review"

    if _RELEASE_RE.search(title) or _RELEASE_RE.search(summary):
        return "release"

    return "news"
