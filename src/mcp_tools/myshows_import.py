import logging
from collections.abc import Callable
from typing import Any, Protocol

from domain.models import Category
from myshows_client.settings import MyShowsSettings

logger = logging.getLogger(__name__)

ImportRecorder = Callable[[str, float, Category, dict[str, Any] | None], None]


class MyShowsHistoryClient(Protocol):
    """Задаёт минимальный контракт клиента для импорта истории."""

    def iter_all_watched_movies(self) -> list[dict[str, Any]]:
        """Возвращает просмотренные фильмы."""
        ...

    def get_shows(self) -> list[dict[str, Any]]:
        """Возвращает сериалы пользователя."""
        ...


def import_myshows_history_data(
    client: MyShowsHistoryClient,
    record_import: ImportRecorder,
    settings: MyShowsSettings,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Импортирует историю MyShows (просмотрено/брошено) в графовую память предпочтений.

    Фильмы взвешиваются по пользовательскому рейтингу (шкала 1-10):
    `>= 8` -> +3, `6-7` -> +1, `<= 5` -> -1, без рейтинга — пропуск.
    Сериалы — по статусу: смотрит/досмотрел -> +2, брошен -> -2, иначе пропуск.

    Args:
        client: Настроенный клиент MyShows.
        record_import: Инъецируемая функция записи заметки в память.
        settings: Настройки правил импорта MyShows.
        dry_run: Если True — только считает и возвращает сводку, ничего не пишет в память
            (для проверки маппинга полей на реальном ответе перед первым реальным запуском).

    Returns:
        Сводку импорта: `movies_processed`, `shows_processed`, `notes_recorded`
        (записанные заметки; при dry_run — заметки, которые были бы записаны)
        и `skipped` (элементы без полезного сигнала).

    Raises:
        RuntimeError: Если логин в MyShows или вызовы его API завершились ошибкой.
    """
    movies = client.iter_all_watched_movies()
    shows = client.get_shows()

    notes_recorded = 0
    skipped = 0

    for movie in movies:
        recorded = _record_note(
            text=_note_text(movie),
            weight=_movie_weight(movie),
            category=settings.movie_category,
            dry_run=dry_run,
            record_import=record_import,
            settings=settings,
        )
        notes_recorded += int(recorded)
        skipped += int(not recorded)

    for show in shows:
        recorded = _record_note(
            text=_note_text(show),
            weight=_show_weight(show, settings),
            category=settings.show_category,
            dry_run=dry_run,
            record_import=record_import,
            settings=settings,
        )
        notes_recorded += int(recorded)
        skipped += int(not recorded)

    summary = {
        "movies_processed": len(movies),
        "shows_processed": len(shows),
        "notes_recorded": notes_recorded,
        "skipped": skipped,
    }
    logger.info("MyShows history import finished", extra={**summary, "dry_run": dry_run})
    return summary


def _record_note(
    text: str,
    weight: int,
    category: Category,
    dry_run: bool,
    record_import: ImportRecorder,
    settings: MyShowsSettings,
) -> bool:
    """Записывает одну заметку импорта, если у элемента есть полезный сигнал.

    Args:
        text: Текст заметки (название плюс жанры).
        weight: Знаковый вес заметки; 0 означает отсутствие сигнала.
        category: Категория заметки.
        dry_run: Если True — заметка только учитывается, но не записывается.
        record_import: Инъецируемая функция записи заметки.
        settings: Настройки веса и source metadata.

    Returns:
        True, если заметка записана (или была бы записана при dry_run), иначе False.
    """
    if weight == 0 or not text:
        return False
    if not dry_run:
        record_import(text, weight * settings.import_weight_factor, category, {"source": settings.import_source})
    return True


def _movie_rating(item: dict[str, Any]) -> float | None:
    """Извлекает пользовательский рейтинг фильма (ожидается шкала 1-10).

    Args:
        item: Сырые данные фильма из `profile.WatchedMovies`.

    Returns:
        Рейтинг либо None, если пользователь не оценивал фильм.
    """
    for key in ("rating", "userRating", "myRating"):
        value = item.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None


def _movie_weight(item: dict[str, Any]) -> int:
    """Переводит рейтинг фильма в знаковый вес заметки.

    Args:
        item: Сырые данные фильма из `profile.WatchedMovies`.

    Returns:
        Вес заметки: +3, +1, -1 либо 0 (нет рейтинга — слишком слабый сигнал).
    """
    rating = _movie_rating(item)
    if rating is None:
        return 0
    if rating >= 8:
        return 3
    if rating >= 6:
        return 1
    return -1


def _show_status(item: dict[str, Any]) -> str | None:
    """Извлекает пользовательский статус просмотра сериала.

    Смотрит только на поля верхнего уровня: вложенный объект шоу может содержать
    эфирный статус самого сериала (например, отменён каналом), а не статус пользователя.

    Args:
        item: Сырые данные сериала из `profile.Shows`.

    Returns:
        Нормализованный статус в нижнем регистре либо None.
    """
    for key in ("watchStatus", "status"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _show_weight(item: dict[str, Any], settings: MyShowsSettings) -> int:
    """Переводит статус просмотра и оценку сериала в знаковый вес заметки.

    Явная пользовательская оценка (шкала 1-5, подтверждено на живом ответе)
    сильнее статуса: `>= 4` — любимый сериал, `<= 2` — не понравился,
    даже если досмотрен. Без оценки вес определяется статусом.

    Args:
        item: Сырые данные сериала из `profile.Shows`.
        settings: Настройки положительных и отрицательных статусов.

    Returns:
        Вес заметки: +3/-2 по оценке, +2 (смотрит/досмотрел),
        -2 (брошен) либо 0 (нет сигнала).
    """
    rating = _show_rating(item)
    if rating is not None and rating >= 4:
        return 3
    if rating is not None and rating <= 2:
        return -2
    status = _show_status(item)
    if status in settings.normalized_positive_show_statuses:
        return 2
    if status in settings.normalized_negative_show_statuses:
        return -2
    return 0


def _show_rating(item: dict[str, Any]) -> float | None:
    """Извлекает явную пользовательскую оценку сериала (шкала 1-5).

    Args:
        item: Сырые данные сериала из `profile.Shows`.

    Returns:
        Оценку либо None, если пользователь не оценивал сериал (rating 0).
    """
    value = item.get("rating")
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _note_text(item: dict[str, Any]) -> str:
    """Собирает текст заметки предпочтения из названия и жанров элемента.

    Если у элемента есть отличающееся оригинальное название, оно добавляется
    к локализованному: заметки сравниваются эмбеддингами с кандидатами на
    разных языках, и двуязычный текст матчится заметно лучше.

    Args:
        item: Сырые данные фильма или сериала.

    Returns:
        Текст заметки либо пустую строку, если название не найдено.
    """
    title = _item_title(item)
    if not title:
        return ""
    original = _item_original_title(item)
    if original and original != title:
        title = f"{title} / {original}"
    genres = _item_genres(item)
    if genres:
        return f"{title} ({', '.join(genres)})"
    return title


def _item_title(item: dict[str, Any]) -> str:
    """Извлекает название фильма или сериала.

    Args:
        item: Сырые данные элемента, возможно с вложенным объектом `show`/`movie`.

    Returns:
        Название либо пустую строку.
    """
    for container in _candidate_dicts(item):
        for key in ("title", "titleOriginal", "name"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _item_original_title(item: dict[str, Any]) -> str:
    """Извлекает оригинальное (не локализованное) название элемента.

    Args:
        item: Сырые данные элемента, возможно с вложенным объектом `show`/`movie`.

    Returns:
        Оригинальное название либо пустую строку.
    """
    for container in _candidate_dicts(item):
        value = container.get("titleOriginal")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _item_genres(item: dict[str, Any]) -> list[str]:
    """Извлекает список жанров фильма или сериала.

    Реальный API отдаёт жанры объектами вида `{"title": "Боевик", ...}`
    (подтверждено на живом ответе `profile.Shows`), но на всякий случай
    поддерживаются и plain-строки.

    Args:
        item: Сырые данные элемента, возможно с вложенным объектом `show`/`movie`.

    Returns:
        Непустые строковые жанры либо пустой список.
    """
    for container in _candidate_dicts(item):
        for key in ("genres", "genreTitles"):
            value = container.get(key)
            if not isinstance(value, list):
                continue
            genres = [genre for genre in (_genre_title(entry) for entry in value) if genre]
            if genres:
                return genres
    return []


def _genre_title(entry: Any) -> str:
    """Извлекает название жанра из строки или объекта жанра.

    Args:
        entry: Элемент списка жанров: строка либо объект с полем `title`.

    Returns:
        Название жанра либо пустую строку.
    """
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        title = entry.get("title") or entry.get("alias")
        if isinstance(title, str):
            return title.strip()
    return ""


def _candidate_dicts(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Возвращает словари, в которых стоит искать описательные поля элемента.

    Args:
        item: Сырые данные элемента.

    Returns:
        Сам элемент плюс вложенные объекты `show`/`movie`, если они есть.
    """
    containers = [item]
    for key in ("show", "movie"):
        nested = item.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    return containers
