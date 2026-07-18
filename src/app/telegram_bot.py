import asyncio
import html
import logging
from datetime import UTC, datetime
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.application import WatchQuestApplication
from app.config import RuntimeConfig
from app.logging_config import configure_logging
from app.settings import BackendSettings
from domain.models import Category, category_matches
from domain.storage.json_store import read_json
from mcp_tools.feedback import apply_recommendation_feedback
from mcp_tools.media import list_watchlist_data

LOGGER = logging.getLogger(__name__)


def _parse_utc_datetime(value: Any) -> datetime | None:
    """Преобразует ISO timestamp кеша в timezone-aware UTC datetime.

    Args:
        value: Сериализованное значение времени.

    Returns:
        Нормализованное время либо `None` для невалидного значения.
    """
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def should_refresh_rss_cache(
    settings: BackendSettings,
    category: Category,
    now: datetime | None = None,
) -> bool:
    """Проверяет, нужно ли обновить RSS-кеш для Telegram-запроса.

    Args:
        settings: Backend-настройки с путём кеша и его TTL.
        category: Категория текущего Telegram-запроса.
        now: Текущее время для детерминированного тестирования.

    Returns:
        `True`, если кеш отсутствует, отключён TTL или файл устарел.
    """
    cache_file = settings.cache_file
    ttl_seconds = settings.telegram_rss_refresh_ttl_seconds
    if ttl_seconds <= 0 or not cache_file.exists():
        return True

    current_time = now or datetime.now(UTC)
    cached: dict[str, Any] = read_json(cache_file, {})
    raw_refresh_times = cached.get("refreshed_at_by_category")
    if not isinstance(raw_refresh_times, dict):
        refreshed_at = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC)
        return (current_time - refreshed_at).total_seconds() >= ttl_seconds

    matching_times = [
        parsed
        for cached_category, value in raw_refresh_times.items()
        if (cached_category == "all" or (category != "all" and category_matches(str(cached_category), category)))
        if (parsed := _parse_utc_datetime(value)) is not None
    ]
    if not matching_times:
        return True
    return (current_time - max(matching_times)).total_seconds() >= ttl_seconds


def infer_category(text: str) -> Category:
    """Определяет медиакатегорию по тексту Telegram-запроса.

    Args:
        text: Пользовательское сообщение.

    Returns:
        Категорию для дальнейшего recommendation pipeline.
    """
    normalized = text.lower()
    if any(word in normalized for word in ("игра", "игру", "игры", "game", "games")):
        return "games"
    if any(word in normalized for word in ("сериал", "сериалы", "series", "show")):
        return "series"
    if any(word in normalized for word in ("фильм", "кино", "movie", "film")):
        return "movies_series"
    return "all"


def split_telegram_text(text: str, settings: BackendSettings, limit: int | None = None) -> list[str]:
    """Разбивает длинный текст на Telegram-совместимые чанки.

    Args:
        text: Исходный текст ответа.
        settings: Backend-настройки Telegram.
        limit: Максимальная длина одного чанка.

    Returns:
        Непустые части текста, готовые к отправке.
    """
    message_limit = limit if limit is not None else settings.telegram_max_message_length
    if len(text) <= message_limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        chunk = remaining[:message_limit]
        split_at = chunk.rfind("\n\n")
        if split_at < message_limit // 2:
            split_at = chunk.rfind("\n")
        if split_at < message_limit // 2:
            split_at = message_limit

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    result = [chunk for chunk in chunks if chunk]
    LOGGER.debug("Telegram response split", extra={"chunk_count": len(result), "limit": message_limit})
    return result


def is_watchlist_request(text: str, settings: BackendSettings) -> bool:
    """Проверяет, просит ли пользователь показать watchlist.

    Args:
        text: Пользовательское сообщение.
        settings: Backend-настройки Telegram.

    Returns:
        `True`, если сообщение похоже на запрос watchlist.
    """
    normalized = text.lower()
    return any(phrase in normalized for phrase in settings.normalized_telegram_watchlist_request_phrases)


def recommendation_feedback_keyboard(recommendation_id: str, settings: BackendSettings) -> InlineKeyboardMarkup:
    """Создаёт inline-клавиатуру оценки рекомендации.

    Args:
        recommendation_id: Идентификатор сохранённой рекомендации.
        settings: Backend-настройки Telegram.

    Returns:
        Telegram inline-клавиатуру с feedback-действиями.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=settings.telegram_feedback_like_label,
                    callback_data=f"{settings.normalized_telegram_feedback_callback_prefix}:like:{recommendation_id}",
                ),
                InlineKeyboardButton(
                    text=settings.telegram_feedback_dislike_label,
                    callback_data=f"{settings.normalized_telegram_feedback_callback_prefix}:dislike:{recommendation_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=settings.telegram_feedback_watchlist_label,
                    callback_data=f"{settings.normalized_telegram_feedback_callback_prefix}:watchlist:{recommendation_id}",
                ),
                InlineKeyboardButton(
                    text=settings.telegram_feedback_block_similar_label,
                    callback_data=f"{settings.normalized_telegram_feedback_callback_prefix}:block_similar:{recommendation_id}",
                ),
            ],
        ]
    )


def format_recommendation_response(result: dict[str, Any], settings: BackendSettings) -> str:
    """Форматирует результат recommendation pipeline для Telegram.

    Args:
        result: Ответ `recommend_media_data`.
        settings: Backend-настройки Telegram.

    Returns:
        Пользовательский текст с рекомендацией и RSS-кандидатами.
    """
    recommendation = str(result.get("recommendation") or "").strip()
    candidates = result.get("candidates", [])
    candidate_lines: list[str] = []

    if isinstance(candidates, list) and candidates:
        for item in candidates[: settings.telegram_candidate_preview_limit]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Без названия").strip()
            source = str(item.get("source") or "unknown").strip()
            url = str(item.get("url") or "").strip()
            candidate_lines.append(
                f'• <a href="{html.escape(url, quote=True)}">{html.escape(title)}</a> — <i>{html.escape(source)}</i>'
            )

    parts = [recommendation or "Не смогла собрать рекомендацию по текущим источникам."]
    if candidate_lines:
        parts.append("Кандидаты из RSS:\n" + "\n".join(candidate_lines))

    return "\n\n".join(parts)


def format_watchlist_response(items: list[dict[str, Any]]) -> str:
    """Форматирует watchlist для Telegram.

    Args:
        items: Watchlist items из хранилища.

    Returns:
        Пользовательский текст со списком элементов.
    """
    if not items:
        return "Watchlist пока пустой."

    lines = ["Твой watchlist:"]
    for index, item in enumerate(items, start=1):
        title = str(item.get("title") or "Без названия").strip()
        media_type = str(item.get("type") or "unknown").strip()
        status = str(item.get("status") or "planned").strip()
        source = str(item.get("source") or "").strip()
        suffix = f" — {source}" if source else ""
        lines.append(f"{index}. {title} ({media_type}, {status}){suffix}")
    return "\n".join(lines)


async def answer_chunk(message: Message, chunk: str) -> None:
    """Отправляет один chunk с откатом на plain text при ошибке HTML-парсинга.

    LLM может нарушить инструкцию про Telegram-safe HTML, тогда Telegram
    вернёт ошибку "can't parse entities" — в этом случае тот же chunk
    отправляется повторно без разметки, чтобы пользователь не потерял ответ.

    Args:
        message: Входящее Telegram-сообщение, на которое отвечаем.
        chunk: Готовый фрагмент текста ответа.
    """
    try:
        await message.answer(chunk, disable_web_page_preview=True)
    except TelegramBadRequest:
        LOGGER.warning("Telegram HTML parse failed, resending chunk as plain text")
        await message.answer(chunk, parse_mode=None, disable_web_page_preview=True)


async def answer_recommendation(message: Message, query: str, application: WatchQuestApplication) -> None:
    """Обрабатывает один Telegram-запрос на рекомендацию.

    Args:
        message: Входящее Telegram-сообщение.
        query: Пользовательский текст запроса.
        application: Собранное приложение WatchQuest.
    """
    query = query.strip()
    if not query:
        await message.answer("Напиши запрос после команды, например: /recommend посоветуй вайбовую игру")
        return

    category = infer_category(query)
    settings = application.config.backend
    refresh = should_refresh_rss_cache(settings, category)
    status_text = (
        "Обновляю RSS и собираю рекомендацию." if refresh else "Использую свежий RSS-кеш и собираю рекомендацию."
    )
    await message.answer(status_text)
    LOGGER.info(
        "Telegram recommendation requested",
        extra={"category": category, "query": query, "refresh": refresh},
    )

    try:
        result = await asyncio.to_thread(
            application.recommendation.recommend,
            query=query,
            category=category,
            refresh=refresh,
            limit=application.config.backend.telegram_recommendation_limit,
            limit_per_source=application.config.backend.telegram_recommendation_source_limit,
        )
    except Exception:
        LOGGER.exception("Telegram recommendation failed")
        await message.answer("Не смогла собрать рекомендацию. Посмотри логи сервиса, там будет причина.")
        return

    for chunk in split_telegram_text(format_recommendation_response(result, settings), settings):
        await answer_chunk(message, chunk)
    recommendation_id = str(result.get("id") or "").strip()
    if recommendation_id:
        await message.answer(
            "Оцени рекомендацию, чтобы я лучше подстраивалась под твой вкус.",
            reply_markup=recommendation_feedback_keyboard(recommendation_id, settings),
        )
    LOGGER.info(
        "Telegram recommendation delivered",
        extra={"category": category, "candidate_count": result.get("candidate_count", 0)},
    )


async def answer_watchlist(message: Message, application: WatchQuestApplication) -> None:
    """Отправляет пользователю текущий watchlist.

    Args:
        message: Входящее Telegram-сообщение.
        application: Собранное приложение WatchQuest.
    """
    items = await asyncio.to_thread(
        list_watchlist_data,
        application.config,
        media_type="all",
        status="all",
    )
    for chunk in split_telegram_text(format_watchlist_response(items), application.config.backend):
        await answer_chunk(message, chunk)


async def answer_feedback(callback: CallbackQuery, application: WatchQuestApplication) -> None:
    """Обрабатывает inline feedback по рекомендации.

    Args:
        callback: Telegram callback query от inline-кнопки.
        application: Собранное приложение WatchQuest.
    """
    raw_data = callback.data or ""
    parts = raw_data.split(":", maxsplit=2)
    if len(parts) != 3:
        await callback.answer("Некорректная кнопка feedback.", show_alert=True)
        return
    _, action, recommendation_id = parts
    try:
        result = await asyncio.to_thread(
            apply_recommendation_feedback,
            application.config,
            recommendation_id,
            action,
            application.memory.record_preference,
        )
    except Exception:
        LOGGER.exception("Telegram feedback failed")
        await callback.answer("Не смогла сохранить оценку. Посмотри логи.", show_alert=True)
        return

    messages = {
        "like": application.config.backend.telegram_feedback_like_response,
        "dislike": application.config.backend.telegram_feedback_dislike_response,
        "watchlist": application.config.backend.telegram_feedback_watchlist_response,
        "block_similar": application.config.backend.telegram_feedback_block_similar_response,
    }
    await callback.answer(messages.get(str(result["action"]), "Оценка сохранена."))


def create_dispatcher(application: WatchQuestApplication) -> Dispatcher:
    """Создаёт и конфигурирует aiogram dispatcher.

    Args:
        application: Собранное приложение WatchQuest.

    Returns:
        Dispatcher с зарегистрированными обработчиками команд и текста.
    """
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start_handler(message: Message) -> None:
        await message.answer(application.config.backend.telegram_welcome_text)

    @dp.message(Command("help"))
    async def help_handler(message: Message) -> None:
        await message.answer(application.config.backend.telegram_help_text)

    @dp.message(Command("recommend"))
    async def recommend_handler(message: Message) -> None:
        text = message.text or ""
        query = text.partition(" ")[2]
        await answer_recommendation(message, query, application)

    @dp.message(Command("watchlist"))
    async def watchlist_handler(message: Message) -> None:
        await answer_watchlist(message, application)

    @dp.callback_query(F.data.startswith(f"{application.config.backend.normalized_telegram_feedback_callback_prefix}:"))
    async def feedback_handler(callback: CallbackQuery) -> None:
        await answer_feedback(callback, application)

    @dp.message(F.text)
    async def text_handler(message: Message) -> None:
        text = message.text or ""
        if is_watchlist_request(text, application.config.backend):
            await answer_watchlist(message, application)
            return
        await answer_recommendation(message, text, application)

    return dp


async def run_bot(application: WatchQuestApplication) -> None:
    """Запускает Telegram polling lifecycle.

    Args:
        application: Собранное приложение WatchQuest.

    Raises:
        RuntimeError: Если токен Telegram-бота отсутствует.
    """
    settings = application.config.backend
    if not settings.normalized_telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    bot = Bot(
        token=settings.normalized_telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = create_dispatcher(application)
    await bot.delete_webhook(drop_pending_updates=True)
    LOGGER.info("Telegram bot polling started")

    try:
        await dispatcher.start_polling(bot)
    finally:
        LOGGER.info("Telegram bot polling stopped")
        await bot.session.close()


def main() -> None:
    """Точка входа CLI-команды Telegram-бота."""
    config = RuntimeConfig.from_env()
    configure_logging(config.backend)
    asyncio.run(run_bot(WatchQuestApplication(config)))


if __name__ == "__main__":
    main()
