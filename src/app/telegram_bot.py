import asyncio
import html
import logging
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.logging_config import configure_logging
from app.settings import backend_settings
from domain.models import Category
from mcp_tools.feedback import apply_recommendation_feedback
from mcp_tools.media import list_watchlist_data
from mcp_tools.recommendation import recommend_media_data

LOGGER = logging.getLogger(__name__)

WELCOME_TEXT = (
    "Привет! Я WatchQuest.\n\n"
    "Напиши, что хочется посмотреть или во что поиграть, а я проверю RSS-источники и соберу рекомендацию.\n\n"
    "Примеры:\n"
    "- посоветуй вайбовую игру\n"
    "- хочу сериал на вечер\n"
    "- найди что-нибудь атмосферное про sci-fi"
)

HELP_TEXT = (
    "Команды:\n"
    "/start - начать\n"
    "/help - помощь\n"
    "/recommend <запрос> - рекомендация через RSS и LLM\n\n"
    "/watchlist - показать watchlist\n\n"
    "Можно просто написать запрос обычным сообщением."
)

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


def split_telegram_text(text: str, limit: int = backend_settings.telegram_max_message_length) -> list[str]:
    """Разбивает длинный текст на Telegram-совместимые чанки.

    Args:
        text: Исходный текст ответа.
        limit: Максимальная длина одного чанка.

    Returns:
        Непустые части текста, готовые к отправке.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        chunk = remaining[:limit]
        split_at = chunk.rfind("\n\n")
        if split_at < limit // 2:
            split_at = chunk.rfind("\n")
        if split_at < limit // 2:
            split_at = limit

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    result = [chunk for chunk in chunks if chunk]
    LOGGER.debug("Telegram response split", extra={"chunk_count": len(result), "limit": limit})
    return result


def is_watchlist_request(text: str) -> bool:
    """Проверяет, просит ли пользователь показать watchlist.

    Args:
        text: Пользовательское сообщение.

    Returns:
        `True`, если сообщение похоже на запрос watchlist.
    """
    normalized = text.lower()
    return any(phrase in normalized for phrase in backend_settings.normalized_telegram_watchlist_request_phrases)


def recommendation_feedback_keyboard(recommendation_id: str) -> InlineKeyboardMarkup:
    """Создаёт inline-клавиатуру оценки рекомендации.

    Args:
        recommendation_id: Идентификатор сохранённой рекомендации.

    Returns:
        Telegram inline-клавиатуру с feedback-действиями.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=backend_settings.telegram_feedback_like_label,
                    callback_data=f"{backend_settings.normalized_telegram_feedback_callback_prefix}:like:{recommendation_id}",
                ),
                InlineKeyboardButton(
                    text=backend_settings.telegram_feedback_dislike_label,
                    callback_data=f"{backend_settings.normalized_telegram_feedback_callback_prefix}:dislike:{recommendation_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=backend_settings.telegram_feedback_watchlist_label,
                    callback_data=f"{backend_settings.normalized_telegram_feedback_callback_prefix}:watchlist:{recommendation_id}",
                ),
                InlineKeyboardButton(
                    text=backend_settings.telegram_feedback_block_similar_label,
                    callback_data=f"{backend_settings.normalized_telegram_feedback_callback_prefix}:block_similar:{recommendation_id}",
                ),
            ],
        ]
    )


def format_recommendation_response(result: dict[str, Any]) -> str:
    """Форматирует результат recommendation pipeline для Telegram.

    Args:
        result: Ответ `recommend_media_data`.

    Returns:
        Пользовательский текст с рекомендацией и RSS-кандидатами.
    """
    recommendation = str(result.get("recommendation") or "").strip()
    candidates = result.get("candidates", [])
    candidate_lines: list[str] = []

    if isinstance(candidates, list) and candidates:
        for item in candidates[: backend_settings.telegram_candidate_preview_limit]:
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


async def answer_recommendation(message: Message, query: str) -> None:
    """Обрабатывает один Telegram-запрос на рекомендацию.

    Args:
        message: Входящее Telegram-сообщение.
        query: Пользовательский текст запроса.
    """
    query = query.strip()
    if not query:
        await message.answer("Напиши запрос после команды, например: /recommend посоветуй вайбовую игру")
        return

    await message.answer("Сейчас проверю RSS и соберу рекомендацию.")
    category = infer_category(query)
    LOGGER.info("Telegram recommendation requested", extra={"category": category, "query": query})

    try:
        result = await asyncio.to_thread(
            recommend_media_data,
            query=query,
            category=category,
            refresh=True,
            limit=backend_settings.telegram_recommendation_limit,
            limit_per_source=backend_settings.telegram_recommendation_source_limit,
        )
    except Exception:
        LOGGER.exception("Telegram recommendation failed")
        await message.answer("Не смогла собрать рекомендацию. Посмотри логи сервиса, там будет причина.")
        return

    for chunk in split_telegram_text(format_recommendation_response(result)):
        await answer_chunk(message, chunk)
    recommendation_id = str(result.get("id") or "").strip()
    if recommendation_id:
        await message.answer(
            "Оцени рекомендацию, чтобы я лучше подстраивалась под твой вкус.",
            reply_markup=recommendation_feedback_keyboard(recommendation_id),
        )
    LOGGER.info(
        "Telegram recommendation delivered",
        extra={"category": category, "candidate_count": result.get("candidate_count", 0)},
    )


async def answer_watchlist(message: Message) -> None:
    """Отправляет пользователю текущий watchlist.

    Args:
        message: Входящее Telegram-сообщение.
    """
    items = await asyncio.to_thread(list_watchlist_data, media_type="all", status="all")
    for chunk in split_telegram_text(format_watchlist_response(items)):
        await answer_chunk(message, chunk)


async def answer_feedback(callback: CallbackQuery) -> None:
    """Обрабатывает inline feedback по рекомендации.

    Args:
        callback: Telegram callback query от inline-кнопки.
    """
    raw_data = callback.data or ""
    parts = raw_data.split(":", maxsplit=2)
    if len(parts) != 3:
        await callback.answer("Некорректная кнопка feedback.", show_alert=True)
        return
    _, action, recommendation_id = parts
    try:
        result = await asyncio.to_thread(apply_recommendation_feedback, recommendation_id, action)
    except Exception:
        LOGGER.exception("Telegram feedback failed")
        await callback.answer("Не смогла сохранить оценку. Посмотри логи.", show_alert=True)
        return

    messages = {
        "like": backend_settings.telegram_feedback_like_response,
        "dislike": backend_settings.telegram_feedback_dislike_response,
        "watchlist": backend_settings.telegram_feedback_watchlist_response,
        "block_similar": backend_settings.telegram_feedback_block_similar_response,
    }
    await callback.answer(messages.get(str(result["action"]), "Оценка сохранена."))


def create_dispatcher() -> Dispatcher:
    """Создаёт и конфигурирует aiogram dispatcher.

    Returns:
        Dispatcher с зарегистрированными обработчиками команд и текста.
    """
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start_handler(message: Message) -> None:
        await message.answer(WELCOME_TEXT)

    @dp.message(Command("help"))
    async def help_handler(message: Message) -> None:
        await message.answer(HELP_TEXT)

    @dp.message(Command("recommend"))
    async def recommend_handler(message: Message) -> None:
        text = message.text or ""
        query = text.partition(" ")[2]
        await answer_recommendation(message, query)

    @dp.message(Command("watchlist"))
    async def watchlist_handler(message: Message) -> None:
        await answer_watchlist(message)

    @dp.callback_query(F.data.startswith(f"{backend_settings.normalized_telegram_feedback_callback_prefix}:"))
    async def feedback_handler(callback: CallbackQuery) -> None:
        await answer_feedback(callback)

    @dp.message(F.text)
    async def text_handler(message: Message) -> None:
        text = message.text or ""
        if is_watchlist_request(text):
            await answer_watchlist(message)
            return
        await answer_recommendation(message, text)

    return dp


async def run_bot() -> None:
    """Запускает Telegram polling lifecycle.

    Raises:
        RuntimeError: Если токен Telegram-бота отсутствует.
    """
    if not backend_settings.normalized_telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    bot = Bot(
        token=backend_settings.normalized_telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = create_dispatcher()
    await bot.delete_webhook(drop_pending_updates=True)
    LOGGER.info("Telegram bot polling started")

    try:
        await dispatcher.start_polling(bot)
    finally:
        LOGGER.info("Telegram bot polling stopped")
        await bot.session.close()


def main() -> None:
    """Точка входа CLI-команды Telegram-бота."""
    configure_logging()
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()

