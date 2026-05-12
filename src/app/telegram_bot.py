import asyncio
import logging
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from app.logging_config import configure_logging
from app.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CANDIDATE_PREVIEW_LIMIT,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TELEGRAM_RECOMMENDATION_LIMIT,
    TELEGRAM_RECOMMENDATION_SOURCE_LIMIT,
)
from domain.models import Category
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


def split_telegram_text(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
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
        for item in candidates[:TELEGRAM_CANDIDATE_PREVIEW_LIMIT]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Без названия").strip()
            source = str(item.get("source") or "unknown").strip()
            url = str(item.get("url") or "").strip()
            candidate_lines.append(f"- {title} ({source})\n  {url}".rstrip())

    parts = [recommendation or "Не смогла собрать рекомендацию по текущим источникам."]
    if candidate_lines:
        parts.append("Кандидаты из RSS:\n" + "\n".join(candidate_lines))

    return "\n\n".join(parts)


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
            limit=TELEGRAM_RECOMMENDATION_LIMIT,
            limit_per_source=TELEGRAM_RECOMMENDATION_SOURCE_LIMIT,
        )
    except Exception:
        LOGGER.exception("Telegram recommendation failed")
        await message.answer("Не смогла собрать рекомендацию. Посмотри логи сервиса, там будет причина.")
        return

    for chunk in split_telegram_text(format_recommendation_response(result)):
        await message.answer(chunk, disable_web_page_preview=True)
    LOGGER.info(
        "Telegram recommendation delivered",
        extra={"category": category, "candidate_count": result.get("candidate_count", 0)},
    )


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

    @dp.message(F.text)
    async def text_handler(message: Message) -> None:
        await answer_recommendation(message, message.text or "")

    return dp


async def run_bot() -> None:
    """Запускает Telegram polling lifecycle.

    Raises:
        RuntimeError: Если токен Telegram-бота отсутствует.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
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

