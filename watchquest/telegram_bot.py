# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from watchquest.config import TELEGRAM_BOT_TOKEN
from watchquest.logging_config import configure_logging
from watchquest.models import Category
from watchquest.tools.recommendation import recommend_media_data

LOGGER = logging.getLogger(__name__)
MAX_MESSAGE_LENGTH = 3900

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
    normalized = text.lower()
    if any(word in normalized for word in ("игра", "игру", "игры", "game", "games")):
        return "games"
    if any(word in normalized for word in ("сериал", "сериалы", "series", "show")):
        return "series"
    if any(word in normalized for word in ("фильм", "кино", "movie", "film")):
        return "movies_series"
    return "all"


def split_telegram_text(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
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

    return [chunk for chunk in chunks if chunk]


def format_recommendation_response(result: dict[str, Any]) -> str:
    recommendation = str(result.get("recommendation") or "").strip()
    candidates = result.get("candidates", [])
    candidate_lines: list[str] = []

    if isinstance(candidates, list) and candidates:
        for item in candidates[:5]:
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
    query = query.strip()
    if not query:
        await message.answer("Напиши запрос после команды, например: /recommend посоветуй вайбовую игру")
        return

    await message.answer("Сейчас проверю RSS и соберу рекомендацию.")
    category = infer_category(query)

    try:
        result = await asyncio.to_thread(
            recommend_media_data,
            query=query,
            category=category,
            refresh=True,
            limit=5,
            limit_per_source=8,
        )
    except Exception:
        LOGGER.exception("Telegram recommendation failed")
        await message.answer("Не смогла собрать рекомендацию. Посмотри логи сервиса, там будет причина.")
        return

    for chunk in split_telegram_text(format_recommendation_response(result)):
        await message.answer(chunk, disable_web_page_preview=True)


def create_dispatcher() -> Dispatcher:
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
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dispatcher = create_dispatcher()
    await bot.delete_webhook(drop_pending_updates=True)

    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> None:
    configure_logging()
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
