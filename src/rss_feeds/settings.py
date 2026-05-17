from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RSSSettings(BaseSettings):
    """Хранит настройки RSS-клиента."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rss_fetch_timeout_seconds: float = Field(default=20.0, alias="RSS_FETCH_TIMEOUT_SECONDS")
    rss_fetch_item_limit: int = Field(default=20, alias="RSS_FETCH_ITEM_LIMIT")
    rss_user_agent: str = Field(
        default="WatchQuest/0.1 RSS reader (+https://github.com/IrinaMartynovaR/mcp-watchlist-provider)",
        alias="RSS_USER_AGENT",
    )

    @property
    def normalized_rss_user_agent(self) -> str:
        """Возвращает очищенный User-Agent для RSS-запросов."""
        return self.rss_user_agent.strip()


@lru_cache
def get_rss_settings() -> RSSSettings:
    """Загружает и кеширует RSS-настройки.

    Returns:
        Актуальные настройки RSS-подсистемы.
    """
    return RSSSettings()


rss_settings = get_rss_settings()
