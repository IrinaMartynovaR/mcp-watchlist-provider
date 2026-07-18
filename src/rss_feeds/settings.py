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
    rss_fetch_concurrency: int = Field(default=4, alias="RSS_FETCH_CONCURRENCY")
    rss_user_agent: str = Field(
        default="WatchQuest/0.1 RSS reader (+https://github.com/IrinaMartynovaR/mcp-watchlist-provider)",
        alias="RSS_USER_AGENT",
    )
    rss_bridge_url: str = Field(default="http://localhost:3001", alias="RSS_BRIDGE_URL")
    rss_summary_max_chars: int = Field(default=500, alias="RSS_SUMMARY_MAX_CHARS")
    rss_bridge_placeholder: str = Field(default="{rss_bridge}", alias="RSS_BRIDGE_PLACEHOLDER")

    @property
    def normalized_rss_user_agent(self) -> str:
        """Возвращает очищенный User-Agent для RSS-запросов."""
        return self.rss_user_agent.strip()

    @property
    def normalized_rss_bridge_url(self) -> str:
        """Возвращает базовый URL RSS-Bridge без завершающего slash."""
        return self.rss_bridge_url.strip().rstrip("/")
