from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RSSSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rss_bridge_base_url: str = Field(default="http://localhost:3001", alias="RSS_BRIDGE_BASE_URL")
    rss_bridge_timeout_seconds: float = Field(default=20.0, alias="RSS_BRIDGE_TIMEOUT_SECONDS")
    rss_fetch_timeout_seconds: float = Field(default=20.0, alias="RSS_FETCH_TIMEOUT_SECONDS")
    rss_user_agent: str = Field(
        default="WatchQuest/0.1 RSS reader (+https://github.com/IrinaMartynovaR/mcp-watchlist-provider)",
        alias="RSS_USER_AGENT",
    )

    @property
    def normalized_rss_bridge_base_url(self) -> str:
        return self.rss_bridge_base_url.strip().rstrip("/")

    @property
    def normalized_rss_user_agent(self) -> str:
        return self.rss_user_agent.strip()


@lru_cache
def get_rss_settings() -> RSSSettings:
    return RSSSettings()


rss_settings = get_rss_settings()

RSS_BRIDGE_BASE_URL = rss_settings.normalized_rss_bridge_base_url
RSS_BRIDGE_TIMEOUT_SECONDS = rss_settings.rss_bridge_timeout_seconds
RSS_FETCH_TIMEOUT_SECONDS = rss_settings.rss_fetch_timeout_seconds
RSS_USER_AGENT = rss_settings.normalized_rss_user_agent
