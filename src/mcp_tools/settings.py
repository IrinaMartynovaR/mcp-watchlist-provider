from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ToolSettings(BaseSettings):
    """Хранит лимиты и эвристики для MCP-инструментов."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    source_validation_limit: int = Field(default=3, alias="TOOLS_SOURCE_VALIDATION_LIMIT")
    feed_refresh_limit: int = Field(default=20, alias="TOOLS_FEED_REFRESH_LIMIT")
    cache_search_days: int = Field(default=30, alias="TOOLS_CACHE_SEARCH_DAYS")
    cache_search_limit: int = Field(default=20, alias="TOOLS_CACHE_SEARCH_LIMIT")

    recommendation_limit: int = Field(default=8, alias="TOOLS_RECOMMENDATION_LIMIT")
    recommendation_source_limit: int = Field(default=10, alias="TOOLS_RECOMMENDATION_SOURCE_LIMIT")
    recommendation_fallback_candidate_multiplier: int = Field(
        default=3,
        alias="TOOLS_RECOMMENDATION_FALLBACK_MULTIPLIER",
    )
    recommendation_keyword_marker: str = Field(default="вайб", alias="TOOLS_RECOMMENDATION_KEYWORD_MARKER")
    recommendation_keyword_variants: str = Field(
        default="уютная,атмосферная,cozy,vibe",
        alias="TOOLS_RECOMMENDATION_KEYWORD_VARIANTS",
    )

    @property
    def normalized_keyword_marker(self) -> str:
        """Возвращает нормализованный маркер для вайбовых запросов."""
        return self.recommendation_keyword_marker.strip().lower()

    @property
    def normalized_keyword_variants(self) -> list[str]:
        """Разбирает список дополнительных поисковых вариантов.

        Returns:
            Непустые строковые варианты без внешних пробелов.
        """
        return [item.strip() for item in self.recommendation_keyword_variants.split(",") if item.strip()]


@lru_cache
def get_tool_settings() -> ToolSettings:
    """Загружает и кеширует настройки MCP-инструментов.

    Returns:
        Актуальные tool-настройки.
    """
    return ToolSettings()


tool_settings = get_tool_settings()

SOURCE_VALIDATION_LIMIT = tool_settings.source_validation_limit
FEED_REFRESH_LIMIT = tool_settings.feed_refresh_limit
CACHE_SEARCH_DAYS = tool_settings.cache_search_days
CACHE_SEARCH_LIMIT = tool_settings.cache_search_limit
RECOMMENDATION_LIMIT = tool_settings.recommendation_limit
RECOMMENDATION_SOURCE_LIMIT = tool_settings.recommendation_source_limit
RECOMMENDATION_FALLBACK_CANDIDATE_MULTIPLIER = tool_settings.recommendation_fallback_candidate_multiplier
RECOMMENDATION_KEYWORD_MARKER = tool_settings.normalized_keyword_marker
RECOMMENDATION_KEYWORD_VARIANTS = tool_settings.normalized_keyword_variants
