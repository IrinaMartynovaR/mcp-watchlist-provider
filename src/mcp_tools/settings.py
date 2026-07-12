from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from domain.models import FeedbackAction


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
    classifier_enabled: bool = Field(default=False, alias="TOOLS_CLASSIFIER_ENABLED")
    classifier_model: str = Field(default="google/gemma-4-26b-a4b-it", alias="TOOLS_CLASSIFIER_MODEL")
    classifier_batch_size: int = Field(default=25, alias="TOOLS_CLASSIFIER_BATCH_SIZE")
    rag_enabled: bool = Field(default=False, alias="TOOLS_RAG_ENABLED")
    hyde_enabled: bool = Field(default=False, alias="TOOLS_HYDE_ENABLED")
    memory_enabled: bool = Field(default=False, alias="TOOLS_MEMORY_ENABLED")
    memgraph_url: str = Field(default="bolt://localhost:7687", alias="MEMGRAPH_URL")
    memgraph_username: str = Field(default="memgraph", alias="MEMGRAPH_USERNAME")
    memgraph_password: str = Field(default="", alias="MEMGRAPH_PASSWORD")
    mem0_user_id: str = Field(default="watchquest-user", alias="MEM0_USER_ID")
    memory_top_k: int = Field(default=5, alias="TOOLS_MEMORY_TOP_K")
    memory_embedding_dims: int = Field(default=1536, alias="TOOLS_MEMORY_EMBEDDING_DIMS")
    feedback_like_weight: int = Field(default=1, alias="TOOLS_FEEDBACK_LIKE_WEIGHT")
    feedback_dislike_weight: int = Field(default=-1, alias="TOOLS_FEEDBACK_DISLIKE_WEIGHT")
    feedback_watchlist_weight: int = Field(default=2, alias="TOOLS_FEEDBACK_WATCHLIST_WEIGHT")
    feedback_block_similar_weight: int = Field(default=-3, alias="TOOLS_FEEDBACK_BLOCK_SIMILAR_WEIGHT")

    @property
    def normalized_classifier_model(self) -> str:
        """Возвращает очищенное имя модели LLM-классификатора."""
        return self.classifier_model.strip()

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

    @property
    def feedback_weights(self) -> dict[FeedbackAction, int]:
        """Возвращает веса пользовательского feedback для обучения профиля.

        Returns:
            Словарь весов по поддерживаемым feedback-действиям.
        """
        return {
            "like": self.feedback_like_weight,
            "dislike": self.feedback_dislike_weight,
            "watchlist": self.feedback_watchlist_weight,
            "block_similar": self.feedback_block_similar_weight,
        }


@lru_cache(1)
def get_tool_settings() -> ToolSettings:
    """Загружает и кеширует настройки MCP-инструментов.

    Returns:
        Актуальные tool-настройки.
    """
    return ToolSettings()


tool_settings = get_tool_settings()
