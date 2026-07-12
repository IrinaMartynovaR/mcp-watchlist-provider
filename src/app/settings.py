from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Хранит backend-настройки, пути данных и observability-конфиг."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    watchquest_data_dir: Path = Field(default=Path("./data"), alias="WATCHQUEST_DATA_DIR")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_max_message_length: int = Field(default=3900, alias="TELEGRAM_MAX_MESSAGE_LENGTH")
    telegram_candidate_preview_limit: int = Field(default=5, alias="TELEGRAM_CANDIDATE_PREVIEW_LIMIT")
    telegram_recommendation_limit: int = Field(default=5, alias="TELEGRAM_RECOMMENDATION_LIMIT")
    telegram_recommendation_source_limit: int = Field(default=8, alias="TELEGRAM_RECOMMENDATION_SOURCE_LIMIT")
    telegram_feedback_callback_prefix: str = Field(default="feedback", alias="TELEGRAM_FEEDBACK_CALLBACK_PREFIX")
    telegram_watchlist_request_phrases: str = Field(
        default="watchlist,вотчлист,покажи список,покажи мой список,что в списке,мой список",
        alias="TELEGRAM_WATCHLIST_REQUEST_PHRASES",
    )
    telegram_feedback_like_label: str = Field(default="👍 Подходит", alias="TELEGRAM_FEEDBACK_LIKE_LABEL")
    telegram_feedback_dislike_label: str = Field(default="👎 Не то", alias="TELEGRAM_FEEDBACK_DISLIKE_LABEL")
    telegram_feedback_watchlist_label: str = Field(default="➕ В watchlist", alias="TELEGRAM_FEEDBACK_WATCHLIST_LABEL")
    telegram_feedback_block_similar_label: str = Field(
        default="🚫 Не предлагать похожее",
        alias="TELEGRAM_FEEDBACK_BLOCK_SIMILAR_LABEL",
    )
    telegram_feedback_like_response: str = Field(
        default="Запомнила: это тебе подходит.",
        alias="TELEGRAM_FEEDBACK_LIKE_RESPONSE",
    )
    telegram_feedback_dislike_response: str = Field(
        default="Запомнила: это не то.",
        alias="TELEGRAM_FEEDBACK_DISLIKE_RESPONSE",
    )
    telegram_feedback_watchlist_response: str = Field(
        default="Добавила в watchlist.",
        alias="TELEGRAM_FEEDBACK_WATCHLIST_RESPONSE",
    )
    telegram_feedback_block_similar_response: str = Field(
        default="Запомнила: похожее лучше не предлагать.",
        alias="TELEGRAM_FEEDBACK_BLOCK_SIMILAR_RESPONSE",
    )
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        alias="LOG_FORMAT",
    )
    log_date_format: str = Field(default="%Y-%m-%d %H:%M:%S", alias="LOG_DATE_FORMAT")

    @property
    def data_dir(self) -> Path:
        """Возвращает рабочую директорию данных и гарантирует её наличие."""
        path = self.watchquest_data_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def profile_file(self) -> Path:
        """Возвращает путь к JSON-файлу профиля пользователя."""
        return self.data_dir / "profile.json"

    @property
    def watchlist_file(self) -> Path:
        """Возвращает путь к JSON-файлу watchlist."""
        return self.data_dir / "watchlist.json"

    @property
    def sources_file(self) -> Path:
        """Возвращает путь к JSON-файлу RSS-источников."""
        return self.data_dir / "sources.json"

    @property
    def cache_file(self) -> Path:
        """Возвращает путь к JSON-файлу кеша RSS-записей."""
        return self.data_dir / "cache.json"

    @property
    def recommendations_file(self) -> Path:
        """Возвращает путь к JSON-файлу истории рекомендаций."""
        return self.data_dir / "recommendations.json"

    @property
    def feedback_file(self) -> Path:
        """Возвращает путь к JSON-файлу пользовательского feedback."""
        return self.data_dir / "feedback.json"

    @property
    def embeddings_cache_file(self) -> Path:
        """Возвращает путь к JSON-файлу кеша embedding-векторов."""
        return self.data_dir / "embeddings_cache.json"

    @property
    def classification_cache_file(self) -> Path:
        """Возвращает путь к JSON-файлу кеша LLM-классификации RSS-записей."""
        return self.data_dir / "classification_cache.json"

    @property
    def mem0_vector_store_dir(self) -> Path:
        """Возвращает директорию локального Chroma vector store для Mem0."""
        return self.data_dir / "mem0_chroma"

    @property
    def normalized_telegram_bot_token(self) -> str:
        """Возвращает очищенный Telegram bot token."""
        return self.telegram_bot_token.strip()

    @property
    def normalized_telegram_feedback_callback_prefix(self) -> str:
        """Возвращает безопасный префикс Telegram callback для feedback."""
        return self.telegram_feedback_callback_prefix.strip() or "feedback"

    @property
    def normalized_telegram_watchlist_request_phrases(self) -> tuple[str, ...]:
        """Разбирает фразы, по которым Telegram-бот показывает watchlist.

        Returns:
            Непустые фразы в нижнем регистре.
        """
        return tuple(
            phrase.strip().lower()
            for phrase in self.telegram_watchlist_request_phrases.split(",")
            if phrase.strip()
        )

    @property
    def normalized_langfuse_public_key(self) -> str:
        """Возвращает очищенный Langfuse public key."""
        return self.langfuse_public_key.strip()

    @property
    def normalized_langfuse_secret_key(self) -> str:
        """Возвращает очищенный Langfuse secret key."""
        return self.langfuse_secret_key.strip()

    @property
    def normalized_langfuse_host(self) -> str:
        """Возвращает нормализованный Langfuse host."""
        return self.langfuse_host.strip().rstrip("/")

    @property
    def normalized_log_level(self) -> str:
        """Возвращает нормализованный уровень логирования."""
        return self.log_level.strip().upper() or "INFO"

    @property
    def normalized_log_format(self) -> str:
        """Возвращает строку формата логов с безопасным fallback."""
        return self.log_format.strip() or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    @property
    def normalized_log_date_format(self) -> str:
        """Возвращает формат даты логов с безопасным fallback."""
        return self.log_date_format.strip() or "%Y-%m-%d %H:%M:%S"


@lru_cache
def get_backend_settings() -> BackendSettings:
    """Загружает и кеширует backend-настройки.

    Returns:
        Актуальные настройки backend-слоя.
    """
    return BackendSettings()


backend_settings = get_backend_settings()
