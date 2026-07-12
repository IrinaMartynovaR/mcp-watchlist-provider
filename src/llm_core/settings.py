from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Хранит настройки LLM-провайдера и нормализует env-ввод."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    api_key: str = Field(default="", alias="LLM_API_KEY")
    base_url: str = Field(default="https://openrouter.ai/api/v1", alias="LLM_BASE_URL")
    model: str = Field(default="openai/gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="openai/text-embedding-3-small", alias="LLM_EMBEDDING_MODEL")
    timeout_seconds: float = Field(default=90.0, alias="LLM_TIMEOUT_SECONDS")
    temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")

    @property
    def normalized_provider(self) -> str:
        """Возвращает нормализованное имя LLM-провайдера."""
        return self.provider.strip().lower()

    @property
    def normalized_api_key(self) -> str:
        """Возвращает API-ключ без внешних пробелов."""
        return self.api_key.strip()

    @property
    def normalized_base_url(self) -> str:
        """Возвращает базовый URL API без завершающего slash."""
        return self.base_url.strip().rstrip("/")

    @property
    def normalized_model(self) -> str:
        """Возвращает имя модели без внешних пробелов."""
        return self.model.strip()

    @property
    def normalized_embedding_model(self) -> str:
        """Возвращает имя embedding-модели без внешних пробелов."""
        return self.embedding_model.strip()


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Загружает и кеширует настройки LLM-подсистемы.

    Returns:
        Актуальные настройки LLM-провайдера.
    """
    return LLMSettings()


llm_settings = get_llm_settings()
