from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Хранит настройки LLM-провайдера и нормализует env-ввод."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = Field(default="zai_glm", validation_alias=AliasChoices("LLM_PROVIDER", "ZAI_PROVIDER"))
    api_key: str = Field(default="", validation_alias=AliasChoices("LLM_API_KEY", "ZAI_API_KEY"))
    base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        validation_alias=AliasChoices("LLM_BASE_URL", "ZAI_BASE_URL"),
    )
    model: str = Field(default="glm-4.7-flash", validation_alias=AliasChoices("LLM_MODEL", "ZAI_MODEL"))
    timeout_seconds: float = Field(
        default=90.0,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "ZAI_TIMEOUT_SECONDS"),
    )
    temperature: float = Field(default=0.7, validation_alias=AliasChoices("LLM_TEMPERATURE", "ZAI_TEMPERATURE"))

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


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Загружает и кеширует настройки LLM-подсистемы.

    Returns:
        Актуальные настройки LLM-провайдера.
    """
    return LLMSettings()


llm_settings = get_llm_settings()

LLM_PROVIDER = llm_settings.normalized_provider
LLM_API_KEY = llm_settings.normalized_api_key
LLM_BASE_URL = llm_settings.normalized_base_url
LLM_MODEL = llm_settings.normalized_model
LLM_TIMEOUT_SECONDS = llm_settings.timeout_seconds
LLM_TEMPERATURE = llm_settings.temperature
