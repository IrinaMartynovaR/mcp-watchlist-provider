from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MyShowsSettings(BaseSettings):
    """Хранит настройки клиента неофициального API MyShows.me."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    login: str = Field(default="", alias="MYSHOWS_LOGIN")
    password: str = Field(default="", alias="MYSHOWS_PASSWORD")
    timeout_seconds: float = Field(default=30.0, alias="MYSHOWS_TIMEOUT_SECONDS")


@lru_cache
def get_myshows_settings() -> MyShowsSettings:
    """Загружает и кеширует настройки MyShows-клиента.

    Returns:
        Актуальные настройки MyShows-клиента.
    """
    return MyShowsSettings()


myshows_settings = get_myshows_settings()
