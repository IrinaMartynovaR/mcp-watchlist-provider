from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    watchquest_data_dir: Path = Field(default=Path("./data"), alias="WATCHQUEST_DATA_DIR")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")

    @property
    def data_dir(self) -> Path:
        path = self.watchquest_data_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def profile_file(self) -> Path:
        return self.data_dir / "profile.json"

    @property
    def watchlist_file(self) -> Path:
        return self.data_dir / "watchlist.json"

    @property
    def sources_file(self) -> Path:
        return self.data_dir / "sources.json"

    @property
    def cache_file(self) -> Path:
        return self.data_dir / "cache.json"

    @property
    def normalized_telegram_bot_token(self) -> str:
        return self.telegram_bot_token.strip()

    @property
    def normalized_langfuse_public_key(self) -> str:
        return self.langfuse_public_key.strip()

    @property
    def normalized_langfuse_secret_key(self) -> str:
        return self.langfuse_secret_key.strip()

    @property
    def normalized_langfuse_host(self) -> str:
        return self.langfuse_host.strip().rstrip("/")


@lru_cache
def get_backend_settings() -> BackendSettings:
    return BackendSettings()


backend_settings = get_backend_settings()

DATA_DIR = backend_settings.data_dir
PROFILE_FILE = backend_settings.profile_file
WATCHLIST_FILE = backend_settings.watchlist_file
SOURCES_FILE = backend_settings.sources_file
CACHE_FILE = backend_settings.cache_file
TELEGRAM_BOT_TOKEN = backend_settings.normalized_telegram_bot_token
LANGFUSE_PUBLIC_KEY = backend_settings.normalized_langfuse_public_key
LANGFUSE_SECRET_KEY = backend_settings.normalized_langfuse_secret_key
LANGFUSE_HOST = backend_settings.normalized_langfuse_host
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_HOST)
