from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from domain.models import Category


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
    session_url: str = Field(default="https://myshows.me/api/session", alias="MYSHOWS_SESSION_URL")
    rpc_url: str = Field(default="https://myshows.me/v3/rpc/", alias="MYSHOWS_RPC_URL")
    movies_page_size: int = Field(default=20, alias="MYSHOWS_MOVIES_PAGE_SIZE")
    max_movie_pages: int = Field(default=200, alias="MYSHOWS_MAX_MOVIE_PAGES")
    movie_category: Category = Field(default="movies_series", alias="MYSHOWS_MOVIE_CATEGORY")
    show_category: Category = Field(default="series", alias="MYSHOWS_SHOW_CATEGORY")
    import_source: str = Field(default="myshows_import", alias="MYSHOWS_IMPORT_SOURCE")
    import_weight_factor: float = Field(default=0.5, alias="MYSHOWS_IMPORT_WEIGHT_FACTOR")
    positive_show_statuses: str = Field(
        default="watching,finished,watched",
        alias="MYSHOWS_POSITIVE_SHOW_STATUSES",
    )
    negative_show_statuses: str = Field(default="cancelled,canceled", alias="MYSHOWS_NEGATIVE_SHOW_STATUSES")

    @property
    def normalized_positive_show_statuses(self) -> frozenset[str]:
        """Возвращает положительные статусы просмотра."""
        return frozenset(value.strip().lower() for value in self.positive_show_statuses.split(",") if value.strip())

    @property
    def normalized_negative_show_statuses(self) -> frozenset[str]:
        """Возвращает отрицательные статусы просмотра."""
        return frozenset(value.strip().lower() for value in self.negative_show_statuses.split(",") if value.strip())
