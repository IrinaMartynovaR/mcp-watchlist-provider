from dataclasses import dataclass, field

from app.settings import BackendSettings
from llm_core.prompts.config import PromptConfig
from llm_core.settings import LLMSettings
from mcp_tools.settings import ToolSettings
from myshows_client.settings import MyShowsSettings
from rss_feeds.settings import RSSSettings


@dataclass(frozen=True)
class RuntimeConfig:
    """Объединяет runtime-настройки всех подсистем приложения.

    Attributes:
        backend: Пути данных, Telegram, logging и observability.
        llm: Провайдер, модели и параметры LLM-запросов.
        tools: Лимиты и feature flags recommendation-инструментов.
        rss: Параметры загрузки RSS-источников.
        myshows: Доступ и таймаут клиента MyShows.
        prompts: Системные prompts LLM-сценариев.
    """

    backend: BackendSettings
    llm: LLMSettings
    tools: ToolSettings
    rss: RSSSettings
    myshows: MyShowsSettings
    prompts: PromptConfig = field(default_factory=PromptConfig)

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Загружает единый снимок настроек из окружения и ``.env``.

        Returns:
            Новый неизменяемый runtime config.
        """
        return cls(
            backend=BackendSettings(),
            llm=LLMSettings(),
            tools=ToolSettings(),
            rss=RSSSettings(),
            myshows=MyShowsSettings(),
            prompts=PromptConfig(),
        )
