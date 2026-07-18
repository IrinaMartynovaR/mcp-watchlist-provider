from functools import partial
from typing import Any

from app.config import RuntimeConfig
from llm_core.client import create_llm_client
from llm_core.embeddings import create_embeddings
from mcp_tools.feedback import save_recommendation_result
from mcp_tools.llm import ask_llm_data
from mcp_tools.media import (
    fetch_latest_items_data,
    get_profile_data,
    list_watchlist_data,
    search_cached_items_data,
)
from mcp_tools.memory import MemoryService
from mcp_tools.myshows_import import import_myshows_history_data
from mcp_tools.query_analysis import analyze_query_data
from mcp_tools.rag import semantic_search_data
from mcp_tools.recommendation import RecommendationDependencies, RecommendationService
from myshows_client.client import MyShowsClient


class WatchQuestApplication:
    """Композиционный корень runtime-зависимостей WatchQuest.

    Args:
        config: Единый снимок настроек приложения.
        memory_client: Необязательный Mem0-совместимый клиент для тестов.

    Attributes:
        config: Runtime-настройки приложения.
        memory: Сервис графовой памяти.
        recommendation: Полностью собранный recommendation service.
    """

    def __init__(self, config: RuntimeConfig, memory_client: Any | None = None) -> None:
        """Инициализирует все runtime-сервисы приложения.

        Args:
            config: Единый снимок настроек приложения.
            memory_client: Необязательный Mem0-совместимый клиент.
        """
        self.config = config
        self.memory = MemoryService(config, client=memory_client)
        self.myshows_client = MyShowsClient(config.myshows)

        llm_client = create_llm_client(config.llm)
        embeddings = create_embeddings(
            llm_client,
            config.backend.embeddings_cache_file,
            f"{config.llm.normalized_provider}::{config.llm.normalized_embedding_model}",
        )
        ask_llm = partial(
            ask_llm_data,
            settings=config.llm,
            default_system=config.prompts.recommendation_system,
        )
        search_cached = partial(search_cached_items_data, config)
        semantic_search = partial(
            semantic_search_data,
            config=config,
            search_cached=search_cached,
            embeddings=embeddings,
            ask_llm=ask_llm,
        )
        dependencies = RecommendationDependencies(
            fetch_latest=partial(fetch_latest_items_data, config),
            search_cached=search_cached,
            get_profile=partial(get_profile_data, config),
            list_watchlist=partial(list_watchlist_data, config),
            analyze_query=partial(analyze_query_data, config=config, ask_llm=ask_llm),
            semantic_search=semantic_search,
            semantic_memory_scores=self.memory.semantic_scores,
            ask_llm=ask_llm,
            save_result=partial(save_recommendation_result, config),
        )
        self.recommendation = RecommendationService(config, dependencies)

    def import_myshows_history(self, dry_run: bool = True) -> dict[str, Any]:
        """Импортирует историю MyShows через зависимости приложения.

        Args:
            dry_run: Если True, только рассчитывает сводку.

        Returns:
            Сводка обработанных и записанных элементов.
        """
        return import_myshows_history_data(
            self.myshows_client,
            self.memory.record_import,
            self.config.myshows,
            dry_run=dry_run,
        )
