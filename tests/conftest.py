from pathlib import Path

import pytest

from app.config import RuntimeConfig
from app.settings import BackendSettings
from llm_core.settings import LLMSettings
from mcp_tools.settings import ToolSettings
from myshows_client.settings import MyShowsSettings
from rss_feeds.settings import RSSSettings


@pytest.fixture()
def runtime_config(tmp_path: Path) -> RuntimeConfig:
    """Создаёт изолированный config без сетевых feature flags.

    Args:
        tmp_path: Временная директория pytest.

    Returns:
        RuntimeConfig для одного теста.
    """
    return RuntimeConfig(
        backend=BackendSettings(WATCHQUEST_DATA_DIR=tmp_path),
        llm=LLMSettings(LLM_API_KEY=""),
        tools=ToolSettings(
            TOOLS_CLASSIFIER_ENABLED=False,
            TOOLS_RAG_ENABLED=False,
            TOOLS_HYDE_ENABLED=False,
            TOOLS_MEMORY_ENABLED=False,
        ),
        rss=RSSSettings(),
        myshows=MyShowsSettings(MYSHOWS_LOGIN="", MYSHOWS_PASSWORD=""),
    )
