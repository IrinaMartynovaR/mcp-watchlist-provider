from typing import Any

import httpx
import pytest

from mcp_tools.settings import tool_settings


@pytest.fixture(autouse=True)
def disable_llm_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Отключает LLM-классификатор по умолчанию во всех тестах.

    Настройки читаются из локального `.env`, где классификатор может быть
    включён — без этого refresh-тесты пытались бы ходить в сеть. Тесты
    классификатора включают его обратно собственной фикстурой.
    """
    monkeypatch.setattr(tool_settings, "classifier_enabled", False)


@pytest.fixture()
def poison_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Запрещает сетевые вызовы httpx: любой POST падает с AssertionError.

    Returns:
        Список зафиксированных попыток вызова — в тестах проверяется, что он пуст.
    """
    calls: list[str] = []

    def poisoned_post(*args: Any, **kwargs: Any) -> httpx.Response:
        calls.append("post")
        raise AssertionError("Unexpected network I/O")

    monkeypatch.setattr(httpx.Client, "post", poisoned_post)
    return calls
