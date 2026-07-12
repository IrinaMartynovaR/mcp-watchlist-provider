from typing import Any

import httpx
import pytest


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
