import pytest

from llm_core import client as llm_client_module
from llm_core.providers.openrouter import OpenRouterClient
from llm_core.settings import LLMSettings, llm_settings


def _settings(provider: str) -> LLMSettings:
    return llm_settings.model_copy(
        update={
            "provider": provider,
            "api_key": "test-key",
            "base_url": "https://example.com",
            "model": "test-model",
            "embedding_model": "test-embedding-model",
        }
    )


def test_create_llm_client_returns_openrouter_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client_module, "llm_settings", _settings("openrouter"))

    client = llm_client_module.create_llm_client()

    assert isinstance(client, OpenRouterClient)
    assert client.settings.embedding_model == "test-embedding-model"


def test_create_llm_client_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client_module, "llm_settings", _settings("unknown_provider"))

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        llm_client_module.create_llm_client()


def test_default_provider_is_openrouter() -> None:
    assert LLMSettings.model_fields["provider"].default == "openrouter"
