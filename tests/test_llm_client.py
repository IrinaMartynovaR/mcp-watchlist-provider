import pytest

from llm_core.client import create_llm_client
from llm_core.providers.openrouter import OpenRouterClient
from llm_core.settings import LLMSettings


def _settings(provider: str) -> LLMSettings:
    return LLMSettings(
        LLM_PROVIDER=provider,
        LLM_API_KEY="test-key",
        LLM_BASE_URL="https://example.com",
        LLM_MODEL="test-model",
        LLM_EMBEDDING_MODEL="test-embedding-model",
    )


def test_create_llm_client_returns_openrouter_client() -> None:
    client = create_llm_client(_settings("openrouter"))

    assert isinstance(client, OpenRouterClient)
    assert client.settings.embedding_model == "test-embedding-model"


def test_create_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(_settings("unknown_provider"))


def test_default_provider_is_openrouter() -> None:
    assert LLMSettings.model_fields["provider"].default == "openrouter"
