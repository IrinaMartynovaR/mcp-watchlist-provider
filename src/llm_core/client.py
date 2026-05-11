from typing import Protocol

from llm_core.providers.zai_glm import ZAIGLMClient, ZAIGLMSettings
from llm_core.schemas import ChatMessage
from llm_core.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)


class LLMClient(Protocol):
    def chat(self, messages: list[ChatMessage], max_tokens: int = 1500) -> str: ...


def create_llm_client() -> LLMClient:
    if LLM_PROVIDER == "zai_glm":
        return ZAIGLMClient(
            settings=ZAIGLMSettings(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                model=LLM_MODEL,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
                temperature=LLM_TEMPERATURE,
            )
        )
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
