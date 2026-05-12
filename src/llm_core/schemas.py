from typing import Literal, TypedDict

ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    """Описывает одно сообщение диалога для LLM."""
    role: ChatRole
    content: str
