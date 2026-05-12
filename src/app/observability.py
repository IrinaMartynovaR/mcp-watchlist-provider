import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from langfuse import Langfuse
from langfuse import observe as langfuse_observe

from app.settings import LANGFUSE_ENABLED, LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_langfuse_client: Any | None = None


def get_langfuse_client() -> Any | None:
    """Возвращает singleton Langfuse-клиента для трассировки.

    Returns:
        Инициализированный Langfuse-клиент или `None`, если трассировка
        отключена конфигурацией либо клиент не удалось создать.
    """
    global _langfuse_client

    if not LANGFUSE_ENABLED:
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        logger.info("Langfuse client initialized")
    except Exception:
        logger.exception("Failed to initialize Langfuse client")
        return None

    return _langfuse_client


def observe(name: str | None = None) -> Callable[[F], F]:
    """Создаёт декоратор трассировки с безопасным fallback.

    Args:
        name: Необязательное имя trace/span.

    Returns:
        Декоратор, который включает Langfuse-observe при доступном клиенте
        или прозрачно вызывает исходную функцию без трассировки.
    """
    client = get_langfuse_client()

    if client is not None:
        return langfuse_observe(name=name) if name else langfuse_observe()

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug(
                "Tracing disabled for function",
                extra={
                    "function_name": func.__name__,
                    "trace_name": name,
                },
            )
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator

