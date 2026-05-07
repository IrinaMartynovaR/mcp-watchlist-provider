from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import TypeVar
from typing import cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000").strip()

LANGFUSE_ENABLED = bool(
    LANGFUSE_PUBLIC_KEY
    and LANGFUSE_SECRET_KEY
    and LANGFUSE_HOST
)

try:
    from langfuse import Langfuse
    from langfuse import observe as langfuse_observe
except Exception:  # pragma: no cover
    Langfuse = None
    langfuse_observe = None

_langfuse_client: Any | None = None


def get_langfuse_client() -> Any | None:
    """
    Возвращает singleton-клиент Langfuse.

    Returns:
        Any | None: Инициализированный клиент Langfuse или None.
    """
    global _langfuse_client

    if not LANGFUSE_ENABLED:
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    if Langfuse is None:
        logger.warning("Langfuse SDK is unavailable")
        return None

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
    """
    Создает tracing decorator с graceful fallback.

    Args:
        name (str | None): Имя trace/span операции.

    Returns:
        Callable[[F], F]: Декоратор tracing-функции.
    """
    client = get_langfuse_client()

    if client is not None and langfuse_observe is not None:
        return cast(
            Callable[[F], F],
            langfuse_observe(name=name) if name else langfuse_observe(),
        )

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
