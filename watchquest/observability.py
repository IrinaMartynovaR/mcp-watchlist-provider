from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

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

_langfuse_client: Any | None = None
_Langfuse: Any = None
_langfuse_observe: Any = None

try:
    from langfuse import Langfuse
    from langfuse import observe as imported_observe

    _Langfuse = Langfuse
    _langfuse_observe = imported_observe
except Exception:  # pragma: no cover
    pass


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

    if _Langfuse is None:
        logger.warning("Langfuse SDK is unavailable")
        return None

    try:
        _langfuse_client = _Langfuse(
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
    Create a tracing decorator with graceful fallback.

    Args:
        name (str | None): Имя trace/span операции.

    Returns:
        Callable[[F], F]: Декоратор tracing-функции.
    """
    client = get_langfuse_client()

    if client is not None and _langfuse_observe is not None:
        return cast(
            Callable[[F], F],
            _langfuse_observe(name=name) if name else _langfuse_observe(),
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
