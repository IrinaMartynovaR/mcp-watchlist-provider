from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

try:
    # Langfuse SDK v4 style import.
    from langfuse import observe as _observe
except Exception:  # pragma: no cover - tracing is optional
    _observe = None


def observe(name: str | None = None) -> Callable[[F], F]:
    if _observe is not None:
        return cast(Callable[[F], F], _observe(name=name) if name else _observe())

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
