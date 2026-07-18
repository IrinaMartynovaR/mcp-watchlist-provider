import json
import logging
import sys
from datetime import UTC, datetime
from typing import Literal

from app.settings import BackendSettings


class JsonLogFormatter(logging.Formatter):
    """Сериализует стандартную LogRecord в компактный JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует одну запись лога.

        Args:
            record: Стандартная запись Python logging.

        Returns:
            JSON-строка с базовыми полями и исключением, если оно есть.
        """
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: BackendSettings, stream: Literal["stdout", "stderr"] = "stderr") -> None:
    """Настраивает централизованное логирование приложения.

    Args:
        settings: Backend-настройки logging.
        stream: Целевой поток. MCP обязан использовать ``stderr``, поскольку
            ``stdout`` зарезервирован под JSON-RPC.
    """
    level = getattr(logging, settings.normalized_log_level, logging.INFO)
    target = sys.stderr if stream == "stderr" else sys.stdout
    handler = logging.StreamHandler(target)
    if settings.normalized_log_format.lower() == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                settings.normalized_log_format,
                datefmt=settings.normalized_log_date_format,
            )
        )
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger(__name__).info("Logging configured", extra={"log_level": logging.getLevelName(level)})
