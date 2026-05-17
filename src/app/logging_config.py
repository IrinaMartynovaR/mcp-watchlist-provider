import logging
import sys

from app.settings import backend_settings


def configure_logging() -> None:
    """Настраивает централизованное логирование приложения."""
    level = getattr(logging, backend_settings.normalized_log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format=backend_settings.normalized_log_format,
        datefmt=backend_settings.normalized_log_date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger(__name__).info("Logging configured", extra={"log_level": logging.getLevelName(level)})

