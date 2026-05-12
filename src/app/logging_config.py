import logging
import sys

from app.settings import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL


def configure_logging() -> None:
    """Настраивает централизованное логирование приложения."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger(__name__).info("Logging configured", extra={"log_level": logging.getLevelName(level)})

