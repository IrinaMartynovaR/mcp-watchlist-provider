from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Настраивает централизованную конфигурацию логирования.

    Args:
        level (str): Уровень логирования.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
