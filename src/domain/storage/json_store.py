import json
import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


def read_json[T](path: Path, default: T) -> T:
    """Читает JSON-файл или инициализирует его значением по умолчанию.

    Args:
        path: Путь к JSON-файлу.
        default: Значение, которое нужно записать при отсутствии файла.

    Returns:
        Десериализованное содержимое файла.

    Raises:
        json.JSONDecodeError: Если файл содержит некорректный JSON.
    """
    if not path.exists():
        logger.info("JSON file missing; initializing default", extra={"path": str(path)})
        write_json(path, default)
        return default

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON file", extra={"path": str(path)})
        raise

    logger.debug("JSON file loaded", extra={"path": str(path)})
    return cast(T, payload)


def write_json(path: Path, data: Any) -> None:
    """Сериализует данные в JSON-файл.

    Args:
        path: Путь к JSON-файлу.
        data: Данные для записи.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    path.write_text(payload, encoding="utf-8")
    logger.debug("JSON file written", extra={"path": str(path)})

