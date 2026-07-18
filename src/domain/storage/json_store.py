import copy
import fcntl
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


@contextmanager
def _locked(path: Path, *, exclusive: bool) -> Iterator[None]:
    """Блокирует JSON-файл между потоками и процессами.

    Sidecar-файл используется вместо блокировки самого JSON: атомарный
    ``os.replace`` меняет inode JSON-файла, тогда как inode ``.lock`` остаётся
    стабильным для MCP-сервера и Telegram-бота.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_file.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_unlocked[T](path: Path, default: T) -> T:
    """Читает JSON внутри уже удерживаемой блокировки."""
    if not path.exists():
        value = copy.deepcopy(default)
        _write_unlocked(path, value)
        return value

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON file", extra={"path": str(path)})
        raise
    return cast(T, payload)


def _write_unlocked(path: Path, data: Any) -> None:
    """Атомарно заменяет JSON-файл внутри удерживаемой блокировки."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    with _locked(path, exclusive=False):
        if path.exists():
            payload = _read_unlocked(path, default)
            logger.debug("JSON file loaded", extra={"path": str(path)})
            return payload

    # Инициализация требует exclusive lock и повторной проверки: другой процесс
    # мог создать файл после освобождения shared lock.
    with _locked(path, exclusive=True):
        initialized = not path.exists()
        payload = _read_unlocked(path, default)
    if initialized:
        logger.info("JSON file missing; initialized default", extra={"path": str(path)})
    else:
        logger.debug("JSON file loaded", extra={"path": str(path)})
    return payload


def write_json(path: Path, data: Any) -> None:
    """Сериализует данные в JSON-файл.

    Args:
        path: Путь к JSON-файлу.
        data: Данные для записи.
    """
    with _locked(path, exclusive=True):
        _write_unlocked(path, data)
    logger.debug("JSON file written", extra={"path": str(path)})


def update_json[T](path: Path, default: T, update: Callable[[T], T]) -> T:
    """Атомарно выполняет read-modify-write под одной блокировкой.

    Args:
        path: Путь к JSON-файлу.
        default: Начальное значение при отсутствии файла.
        update: Чистая или мутирующая функция, возвращающая новое содержимое.

    Returns:
        Сохранённое содержимое после применения ``update``.

    Notes:
        Если ``update`` выбрасывает исключение, исходный файл не изменяется.
    """
    with _locked(path, exclusive=True):
        current = _read_unlocked(path, default)
        updated = update(current)
        _write_unlocked(path, updated)
    logger.debug("JSON file updated", extra={"path": str(path)})
    return updated
