import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from domain.storage.json_store import read_json, update_json, write_json


def _increment_many(path: Path, count: int) -> None:
    """Увеличивает JSON-счётчик из отдельного процесса."""

    def increment(data: dict[str, int]) -> dict[str, int]:
        data["value"] += 1
        return data

    for _ in range(count):
        update_json(path, {"value": 0}, increment)


def test_write_json_is_atomic_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"

    write_json(path, {"value": "готово"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "готово"}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_failed_update_keeps_previous_content(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_json(path, {"value": 1})

    def fail(data: dict[str, int]) -> dict[str, int]:
        data["value"] = 2
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        update_json(path, {"value": 0}, fail)

    assert read_json(path, {}) == {"value": 1}


def test_concurrent_thread_updates_do_not_lose_changes(tmp_path: Path) -> None:
    path = tmp_path / "counter.json"

    def increment(_: int) -> None:
        def apply(data: dict[str, int]) -> dict[str, int]:
            data["value"] += 1
            return data

        update_json(path, {"value": 0}, apply)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(increment, range(100)))

    assert read_json(path, {}) == {"value": 100}


def test_concurrent_process_updates_do_not_lose_changes(tmp_path: Path) -> None:
    path = tmp_path / "counter.json"
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_increment_many, args=(path, 20)) for _ in range(4)]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    assert read_json(path, {}) == {"value": 80}


def test_update_json_can_atomically_append_records(tmp_path: Path) -> None:
    path = tmp_path / "events.json"

    def append(index: int) -> None:
        def apply(data: dict[str, Any]) -> dict[str, Any]:
            data["items"].append(index)
            return data

        update_json(path, {"items": []}, apply)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(50)))

    stored: dict[str, Any] = read_json(path, {"items": []})
    assert sorted(stored["items"]) == list(range(50))
