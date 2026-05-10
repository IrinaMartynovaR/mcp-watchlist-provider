from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def read_json[T](path: Path, default: T) -> T:
    if not path.exists():
        write_json(path, default)
        return default

    with path.open("r", encoding="utf-8") as file:
        return cast(T, json.load(file))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)
