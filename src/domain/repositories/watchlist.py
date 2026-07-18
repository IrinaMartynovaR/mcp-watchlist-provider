from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.models import WatchlistItem
from domain.storage.json_store import read_json, update_json


@dataclass(frozen=True)
class WatchlistRepository:
    """Управляет watchlist поверх атомарного JSON-хранилища."""

    path: Path

    def add(self, item: WatchlistItem) -> dict[str, Any]:
        """Атомарно добавляет элемент.

        Args:
            item: Валидированный watchlist item.

        Returns:
            Сериализованная форма элемента.
        """
        serialized = item.model_dump(mode="json")

        def apply(data: dict[str, Any]) -> dict[str, Any]:
            data.setdefault("items", []).append(serialized)
            return data

        update_json(self.path, {"items": []}, apply)
        return serialized

    def list(self, *, media_type: str = "all", status: str = "planned") -> list[dict[str, Any]]:
        """Возвращает элементы с фильтрацией по типу и статусу.

        Args:
            media_type: Тип медиа или ``all``.
            status: Статус элемента или ``all``.

        Returns:
            Отфильтрованные watchlist items.
        """
        data: dict[str, Any] = read_json(self.path, {"items": []})
        items = data.get("items", [])
        if media_type != "all":
            items = [item for item in items if item.get("type") == media_type]
        if status != "all":
            items = [item for item in items if item.get("status") == status]
        return [dict(item) for item in items]

    def rate(self, *, title: str, rating: int, comment: str = "") -> dict[str, Any]:
        """Атомарно обновляет оценку элемента, найденного по названию.

        Args:
            title: Название элемента без учёта регистра.
            rating: Пользовательская оценка.
            comment: Пользовательский комментарий.

        Returns:
            Обновлённый элемент.

        Raises:
            ValueError: Если элемент не найден.
        """
        rated: dict[str, Any] | None = None

        def apply(data: dict[str, Any]) -> dict[str, Any]:
            nonlocal rated
            for item in data.get("items", []):
                if item.get("title", "").lower() == title.lower():
                    item["rating"] = rating
                    item["comment"] = comment
                    rated = dict(item)
                    return data
            raise ValueError(f"Item not found in watchlist: {title}")

        update_json(self.path, {"items": []}, apply)
        if rated is None:
            raise ValueError(f"Item not found in watchlist: {title}")
        return rated
