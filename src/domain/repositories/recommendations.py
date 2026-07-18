from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.models import RecommendationRecord
from domain.storage.json_store import read_json, update_json


@dataclass(frozen=True)
class RecommendationRepository:
    """Хранит историю сформированных рекомендаций."""

    path: Path

    def save(self, record: RecommendationRecord) -> RecommendationRecord:
        """Атомарно добавляет или заменяет рекомендацию по идентификатору.

        Args:
            record: Валидированная recommendation-запись.

        Returns:
            Сохранённая запись.
        """
        serialized = record.model_dump(mode="json")

        def apply(data: dict[str, Any]) -> dict[str, Any]:
            items = [item for item in data.get("items", []) if item.get("id") != record.id]
            items.append(serialized)
            data["items"] = items
            return data

        update_json(self.path, {"items": []}, apply)
        return record

    def get(self, recommendation_id: str) -> RecommendationRecord | None:
        """Возвращает рекомендацию по идентификатору.

        Args:
            recommendation_id: Идентификатор recommendation-записи.

        Returns:
            Найденная запись либо None.
        """
        data: dict[str, Any] = read_json(self.path, {"items": []})
        for item in data.get("items", []):
            if item.get("id") == recommendation_id:
                return RecommendationRecord.model_validate(item)
        return None
