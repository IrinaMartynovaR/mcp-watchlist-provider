from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.models import RecommendationFeedback
from domain.storage.json_store import update_json


@dataclass(frozen=True)
class FeedbackRepository:
    """Хранит неизменяемую историю пользовательского feedback."""

    path: Path

    def append(self, feedback: RecommendationFeedback) -> None:
        """Атомарно добавляет feedback-событие в конец истории.

        Args:
            feedback: Валидированное feedback-событие.
        """
        serialized = feedback.model_dump(mode="json")

        def apply(data: dict[str, Any]) -> dict[str, Any]:
            data.setdefault("items", []).append(serialized)
            return data

        update_json(self.path, {"items": []}, apply)
