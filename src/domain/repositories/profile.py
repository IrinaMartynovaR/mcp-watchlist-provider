from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.models import Category
from domain.storage.json_store import read_json, update_json


@dataclass(frozen=True)
class ProfileRepository:
    """Хранит профиль и атомарно обновляет накопленные предпочтения."""

    path: Path

    def get(self) -> dict[str, Any]:
        """Возвращает актуальный профиль пользователя.

        Returns:
            Словарь профиля.
        """
        return read_json(self.path, {})

    def add_preferences(
        self,
        likes: list[str] | None = None,
        dislikes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Добавляет явные likes и dislikes без потери параллельных изменений.

        Args:
            likes: Новые предпочтения.
            dislikes: Новые антипредпочтения.

        Returns:
            Обновлённый профиль.
        """

        def apply(profile: dict[str, Any]) -> dict[str, Any]:
            if likes:
                profile["likes"] = sorted(set(profile.get("likes", []) + likes))
            if dislikes:
                profile["dislikes"] = sorted(set(profile.get("dislikes", []) + dislikes))
            return profile

        return update_json(self.path, {}, apply)

    def learn(
        self,
        *,
        category: Category,
        source: str | None,
        tags: list[str],
        weight: int,
    ) -> dict[str, Any]:
        """Атомарно применяет один feedback-сигнал к learned preferences.

        Args:
            category: Категория рекомендации.
            source: Источник кандидата.
            tags: Нормализованные теги кандидата.
            weight: Знаковый вес feedback.

        Returns:
            Обновлённый профиль.
        """

        def apply(profile: dict[str, Any]) -> dict[str, Any]:
            learned = profile.setdefault("learned_preferences", {})
            if not isinstance(learned, dict):
                learned = {}
                profile["learned_preferences"] = learned

            _bump_weight(learned, "categories", category, weight)
            if source:
                _bump_weight(learned, "sources", source, weight)
            for tag in tags:
                _bump_weight(learned, "tags", tag, weight)
            return profile

        return update_json(self.path, {}, apply)


def _bump_weight(learned: dict[str, Any], section: str, key: str, delta: int) -> None:
    """Изменяет числовой вес одной preference-секции."""
    values = learned.setdefault(section, {})
    if not isinstance(values, dict):
        values = {}
        learned[section] = values
    current = values.get(key, 0)
    values[key] = int(current if isinstance(current, int) else 0) + delta
