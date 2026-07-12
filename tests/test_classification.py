import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.settings import BackendSettings
from domain.models import FeedItem
from mcp_tools import classification
from mcp_tools.settings import tool_settings


def make_item(title: str, kind: str = "news") -> FeedItem:
    return FeedItem(
        title=title,
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        source="Feed",
        category="games",
        kind=kind,  # type: ignore[arg-type]
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture()
def enable_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classification, "backend_settings", BackendSettings(WATCHQUEST_DATA_DIR=tmp_path))
    monkeypatch.setattr(tool_settings, "classifier_enabled", True)


def test_classifier_disabled_keeps_items_and_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def poison_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("ask_llm_data must not be called when classifier is disabled")

    monkeypatch.setattr(tool_settings, "classifier_enabled", False)
    monkeypatch.setattr(classification, "ask_llm_data", poison_llm)
    items = [make_item("Some news")]

    assert classification.classify_feed_items(items) == items


def test_classifier_overrides_kind_and_extracts_title_entity(
    enable_classifier: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_calls: list[dict[str, Any]] = []

    def fake_llm(
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        llm_calls.append({"prompt": prompt, "model": model})
        return {
            "response": json.dumps(
                [
                    {"kind": "review", "title_entity": "Elden Ring"},
                    {"kind": "noise", "title_entity": ""},
                ]
            )
        }

    monkeypatch.setattr(classification, "ask_llm_data", fake_llm)
    items = [make_item("Elden Ring is a triumph"), make_item("Big savings on SSDs")]

    classified = classification.classify_feed_items(items)

    assert [item.kind for item in classified] == ["review", "noise"]
    assert classified[0].title_entity == "Elden Ring"
    assert llm_calls[0]["model"] == tool_settings.normalized_classifier_model
    assert "Elden Ring is a triumph" in llm_calls[0]["prompt"]


def test_classifier_uses_persistent_cache_on_second_call(
    enable_classifier: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_calls: list[str] = []

    def fake_llm(
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        llm_calls.append(prompt)
        return {"response": json.dumps([{"kind": "review", "title_entity": "Hades 2"}])}

    monkeypatch.setattr(classification, "ask_llm_data", fake_llm)
    items = [make_item("Hades 2 verdict")]

    first = classification.classify_feed_items(items)
    second = classification.classify_feed_items(items)

    assert len(llm_calls) == 1
    assert first[0].kind == second[0].kind == "review"


def test_classifier_keeps_heuristic_kind_on_llm_failure(
    enable_classifier: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_llm(
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        raise RuntimeError("LLM is down")

    monkeypatch.setattr(classification, "ask_llm_data", broken_llm)
    items = [make_item("Steam sale is live", kind="noise")]

    classified = classification.classify_feed_items(items)

    assert classified[0].kind == "noise"
    assert classified[0].title_entity == ""


def test_classifier_keeps_heuristic_kind_on_malformed_response(
    enable_classifier: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        classification,
        "ask_llm_data",
        lambda prompt, system=None, model=None, max_tokens=1500: {"response": "not json at all"},
    )
    items = [make_item("Some headline")]

    assert classification.classify_feed_items(items)[0].kind == "news"


def test_classifier_failed_batch_is_not_cached_and_retried(
    enable_classifier: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    def flaky_llm(
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        attempts.append(prompt)
        if len(attempts) == 1:
            raise RuntimeError("temporary outage")
        return {"response": json.dumps([{"kind": "release", "title_entity": "Silksong"}])}

    monkeypatch.setattr(classification, "ask_llm_data", flaky_llm)
    items = [make_item("Silksong is out")]

    assert classification.classify_feed_items(items)[0].kind == "news"
    assert classification.classify_feed_items(items)[0].kind == "release"
    assert len(attempts) == 2


def test_parse_verdicts_strips_code_fences() -> None:
    response = '```json\n[{"kind": "review", "title_entity": "Dune"}]\n```'

    verdicts = classification._parse_verdicts(response, expected_count=1)

    assert verdicts == [{"kind": "review", "title_entity": "Dune"}]


def test_parse_verdicts_rejects_wrong_count_and_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Expected JSON array"):
        classification._parse_verdicts("[]", expected_count=1)
    with pytest.raises(ValueError, match="Invalid classification verdict"):
        classification._parse_verdicts('[{"kind": "advert"}]', expected_count=1)
