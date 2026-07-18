import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from app.config import RuntimeConfig
from domain.models import FeedItem
from mcp_tools import classification


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
def classifier_config(runtime_config: RuntimeConfig) -> RuntimeConfig:
    """Включает classifier в изолированном runtime config."""
    return replace(runtime_config, tools=runtime_config.tools.model_copy(update={"classifier_enabled": True}))


def test_classifier_disabled_keeps_items_and_skips_llm(runtime_config: RuntimeConfig) -> None:
    def poison_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("ask_llm_data must not be called when classifier is disabled")

    items = [make_item("Some news")]

    assert classification.classify_feed_items(items, runtime_config, poison_llm) == items


def test_classifier_overrides_kind_and_extracts_title_entity(
    classifier_config: RuntimeConfig,
) -> None:
    llm_calls: list[dict[str, Any]] = []

    def fake_llm(
        prompt: str,
        settings: Any,
        default_system: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1500,
        reasoning_effort: str | None = None,
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

    items = [make_item("Elden Ring is a triumph"), make_item("Big savings on SSDs")]

    classified = classification.classify_feed_items(items, classifier_config, fake_llm)

    assert [item.kind for item in classified] == ["review", "noise"]
    assert classified[0].title_entity == "Elden Ring"
    assert llm_calls[0]["model"] == classifier_config.tools.normalized_classifier_model
    assert "Elden Ring is a triumph" in llm_calls[0]["prompt"]


def test_classifier_uses_persistent_cache_on_second_call(
    classifier_config: RuntimeConfig,
) -> None:
    llm_calls: list[str] = []

    def fake_llm(
        prompt: str,
        settings: Any,
        default_system: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1500,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        llm_calls.append(prompt)
        return {"response": json.dumps([{"kind": "review", "title_entity": "Hades 2"}])}

    items = [make_item("Hades 2 verdict")]

    first = classification.classify_feed_items(items, classifier_config, fake_llm)
    second = classification.classify_feed_items(items, classifier_config, fake_llm)

    assert len(llm_calls) == 1
    assert first[0].kind == second[0].kind == "review"


def test_classifier_keeps_heuristic_kind_on_llm_failure(
    classifier_config: RuntimeConfig,
) -> None:
    def broken_llm(
        prompt: str,
        settings: Any,
        default_system: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1500,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("LLM is down")

    items = [make_item("Steam sale is live", kind="noise")]

    classified = classification.classify_feed_items(items, classifier_config, broken_llm)

    assert classified[0].kind == "noise"
    assert classified[0].title_entity == ""


def test_classifier_keeps_heuristic_kind_on_malformed_response(
    classifier_config: RuntimeConfig,
) -> None:
    def malformed_llm(prompt: str, settings: Any, default_system: str, **_: Any) -> dict[str, Any]:
        return {"response": "not json at all"}

    items = [make_item("Some headline")]

    assert classification.classify_feed_items(items, classifier_config, malformed_llm)[0].kind == "news"


def test_classifier_failed_batch_is_not_cached_and_retried(
    classifier_config: RuntimeConfig,
) -> None:
    attempts: list[str] = []

    def flaky_llm(
        prompt: str,
        settings: Any,
        default_system: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1500,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        attempts.append(prompt)
        if len(attempts) == 1:
            raise RuntimeError("temporary outage")
        return {"response": json.dumps([{"kind": "release", "title_entity": "Silksong"}])}

    items = [make_item("Silksong is out")]

    assert classification.classify_feed_items(items, classifier_config, flaky_llm)[0].kind == "news"
    assert classification.classify_feed_items(items, classifier_config, flaky_llm)[0].kind == "release"
    assert len(attempts) == 2


def test_parse_verdicts_strips_code_fences_and_defaults_medium() -> None:
    response = '```json\n[{"kind": "review", "title_entity": "Dune"}]\n```'

    verdicts = classification._parse_verdicts(response, expected_count=1)

    assert verdicts == [{"kind": "review", "medium": "other", "title_entity": "Dune"}]


def test_classifier_applies_medium_from_verdict(classifier_config: RuntimeConfig) -> None:
    def fake_llm(prompt: str, settings: Any, default_system: str, **_: Any) -> dict[str, Any]:
        return {"response": json.dumps([{"kind": "release", "medium": "anime", "title_entity": "Cocoon"}])}

    items = [make_item("Cocoon anime premiere")]

    classified = classification.classify_feed_items(items, classifier_config, fake_llm)

    assert classified[0].medium == "anime"
    assert classified[0].title_entity == "Cocoon"


def test_parse_verdicts_rejects_wrong_count_and_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Expected JSON array"):
        classification._parse_verdicts("[]", expected_count=1)
    with pytest.raises(ValueError, match="Invalid classification verdict"):
        classification._parse_verdicts('[{"kind": "advert"}]', expected_count=1)
