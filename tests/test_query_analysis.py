import json
from dataclasses import replace
from typing import Any

from app.config import RuntimeConfig
from mcp_tools import query_analysis


def _analyzer_config(config: RuntimeConfig) -> RuntimeConfig:
    """Включает LLM-классификатор в изолированном runtime config."""
    return replace(config, tools=config.tools.model_copy(update={"classifier_enabled": True}))


def _unused_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise AssertionError("LLM must not be called")


def test_analyze_query_extracts_medium_via_llm(runtime_config: RuntimeConfig) -> None:
    def fake_llm(prompt: str, settings: Any, default_system: str, **kwargs: Any) -> dict[str, Any]:
        assert "аниме про лето" in prompt
        return {"response": json.dumps({"medium": "anime"})}

    config = _analyzer_config(runtime_config)

    assert query_analysis.analyze_query_data("аниме про лето", config, fake_llm) == {"medium": "anime"}


def test_analyze_query_caches_result_on_second_call(runtime_config: RuntimeConfig) -> None:
    calls: list[str] = []

    def fake_llm(prompt: str, settings: Any, default_system: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(prompt)
        return {"response": json.dumps({"medium": "game"})}

    config = _analyzer_config(runtime_config)

    first = query_analysis.analyze_query_data("мрачная игра", config, fake_llm)
    second = query_analysis.analyze_query_data("мрачная игра", config, fake_llm)

    assert first == second == {"medium": "game"}
    assert len(calls) == 1


def test_analyze_query_falls_back_to_heuristic_on_llm_failure(runtime_config: RuntimeConfig) -> None:
    def broken_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("LLM is down")

    config = _analyzer_config(runtime_config)

    assert query_analysis.analyze_query_data("хочу аниме", config, broken_llm) == {"medium": "anime"}
    assert query_analysis.analyze_query_data("посоветуй фильм", config, broken_llm) == {"medium": "movie"}


def test_analyze_query_uses_heuristic_when_classifier_disabled(runtime_config: RuntimeConfig) -> None:
    assert query_analysis.analyze_query_data("интересный сериал", runtime_config, _unused_llm) == {"medium": "series"}
    assert query_analysis.analyze_query_data("что-нибудь атмосферное", runtime_config, _unused_llm) == {"medium": None}


def test_analyze_query_rejects_unknown_medium(runtime_config: RuntimeConfig) -> None:
    def fake_llm(prompt: str, settings: Any, default_system: str, **kwargs: Any) -> dict[str, Any]:
        return {"response": json.dumps({"medium": "podcast"})}

    config = _analyzer_config(runtime_config)

    assert query_analysis.analyze_query_data("любой подкаст", config, fake_llm) == {"medium": None}


def test_item_matches_medium_prefers_classifier_field() -> None:
    assert query_analysis.item_matches_medium({"medium": "anime", "title": "x"}, "anime")
    assert not query_analysis.item_matches_medium({"medium": "game", "title": "anime-like"}, "anime")


def test_item_matches_medium_falls_back_to_text_heuristic() -> None:
    item = {"medium": "", "source": "Anime News Network", "title": "Cocoon trailer"}
    assert query_analysis.item_matches_medium(item, "anime")
    assert not query_analysis.item_matches_medium({"medium": "", "title": "Palace intrigue"}, "anime")
