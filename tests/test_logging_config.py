import json
import logging

import pytest

from app.logging_config import configure_logging
from app.settings import BackendSettings


def test_configure_logging_supports_json_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """JSON logging не загрязняет stdout stdio-протокола."""
    settings = BackendSettings(LOG_FORMAT="json", LOG_LEVEL="INFO")

    configure_logging(settings, stream="stderr")
    logging.getLogger("watchquest.test").info("hello")

    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.err.splitlines()]
    assert captured.out == ""
    assert records[-1]["message"] == "hello"
    assert records[-1]["logger"] == "watchquest.test"
