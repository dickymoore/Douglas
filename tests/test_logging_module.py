import importlib
import json
import logging

import pytest

import douglas.logging as logging_module
from douglas.logging import get_logger, log_action, log_exceptions


@pytest.fixture(autouse=True)
def _reload_logging_module():
    importlib.reload(logging_module)
    yield
    importlib.reload(logging_module)


def test_configure_logging_writes_json(tmp_path, monkeypatch):
    monkeypatch.setenv("DOUGLAS_LOG_DIR", str(tmp_path))
    logging_module.configure_logging(level="info")
    logger = get_logger("tests.logging")
    logger.info("structured message", extra={"metadata": {"key": "value"}})
    for handler in logging.getLogger("douglas").handlers:
        handler.flush()
    logging.shutdown()
    log_file = tmp_path / "douglas.log"
    contents = log_file.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(contents[-1])
    assert payload["message"] == "structured message"
    assert payload["metadata"]["key"] == "value"
    assert payload["level"] == "INFO"


def test_log_action_decorator_logs_success(caplog):
    logging_module.configure_logging()
    logger = get_logger("tests.actions")
    base_logger = logging.getLogger("douglas")
    previous = base_logger.propagate
    base_logger.propagate = True
    caplog.set_level("INFO", logger=logger.name)

    @log_action("sample-action", logger_factory=lambda: logger)
    def _run():
        return "ok"

    result = _run()
    assert result == "ok"
    output = caplog.text
    assert "sample-action:start" in output
    assert "sample-action:success" in output
    base_logger.propagate = previous


def test_log_exceptions_records_errors(caplog):
    logging_module.configure_logging()
    logger = get_logger("tests.exceptions")
    base_logger = logging.getLogger("douglas")
    previous = base_logger.propagate
    base_logger.propagate = True
    caplog.set_level("ERROR", logger=logger.name)

    with pytest.raises(RuntimeError):
        with log_exceptions(logger):
            raise RuntimeError("boom")

    assert "Unhandled error" in caplog.text
    base_logger.propagate = previous
