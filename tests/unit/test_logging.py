"""Structured logging tests.

What this protects against
--------------------------
The regression where a log call such as extra={"module": "x"} raised KeyError from
deep inside the standard library: module, name, args and filename collide with
LogRecord attributes. The platform passes that context on nearly every call, so a
collision turns a harmless log line into a failed request. These tests assert both
that the call does not raise and that the JSON output still carries the field
under the name the caller intended.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from agri.common.logging import JsonFormatter, TextFormatter, get_logger

pytestmark = pytest.mark.unit


def capture(logger_name: str, formatter: logging.Formatter, extra: dict) -> str:
    """Log one record through a throwaway handler and return the rendered output."""
    logger = get_logger(logger_name)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    previous_propagate = logger.propagate
    logger.propagate = False
    try:
        logger.info("hello", extra=extra)
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous_propagate
    return stream.getvalue()


def test_reserved_log_record_keys_do_not_raise() -> None:
    output = capture(
        "tests.logging.reserved",
        JsonFormatter(),
        {"module": "x", "name": "y", "args": (1,), "filename": "f.py", "value": 3},
    )
    assert output.strip()


def test_json_formatter_keeps_the_diverted_module_key() -> None:
    output = capture("tests.logging.json", JsonFormatter(), {"module": "irrigation", "value": 3})
    payload = json.loads(output.strip().splitlines()[-1])
    assert payload["module"] == "irrigation"
    assert payload["value"] == 3
    assert payload["msg"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"].endswith("tests.logging.json")
    assert "request_id" in payload


def test_json_formatter_emits_one_valid_json_object_per_line() -> None:
    output = capture("tests.logging.multiline", JsonFormatter(), {"module": "x"})
    lines = [line for line in output.strip().splitlines() if line]
    assert lines
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_text_formatter_renders_the_diverted_context() -> None:
    output = capture("tests.logging.text", TextFormatter(), {"module": "x", "value": 3})
    assert "hello" in output
    assert "module=x" in output
