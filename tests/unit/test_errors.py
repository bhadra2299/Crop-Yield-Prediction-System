"""Error hierarchy tests.

What this protects against
--------------------------
An error reaching the API without a stable machine-readable code or an HTTP
status. Clients branch on the code instead of parsing prose, so a subclass that
forgets one silently becomes an unhandled 500 with no diagnosable contract. The
tests walk the hierarchy instead of listing classes, so a newly added error is
covered the moment it is defined.
"""

from __future__ import annotations

import pytest

from agri.common.errors import AgriError

pytestmark = pytest.mark.unit


def all_error_classes() -> list[type[AgriError]]:
    """Every AgriError subclass reachable from the base class."""
    found: set[type[AgriError]] = set()
    stack: list[type[AgriError]] = [AgriError]
    while stack:
        current = stack.pop()
        if current not in found:
            found.add(current)
            stack.extend(current.__subclasses__())
    return sorted(found, key=lambda cls: cls.__name__)


def test_the_hierarchy_covers_the_documented_errors() -> None:
    names = {cls.__name__ for cls in all_error_classes()}
    assert {
        "AgriError",
        "ConfigError",
        "DataValidationError",
        "ModelNotTrainedError",
        "ModuleExecutionError",
        "ExternalServiceError",
        "RegistryError",
    } <= names


@pytest.mark.parametrize("error_class", all_error_classes(), ids=lambda cls: cls.__name__)
def test_every_error_has_a_code_and_an_http_status(error_class: type[AgriError]) -> None:
    assert issubclass(error_class, AgriError)
    assert isinstance(error_class.code, str)
    assert error_class.code
    assert isinstance(error_class.http_status, int)
    assert 400 <= error_class.http_status <= 599


@pytest.mark.parametrize("error_class", all_error_classes(), ids=lambda cls: cls.__name__)
def test_every_error_serialises_through_to_dict(error_class: type[AgriError]) -> None:
    error = error_class("boom", context={"field": "ph"})
    payload = error.to_dict()
    assert payload["code"] == error_class.code
    assert payload["message"] == "boom"
    assert payload["context"] == {"field": "ph"}
    assert str(error) == "boom"


def test_codes_are_unique_across_the_hierarchy() -> None:
    codes = [cls.code for cls in all_error_classes()]
    assert len(codes) == len(set(codes))


def test_context_defaults_to_an_empty_dict() -> None:
    assert AgriError("boom").to_dict()["context"] == {}
