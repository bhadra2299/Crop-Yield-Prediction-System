"""Module package import and contract tests.

What this protects against
--------------------------
A module package that stops importing, or that stops exposing the Predictor the
API and the training entry points look up by name. The check iterates the
directory rather than a hardcoded list, so a new module is covered the moment it
is added and a removed one cannot linger in a stale list.

A package whose predict module needs an optional third-party dependency that is
not installed is skipped with an explicit reason; it is never silently treated as
passing, and a failure inside agri itself is still a failure.
"""

from __future__ import annotations

import importlib

import pytest

from agri.common.base import BasePredictor
from agri.common.paths import PROJECT_ROOT
from agri.common.tabular import TabularPredictor

pytestmark = pytest.mark.unit

MODULES_DIR = PROJECT_ROOT / "src" / "agri" / "modules"

#: Every importable module package, discovered from the directory. Empty
#: placeholder directories have no __init__.py and are therefore not packages.
PACKAGE_NAMES = sorted(
    path.name for path in MODULES_DIR.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
)


def test_the_module_directory_contains_importable_packages() -> None:
    assert len(PACKAGE_NAMES) >= 7
    assert "crop_recommendation" in PACKAGE_NAMES


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_module_package_imports_cleanly(package_name: str) -> None:
    package = importlib.import_module(f"agri.modules.{package_name}")
    assert package.__name__ == f"agri.modules.{package_name}"


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_module_package_exposes_a_predictor(package_name: str) -> None:
    package = importlib.import_module(f"agri.modules.{package_name}")
    try:
        predictor = package.Predictor
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if not missing.startswith("agri"):
            pytest.skip(
                f"agri.modules.{package_name}.predict needs the optional dependency '{missing}', "
                "which is not installed"
            )
        raise
    assert isinstance(predictor, type), f"{package_name}.Predictor is not a class"
    assert issubclass(predictor, BasePredictor)
    if issubclass(predictor, TabularPredictor):
        assert predictor.request_model is not None
        assert predictor.response_model is not None
