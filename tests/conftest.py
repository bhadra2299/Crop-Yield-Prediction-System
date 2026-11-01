"""Shared fixtures for the agri test suite.

What this protects against
--------------------------
Tests that silently depend on the developer's real model registry under
models/registry, or that write metrics and figures into the working tree.
Without isolation the suite passes on the machine where a model happened to be
trained last and fails everywhere else, and a test that claims "no model is
registered" proves nothing at all.

Every fixture here either supplies synthetic data or redirects the registry and
the report directories to a throwaway location. Importing this module must never
require a trained model to exist, and it must not reach the network.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from agri.common.config import DataSpec, get_settings
from agri.common.registry import ModelRegistry

#: The crop_recommendation feature schema. Repeated here so a unit test never has
#: to read configs/ or data/ to build a frame that satisfies the real validator.
CROP_FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_TARGET_COLUMN = "label"
CROP_CLASSES = ["rice", "maize", "banana"]


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Repository root, resolved exactly as the package resolves it."""
    from agri.common.paths import PROJECT_ROOT

    return PROJECT_ROOT


@pytest.fixture
def crop_frame() -> pd.DataFrame:
    """A small synthetic frame matching the crop_recommendation schema.

    Twelve rows over three classes, every value inside the physical ranges
    declared in agri.common.validation.COMMON_RANGES, so the real validator
    accepts it without any column being dropped or coerced.
    """
    return pd.DataFrame(
        {
            "N": [90, 85, 40, 30, 20, 10, 100, 95, 45, 35, 25, 15],
            "P": [42, 40, 20, 18, 60, 58, 45, 43, 22, 19, 62, 59],
            "K": [43, 41, 30, 28, 45, 44, 46, 42, 32, 29, 47, 45],
            "temperature": [21.0, 22.0, 24.0, 25.0, 26.0, 27.0, 21.5, 22.5, 24.5, 25.5, 26.5, 27.5],
            "humidity": [82.0, 80.0, 65.0, 63.0, 90.0, 88.0, 81.0, 79.0, 66.0, 64.0, 89.0, 87.0],
            "ph": [6.5, 6.4, 6.0, 6.1, 5.8, 5.9, 6.6, 6.3, 6.2, 6.0, 5.7, 5.9],
            "rainfall": [202.9, 200.0, 110.0, 105.0, 150.0, 148.0, 205.0, 201.0, 112.0, 107.0, 152.0, 149.0],
            CROP_TARGET_COLUMN: ["rice"] * 4 + ["maize"] * 4 + ["banana"] * 4,
        }
    )


@pytest.fixture
def crop_data_spec() -> DataSpec:
    """The DataSpec the synthetic frame above is built to satisfy."""
    return DataSpec(
        raw="data/raw/crop_recommendation/crop_recommendation.csv",
        target=CROP_TARGET_COLUMN,
        features=list(CROP_FEATURE_COLUMNS),
        required_columns=[*CROP_FEATURE_COLUMNS, CROP_TARGET_COLUMN],
        min_rows=1,
        classes=list(CROP_CLASSES),
    )


@pytest.fixture
def tmp_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModelRegistry]:
    """A registry rooted in a temporary directory, with the env var set.

    agri.common.paths reads AGRI_MODEL_REGISTRY once at import time into
    MODELS_DIR, and agri.common.registry imported that name, so patching the
    environment alone is not enough: both module attributes are redirected too.
    The settings singleton is cleared so anything that reads settings sees the
    temporary root as well.
    """
    import agri.common.paths as paths_module
    import agri.common.registry as registry_module

    root = tmp_path / "registry"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AGRI_MODEL_REGISTRY", str(root))
    monkeypatch.setattr(paths_module, "MODELS_DIR", root)
    monkeypatch.setattr(registry_module, "MODELS_DIR", root)
    get_settings.cache_clear()
    try:
        yield ModelRegistry(root)
    finally:
        get_settings.cache_clear()


@pytest.fixture
def isolated_output_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Redirect metrics reports and figures out of the working tree.

    agri.common.tabular writes reports/metrics and agri.common.training writes
    reports/figures, both resolved at import time. A training test that forgets
    this leaves generated artefacts in the repository, which then get committed
    by accident and read by a later run.
    """
    import agri.common.tabular as tabular_module
    import agri.common.training as training_module

    metrics = tmp_path / "reports" / "metrics"
    figures = tmp_path / "reports" / "figures"
    monkeypatch.setattr(tabular_module, "METRICS_DIR", metrics)
    monkeypatch.setattr(training_module, "FIGURES_DIR", figures)
    return {"metrics": metrics, "figures": figures}


@pytest.fixture(scope="session")
def trained_crop_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[types.SimpleNamespace]:
    """Train crop_recommendation once into a throwaway registry for the session.

    Training the real model on the real CSV takes tens of seconds, so the
    integration tests share one run instead of repeating it. The registry and
    the report directories are redirected for the lifetime of the fixture and
    restored afterwards; nothing is written into the working tree.
    """
    import agri.common.paths as paths_module
    import agri.common.registry as registry_module
    import agri.common.tabular as tabular_module
    import agri.common.training as training_module
    from agri.common.training import train_module

    root = tmp_path_factory.mktemp("registry")
    reports = tmp_path_factory.mktemp("reports")
    previous = {
        "paths_models": paths_module.MODELS_DIR,
        "registry_models": registry_module.MODELS_DIR,
        "tabular_metrics": tabular_module.METRICS_DIR,
        "training_figures": training_module.FIGURES_DIR,
    }
    paths_module.MODELS_DIR = root
    registry_module.MODELS_DIR = root
    tabular_module.METRICS_DIR = reports / "metrics"
    training_module.FIGURES_DIR = reports / "figures"
    try:
        result = train_module("crop_recommendation", tune=False, register=True)
        yield types.SimpleNamespace(
            result=result,
            registry_root=root,
            reports=reports,
            metrics_report=reports / "metrics" / f"crop_recommendation_v{result['version']}.json",
        )
    finally:
        paths_module.MODELS_DIR = previous["paths_models"]
        registry_module.MODELS_DIR = previous["registry_models"]
        tabular_module.METRICS_DIR = previous["tabular_metrics"]
        training_module.FIGURES_DIR = previous["training_figures"]
