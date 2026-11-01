"""Module configuration tests.

What this protects against
--------------------------
A config file that parses but means the wrong thing. Configuration is the single
source of truth for training and serving, so a typo in a task name, a gate that
accepts everything, or an unexpanded environment placeholder has to fail at load
time rather than at three in the morning in production. The tests here also pin
the module inventory, so deleting a config is a visible, deliberate act.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agri.common.config import (
    GateSpec,
    available_module_configs,
    get_settings,
    interpolate_env,
    load_module_config,
)
from agri.common.errors import ConfigError
from agri.common.schemas import TaskType

pytestmark = pytest.mark.unit

#: Built at runtime so these tests never embed a literal dollar-brace sequence.
DOLLAR = chr(36)


def test_load_module_config_reads_the_crop_recommendation_config() -> None:
    config = load_module_config("crop_recommendation")
    assert config.name == "crop_recommendation"
    assert config.task is TaskType.CLASSIFICATION
    assert config.data.target == "label"
    assert config.data.features == ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    assert config.gates.primary_metric == "macro_f1"
    assert config.source_path is not None and config.source_path.endswith("crop_recommendation.yaml")


def test_available_module_configs_returns_every_declared_module() -> None:
    names = available_module_configs()
    assert len(names) == 10
    assert "crop_recommendation" in names
    assert "irrigation" in names
    assert names == sorted(names)


def test_missing_config_file_raises_config_error_with_the_expected_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_module_config("does_not_exist", config_dir=tmp_path)
    assert "does_not_exist" in excinfo.value.message
    assert "expected" in excinfo.value.context


def test_unknown_task_in_a_config_raises_config_error(tmp_path: Path) -> None:
    payload = {
        "name": "bad_task",
        "version": "1.0.0",
        "task": "not_a_real_task",
        "data": {
            "raw": "data/raw/none.csv",
            "target": "label",
            "features": ["a"],
            "required_columns": ["a", "label"],
        },
        "model": {"family": "sklearn", "algorithm": "RandomForest"},
        "gates": {"primary_metric": "macro_f1", "minimum": 0.5},
    }
    (tmp_path / "bad_task.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_module_config("bad_task", config_dir=tmp_path)


def test_config_digest_is_stable_across_loads() -> None:
    first = load_module_config("crop_recommendation").config_digest
    second = load_module_config("crop_recommendation").config_digest
    assert first == second
    assert len(first) == 64


def test_env_interpolation_reads_a_set_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGRI_TEST_INTERP", "from-env")
    assert interpolate_env(DOLLAR + "{AGRI_TEST_INTERP}") == "from-env"


def test_env_interpolation_uses_the_documented_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGRI_TEST_INTERP_MISSING", raising=False)
    template = DOLLAR + "{AGRI_TEST_INTERP_MISSING:-fallback}"
    assert interpolate_env(template) == "fallback"

    monkeypatch.setenv("AGRI_TEST_INTERP_MISSING", "override")
    assert interpolate_env(template) == "override"


def test_unset_variable_without_a_default_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGRI_TEST_INTERP_REQUIRED", raising=False)
    with pytest.raises(ConfigError):
        interpolate_env(DOLLAR + "{AGRI_TEST_INTERP_REQUIRED}")


def test_gate_spec_enforces_a_minimum_floor() -> None:
    gate = GateSpec(primary_metric="macro_f1", minimum=0.9, must_beat_baseline=False)
    passed, message = gate.check(0.95)
    assert passed is True
    assert "satisfies the bound" in message

    failed, message = gate.check(0.85)
    assert failed is False
    assert "below the floor" in message


def test_gate_spec_enforces_a_maximum_ceiling() -> None:
    gate = GateSpec(primary_metric="rmse", maximum=1.0, must_beat_baseline=False)
    assert gate.check(0.5)[0] is True
    assert gate.check(1.5)[0] is False


def test_gate_spec_accepts_secondary_metric_floors() -> None:
    gate = GateSpec(
        primary_metric="macro_f1",
        minimum=0.9,
        secondary_minimums={"accuracy": 0.8},
        must_beat_baseline=False,
    )
    assert gate.secondary_minimums == {"accuracy": 0.8}


def test_gate_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GateSpec(primary_metric="macro_f1", minimum=0.5, not_a_real_field=1)


def test_settings_registry_path_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "custom-registry"
    monkeypatch.setenv("AGRI_MODEL_REGISTRY", str(expected))
    get_settings.cache_clear()
    try:
        assert get_settings().registry_path == expected
    finally:
        get_settings.cache_clear()
