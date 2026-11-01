"""End-to-end training smoke test.

What this protects against
--------------------------
A training pipeline that runs but never produces the artefacts CI and the API
depend on: the metrics report, the registry manifest and a passing metric gate.
This is the one test that exercises the real CSV, the real split, the real
estimator and the real gate together, so a change that silently stops registering
a model or stops enforcing the floor fails here rather than in production.

It is marked slow because it trains the production RandomForest; the run is
shared with the other integration tests through the trained_crop_run fixture.
"""

from __future__ import annotations

import json

import pytest

from agri.common.registry import ModelRegistry

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_training_registers_a_gate_passing_model_and_writes_the_report(trained_crop_run) -> None:
    result = trained_crop_run.result
    assert result["module"] == "crop_recommendation"

    gate = result["gate"]
    assert gate["passed"] is True
    assert gate["failed_checks"] == []
    assert gate["primary_metric"] == "macro_f1"
    assert gate["primary_value"] == pytest.approx(result["metrics"]["macro_f1"])


def test_training_metrics_report_is_readable_and_complete(trained_crop_run) -> None:
    report_path = trained_crop_run.metrics_report
    assert report_path.is_file(), f"expected a metrics report at {report_path}"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["module"] == "crop_recommendation"
    assert payload["gate"]["passed"] is True
    assert "macro_f1" in payload["metrics"]
    assert payload["metrics"]["macro_f1"] >= 0.0
    assert payload["n_train"] > 0
    assert payload["n_test"] > 0
    assert payload["feature_columns"]
    assert payload["split"]["strategy"] == "stratified"


def test_training_registers_a_promoted_manifest_with_provenance(trained_crop_run) -> None:
    registry = ModelRegistry(trained_crop_run.registry_root)
    assert registry.latest_version("crop_recommendation") == trained_crop_run.result["version"]

    entry = registry.get("crop_recommendation")
    assert entry.exists() is True
    assert entry.manifest["data_digest"]
    assert entry.manifest["config_digest"]
    assert entry.manifest["metrics"]["macro_f1"] == pytest.approx(
        trained_crop_run.result["metrics"]["macro_f1"]
    )
    assert entry.manifest["n_rows"] >= entry.manifest["n_train"] + entry.manifest["n_test"]
