"""CLI gate exit-code test.

What this protects against
--------------------------
A training command that returns success even when the model failed its metric
gate. CI branches on the exit code, so a gate that no longer blocks a merge turns
the metric floor into decoration. The test makes the floor impossible to satisfy
(macro_f1 is bounded by 1.0, the floor is set to 1.5) and asserts that the real
CLI path returns 1 and says why.

The estimator is shrunk to a handful of trees and calibration is disabled so the
test exercises the gate wiring rather than spending a minute fitting the
production forest; the data loading, splitting, metric computation and gate
evaluation are all the real implementations.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


def test_run_train_returns_one_when_the_gate_cannot_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_registry, isolated_output_dirs, capsys
) -> None:
    import agri.common.training as training_module
    from agri.common.config import load_module_config
    from agri.common.runner import run_train

    impossible = load_module_config("crop_recommendation").model_copy(deep=True)
    impossible.gates.minimum = 1.5
    impossible.model.params = {"n_estimators": 5, "n_jobs": 1}
    impossible.model.calibrate = False

    monkeypatch.setattr(training_module, "load_module_config", lambda name, config_dir=None: impossible)

    exit_code = run_train("crop_recommendation", ["--no-tune", "--no-register"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Metric gate failed" in captured.out
    assert '"ok": false' in captured.out
