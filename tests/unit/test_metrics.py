"""Hand-computed metric, baseline and gate tests.

What this protects against
--------------------------
A metric that is subtly wrong and therefore silently overstates a model. Every
expected value here is worked out arithmetically in the test itself, so a change
to the averaging strategy, the zero-division policy or the direction of an error
metric is caught immediately. No test asserts a model accuracy number: those
drift with the data, and a drifting number proves nothing about correctness.

The gate tests protect the CI contract. A value below its floor, an uncomputed
primary metric and a failure to beat the naive baseline must each produce a
failing GateResult that names the check which failed.
"""

from __future__ import annotations

import math

import pytest

from agri.common.config import GateSpec
from agri.common.metrics import (
    accuracy,
    baseline_predictions,
    compute_metrics,
    evaluate_gate,
    macro_f1,
    mae,
    mape,
    r2,
    rmse,
    weighted_f1,
)
from agri.common.schemas import BaselineKind, TaskType

pytestmark = pytest.mark.unit


# ------------------------------------------------------- classification
def test_accuracy_equals_the_share_of_matching_labels() -> None:
    # three of the four labels agree
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.75)


def test_macro_f1_is_the_unweighted_mean_of_the_per_class_f1_scores() -> None:
    # class 1: precision 1.0 and recall 0.5 -> f1 2/3
    # class 0: precision 2/3 and recall 1.0 -> f1 0.8
    expected = ((2 * 1.0 * 0.5) / 1.5 + (2 * (2 / 3) * 1.0) / (5 / 3)) / 2
    assert macro_f1([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(expected)
    assert macro_f1([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.7333333333333334)


def test_weighted_f1_weights_each_class_by_its_support() -> None:
    expected = (2 * (2 / 3) + 2 * 0.8) / 4
    assert weighted_f1([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(expected)


def test_macro_f1_scores_a_never_predicted_class_as_zero() -> None:
    # class 0 is never predicted, so its f1 is 0; class 1 has precision 1/3 and
    # recall 1.0, so its f1 is 0.5; the macro mean is 0.25.
    assert macro_f1([0, 0, 1], [1, 1, 1]) == pytest.approx(0.25)


# ------------------------------------------------------------ regression
def test_rmse_mae_and_r2_match_their_definitions() -> None:
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 4.0]
    # residuals are 0, 0, 1 -> mse 1/3
    assert rmse(actual, predicted) == pytest.approx((1.0 / 3.0) ** 0.5)
    assert mae(actual, predicted) == pytest.approx(1.0 / 3.0)
    # ss_res 1, ss_tot 2
    assert r2(actual, predicted) == pytest.approx(0.5)


def test_mape_is_a_percentage_and_masks_zero_actuals_by_default() -> None:
    # |10/100| and |20/200| average to 0.1, reported as 10 percent
    assert mape([100.0, 200.0], [110.0, 180.0]) == pytest.approx(10.0)
    # a zero actual would divide by zero; it must be dropped, not propagated
    assert mape([0.0, 200.0], [1.0, 180.0]) == pytest.approx(10.0)
    assert mape([0.0, 0.0], [1.0, 1.0]) != mape([0.0, 0.0], [1.0, 1.0])  # NaN


def test_mape_without_masking_is_non_finite_for_zero_actuals() -> None:
    # This is exactly why masking is the default: a zero actual makes the
    # percentage error infinite rather than merely large.
    value = mape([0.0, 200.0], [10.0, 180.0], mask_zeros=False)
    assert not math.isfinite(value)


# -------------------------------------------------------- compute_metrics
def test_compute_metrics_classification_reports_the_primary_family() -> None:
    result = compute_metrics([1, 1, 0, 0], [1, 0, 0, 0], task=TaskType.CLASSIFICATION)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(0.7333333333333334)
    assert result["weighted_f1"] == pytest.approx(0.7333333333333334)


def test_compute_metrics_regression_reports_the_error_family() -> None:
    result = compute_metrics(
        [100.0, 200.0, 300.0, 400.0], [110.0, 190.0, 300.0, 420.0], task=TaskType.REGRESSION
    )
    assert result["mae"] == pytest.approx(10.0)
    assert result["rmse"] == pytest.approx(150.0**0.5)
    assert result["r2"] == pytest.approx(1.0 - 600.0 / 50000.0)
    assert result["mape"] == pytest.approx(5.0)
    assert result["median_ape"] == pytest.approx(5.0)
    assert "directional_accuracy" not in result


def test_compute_metrics_forecasting_adds_directional_accuracy() -> None:
    result = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.5, 2.5], task=TaskType.FORECASTING)
    assert "directional_accuracy" in result


def test_compute_metrics_records_lift_over_the_naive_baseline() -> None:
    result = compute_metrics([1, 1, 0, 0], [1, 0, 0, 0], task=TaskType.CLASSIFICATION, baseline_value=0.25)
    assert result["baseline_score"] == pytest.approx(0.25)
    assert result["lift_over_baseline"] == pytest.approx(result["macro_f1"] - 0.25)
    assert result["relative_lift_over_baseline"] == pytest.approx((result["macro_f1"] - 0.25) / 0.25)


# ------------------------------------------------------- evaluate_gate
def test_evaluate_gate_passes_a_value_above_the_floor() -> None:
    gate = evaluate_gate(
        {"macro_f1": 0.80}, GateSpec(primary_metric="macro_f1", minimum=0.70, must_beat_baseline=False)
    )
    assert gate.passed is True
    assert gate.failed_checks == []
    assert gate.primary_metric == "macro_f1"
    assert gate.primary_value == pytest.approx(0.80)
    assert gate.messages


def test_evaluate_gate_fails_a_value_below_the_floor_and_names_the_check() -> None:
    gate = evaluate_gate(
        {"macro_f1": 0.60}, GateSpec(primary_metric="macro_f1", minimum=0.70, must_beat_baseline=False)
    )
    assert gate.passed is False
    assert gate.failed_checks == ["macro_f1"]
    assert gate.to_dict()["passed"] is False


def test_evaluate_gate_fails_when_the_primary_metric_was_not_computed() -> None:
    gate = evaluate_gate(
        {"accuracy": 0.99}, GateSpec(primary_metric="macro_f1", minimum=0.70, must_beat_baseline=False)
    )
    assert gate.passed is False
    assert gate.failed_checks == ["macro_f1"]
    assert gate.primary_value != gate.primary_value  # NaN: nothing was measured


def test_evaluate_gate_enforces_a_ceiling_for_error_metrics() -> None:
    ceiling = GateSpec(primary_metric="rmse", maximum=1.0, must_beat_baseline=False)
    assert evaluate_gate({"rmse": 0.5}, ceiling).passed is True
    assert evaluate_gate({"rmse": 1.5}, ceiling).passed is False


def test_evaluate_gate_requires_beating_the_baseline_in_the_right_direction() -> None:
    higher_is_better = GateSpec(primary_metric="macro_f1", minimum=0.5, must_beat_baseline=True)
    assert evaluate_gate({"macro_f1": 0.9, "baseline_score": 0.2}, higher_is_better).passed is True
    beaten = evaluate_gate({"macro_f1": 0.1, "baseline_score": 0.2}, higher_is_better)
    assert beaten.passed is False
    assert "baseline" in beaten.failed_checks

    lower_is_better = GateSpec(primary_metric="rmse", maximum=2.0, must_beat_baseline=True)
    assert evaluate_gate({"rmse": 0.5, "baseline_score": 0.8}, lower_is_better).passed is True
    assert evaluate_gate({"rmse": 0.9, "baseline_score": 0.8}, lower_is_better).passed is False


def test_evaluate_gate_enforces_the_secondary_metric_floors() -> None:
    gate = GateSpec(
        primary_metric="macro_f1",
        minimum=0.5,
        secondary_minimums={"accuracy": 0.9},
        must_beat_baseline=False,
    )
    assert evaluate_gate({"macro_f1": 0.8, "accuracy": 0.95}, gate).passed is True

    failed = evaluate_gate({"macro_f1": 0.8, "accuracy": 0.5}, gate)
    assert failed.passed is False
    assert failed.failed_checks == ["accuracy"]


def test_evaluate_gate_treats_absent_secondary_floors_as_unset() -> None:
    gate = GateSpec(primary_metric="macro_f1", minimum=0.5, must_beat_baseline=False)
    assert gate.secondary_minimums == {}
    assert evaluate_gate({"macro_f1": 0.8}, gate).passed is True


# --------------------------------------------------------- baselines
def test_majority_class_baseline_predicts_the_most_frequent_training_label() -> None:
    predictions = baseline_predictions(BaselineKind.MAJORITY_CLASS, [1, 1, 1, 2, 2], n_test=3)
    assert list(predictions) == [1, 1, 1]


def test_median_baseline_predicts_the_training_median() -> None:
    predictions = baseline_predictions(BaselineKind.MEDIAN, [1.0, 2.0, 3.0, 4.0, 5.0], n_test=2)
    assert list(predictions) == [3.0, 3.0]


def test_last_value_baseline_repeats_the_final_training_observation() -> None:
    predictions = baseline_predictions(BaselineKind.LAST_VALUE, [1.0, 2.0, 3.0], n_test=2)
    assert list(predictions) == [3.0, 3.0]
    overridden = baseline_predictions(
        BaselineKind.LAST_VALUE, [1.0, 2.0, 3.0], n_test=2, last_train_value=9.5
    )
    assert list(overridden) == [9.5, 9.5]


def test_persistence_is_an_alias_for_last_value() -> None:
    predictions = baseline_predictions(BaselineKind.PERSISTENCE, [4.0, 5.0], n_test=2)
    assert list(predictions) == [5.0, 5.0]


def test_mean_baseline_predicts_the_training_mean() -> None:
    predictions = baseline_predictions(BaselineKind.MEAN, [1.0, 2.0, 3.0, 4.0], n_test=2)
    assert list(predictions) == [2.5, 2.5]
