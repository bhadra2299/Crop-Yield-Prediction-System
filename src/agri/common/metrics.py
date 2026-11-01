"""Metrics, naive baselines and gate evaluation.

Every metric here is computed on held-out data by the caller. This module never
sees the training set, which is the whole point: a metric that can be computed
on training data will eventually be reported as if it were a test metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn import metrics as skm

from agri.common.schemas import BaselineKind, TaskType

_EPS = 1e-12


# --------------------------------------------------------------------- basics
def _as_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values).ravel()


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(skm.accuracy_score(_as_array(y_true), _as_array(y_pred)))


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(skm.f1_score(_as_array(y_true), _as_array(y_pred), average="macro", zero_division=0))


def weighted_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(skm.f1_score(_as_array(y_true), _as_array(y_pred), average="weighted", zero_division=0))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(skm.mean_squared_error(_as_array(y_true), _as_array(y_pred))))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(skm.mean_absolute_error(_as_array(y_true), _as_array(y_pred)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(skm.r2_score(_as_array(y_true), _as_array(y_pred)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, *, mask_zeros: bool = True) -> float:
    """Mean absolute percentage error, as a percentage.

    Zero and near-zero actuals make MAPE explode, so they are masked by default
    and the caller is told how many rows were dropped.
    """
    actual = _as_array(y_true).astype(float)
    predicted = _as_array(y_pred).astype(float)
    mask = np.abs(actual) > _EPS if mask_zeros else np.ones_like(actual, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def median_ape(y_true: ArrayLike, y_pred: ArrayLike, *, mask_zeros: bool = True) -> float:
    """Median absolute percentage error - robust where MAPE is not."""
    actual = _as_array(y_true).astype(float)
    predicted = _as_array(y_pred).astype(float)
    mask = np.abs(actual) > _EPS if mask_zeros else np.ones_like(actual, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Share of periods where the predicted direction of change is correct.

    This is the metric that matters for a sell-now-or-wait signal, and it is
    independent of the magnitude error that MAPE captures.
    """
    actual = np.diff(_as_array(y_true).astype(float))
    predicted = np.diff(_as_array(y_pred).astype(float))
    if actual.size == 0:
        return float("nan")
    return float(np.mean(np.sign(actual) == np.sign(predicted)))


def confusion(y_true: ArrayLike, y_pred: ArrayLike, labels: list[Any] | None = None) -> dict[str, Any]:
    """Confusion matrix plus its label order, serialisable to JSON."""
    cm = skm.confusion_matrix(_as_array(y_true), _as_array(y_pred), labels=labels)
    resolved = list(labels) if labels is not None else sorted({*_as_array(y_true).tolist()}, key=str)
    return {"labels": [str(label) for label in resolved], "matrix": cm.tolist()}


def per_class_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, dict[str, float]]:
    """Precision/recall/F1/support per class. Never report only the aggregate."""
    report = skm.classification_report(
        _as_array(y_true), _as_array(y_pred), output_dict=True, zero_division=0
    )
    return {
        str(label): {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}
        for label, stats in report.items()
        if isinstance(stats, dict)
    }


def roc_auc_ovr(y_true: ArrayLike, y_score: ArrayLike, *, multi_class: str = "ovr") -> float:
    actual = _as_array(y_true)
    if len(np.unique(actual)) < 2:
        return float("nan")
    try:
        return float(skm.roc_auc_score(actual, y_score, multi_class=multi_class, average="weighted"))
    except ValueError:
        return float("nan")


def per_segment_errors(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    segment: ArrayLike,
    *,
    task: TaskType | None = None,
) -> dict[str, dict[str, float]]:
    """Per-segment error, metric-appropriate to the task.

    An aggregate score hides the crop or state the model is actually bad at;
    this surfaces it. Regression segments report RMSE/MAE/bias, classification
    segments report accuracy and support - computing RMSE over label strings
    would raise, and comparing labels with MSE would be meaningless anyway.
    """
    actual, predicted, groups = _as_array(y_true), _as_array(y_pred), _as_array(segment)
    classification = task in {TaskType.CLASSIFICATION, TaskType.INFERENCE}
    out: dict[str, dict[str, float]] = {}
    for value in np.unique(groups):
        mask = groups == value
        if mask.sum() < 2:
            continue
        if classification:
            out[str(value)] = {
                "n": int(mask.sum()),
                "accuracy": accuracy(actual[mask], predicted[mask]),
                "errors": int(np.sum(actual[mask] != predicted[mask])),
            }
        else:
            out[str(value)] = {
                "n": int(mask.sum()),
                "rmse": rmse(actual[mask], predicted[mask]),
                "mae": mae(actual[mask], predicted[mask]),
                "bias": float(np.mean(predicted[mask].astype("float64") - actual[mask].astype("float64"))),
            }
    return out


# ------------------------------------------------------------------ baselines
def baseline_predictions(
    kind: BaselineKind,
    y_train: ArrayLike,
    *,
    n_test: int,
    y_test: ArrayLike | None = None,
    last_train_value: float | None = None,
) -> np.ndarray:
    """Predictions from the naive comparator a real model must beat."""
    train = _as_array(y_train)
    if kind is BaselineKind.MAJORITY_CLASS:
        values, counts = np.unique(train, return_counts=True)
        return np.full(n_test, values[int(np.argmax(counts))])
    if kind is BaselineKind.MEAN:
        return np.full(n_test, float(np.mean(train)))
    if kind is BaselineKind.MEDIAN:
        return np.full(n_test, float(np.median(train)))
    if kind in (BaselineKind.LAST_VALUE, BaselineKind.PERSISTENCE):
        seed = last_train_value if last_train_value is not None else float(train[-1])
        return np.full(n_test, float(seed))
    raise ValueError(f"Unsupported baseline kind: {kind}")


def default_baseline(task: TaskType) -> BaselineKind:
    """The baseline appropriate to a task type when the config does not name one."""
    if task is TaskType.CLASSIFICATION:
        return BaselineKind.MAJORITY_CLASS
    if task is TaskType.FORECASTING:
        return BaselineKind.LAST_VALUE
    return BaselineKind.MEDIAN


# ---------------------------------------------------------------- aggregation
def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    task: TaskType,
    y_score: ArrayLike | None = None,
    segment: ArrayLike | None = None,
    baseline_value: float | None = None,
) -> dict[str, float]:
    """All primary and secondary metrics for a task, flattened for the manifest."""
    actual = _as_array(y_true)
    predicted = _as_array(y_pred)
    metrics: dict[str, float] = {}

    if task in (TaskType.CLASSIFICATION, TaskType.INFERENCE):
        metrics["accuracy"] = accuracy(actual, predicted)
        metrics["macro_f1"] = macro_f1(actual, predicted)
        metrics["weighted_f1"] = weighted_f1(actual, predicted)
        if y_score is not None:
            auc = roc_auc_ovr(actual, y_score)
            if not np.isnan(auc):
                metrics["roc_auc_ovr"] = auc
        report = per_class_report(actual, predicted)
        high = report.get("High") or report.get("high")
        if high:
            metrics["recall_high_risk"] = float(high.get("recall", float("nan")))
            metrics["precision_high_risk"] = float(high.get("precision", float("nan")))
    else:
        metrics["rmse"] = rmse(actual, predicted)
        metrics["mae"] = mae(actual, predicted)
        metrics["r2"] = r2(actual, predicted)
        metrics["mape"] = mape(actual, predicted)
        metrics["median_ape"] = median_ape(actual, predicted)
        if task is TaskType.FORECASTING:
            metrics["directional_accuracy"] = directional_accuracy(actual, predicted)

    if baseline_value is not None and not np.isnan(baseline_value):
        metrics["baseline_score"] = float(baseline_value)
        primary = "macro_f1" if task in (TaskType.CLASSIFICATION, TaskType.INFERENCE) else "rmse"
        if primary in metrics:
            delta = metrics[primary] - baseline_value
            metrics["lift_over_baseline"] = float(delta)
        # A ratio makes lift comparable across modules with different units.
        denominator = abs(baseline_value) if abs(baseline_value) > _EPS else _EPS
        metrics["relative_lift_over_baseline"] = float(
            (metrics[primary] - baseline_value) / denominator if primary in metrics else float("nan")
        )
    return metrics


@dataclass
class GateResult:
    """Outcome of comparing metrics against the configured floor."""

    passed: bool
    primary_metric: str
    primary_value: float
    messages: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "messages": self.messages,
            "failed_checks": self.failed_checks,
        }


def evaluate_gate(metrics: dict[str, float], gates: Any) -> GateResult:
    """Apply GateSpec to a metrics dict. Missing primary metric is a failure."""
    primary = gates.primary_metric
    if primary not in metrics or np.isnan(metrics[primary]):
        return GateResult(
            passed=False,
            primary_metric=primary,
            primary_value=float("nan"),
            messages=[f"Primary metric '{primary}' was not computed"],
            failed_checks=[primary],
        )

    value = float(metrics[primary])
    ok, message = gates.check(value)
    result = GateResult(passed=ok, primary_metric=primary, primary_value=value, messages=[message])
    if not ok:
        result.failed_checks.append(primary)

    for metric_name, floor in getattr(gates, "secondary_minimums", {}).items():
        if metric_name not in metrics:
            result.passed = False
            result.failed_checks.append(metric_name)
            result.messages.append(f"secondary metric '{metric_name}' was not computed")
            continue
        observed = float(metrics[metric_name])
        if np.isnan(observed):
            result.passed = False
            result.failed_checks.append(metric_name)
            result.messages.append(f"secondary metric '{metric_name}' was not computed")
        elif observed < float(floor):
            result.passed = False
            result.failed_checks.append(metric_name)
            result.messages.append(f"{metric_name}={observed:.4f} is below the floor {floor}")

    if gates.must_beat_baseline and "baseline_score" in metrics:
        # Lower is better for error metrics; higher is better for scores.
        lower_is_better = primary in {"rmse", "mae", "mape", "median_ape"}
        baseline = float(metrics["baseline_score"])
        better = value < baseline if lower_is_better else value > baseline
        if not better:
            result.passed = False
            result.failed_checks.append("baseline")
            result.messages.append(f"{primary}={value:.4f} does not beat the naive baseline {baseline:.4f}")
        else:
            result.messages.append(f"beats the naive baseline ({value:.4f} vs {baseline:.4f})")
    return result
