"""The training and evaluation pipelines.

train.py and evaluate.py in every tabular module are thin wrappers around the two
functions here, so the discipline cannot vary between modules:

  * the test partition is materialised once and never used for tuning
  * tuning happens with cross-validation on the training partition only
  * a metric gate is always enforced, and a failed gate is a non-zero exit code
  * every artefact is registered with a manifest that makes it reproducible

evaluate_module deliberately re-reads the dataset and rebuilds the split rather
than loading a serialised test set. If the data changed on disk, the manifest
digest will not match and the evaluation refuses to run - which is what stops a
stale metric from being published as a fresh one.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from agri.common.config import ModuleConfig, load_module_config
from agri.common.errors import AgriError, ConfigError
from agri.common.logging import get_logger
from agri.common.metrics import (
    baseline_predictions,
    compute_metrics,
    default_baseline,
    evaluate_gate,
)
from agri.common.paths import FIGURES_DIR, git_sha, sha256_file
from agri.common.registry import ModelRegistry
from agri.common.schemas import BaselineKind, TaskType
from agri.common.seed import set_global_seed
from agri.common.splits import cross_validation_folds
from agri.common.tabular import (
    BUNDLE_FILENAME,
    TabularBundle,
    TabularPredictor,
    apply_target_transform,
    build_estimator,
    evaluate_predictions,
    feature_matrix,
    invert_target_transform,
    plot_confusion,
    plot_predictions,
    read_frame,
    split_and_validate,
    write_metrics_report,
)
from agri.common.validation import validate_frame
from agri.features import Pipeline

_log = get_logger("training")

CLASSIFICATION_TASKS = {TaskType.CLASSIFICATION, TaskType.INFERENCE}

#: Deliberately modest grids: a wide random search on a 100-row dataset is noise,
#: and tuning is where overfitting to the validation set creeps in unnoticed.
TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 2, 4],
    },
    "ExtraTrees": {
        "n_estimators": [200, 400],
        "max_depth": [None, 12],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
    },
    "GradientBoosting": {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
}


def load_predictor_class(module_name: str) -> type[TabularPredictor] | None:
    """Import a module's Predictor class. Returns None when the package is absent."""
    try:
        package = importlib.import_module(f"agri.modules.{module_name}")
    except ImportError as exc:
        _log.warning("could not import module package", extra={"module": module_name, "error": str(exc)})
        return None
    predictor = getattr(package, "Predictor", None)
    return predictor if isinstance(predictor, type) else None


def _is_classification(config: ModuleConfig) -> bool:
    return config.task in CLASSIFICATION_TASKS


def _encode_labels(labels: list[str], values: np.ndarray) -> np.ndarray:
    encoder = {label: index for index, label in enumerate(labels)}
    encoded = np.array([encoder.get(str(v), -1) for v in values])
    if (encoded == -1).any():
        offenders = sorted({str(v) for v in values if str(v) not in encoder})
        raise ConfigError(
            "Target contains values outside the declared class list",
            context={"offenders": offenders, "declared": labels},
        )
    return encoded


def _decode_labels(labels: list[str], encoded: np.ndarray) -> np.ndarray:
    return np.array([labels[int(i)] if 0 <= int(i) < len(labels) else str(i) for i in encoded])


def tune_hyperparameters(
    config: ModuleConfig, matrix: pd.DataFrame, target: np.ndarray, *, enable: bool, n_iter: int
) -> dict[str, Any]:
    """Cross-validated search on the training partition. The test set is not in scope."""
    if not enable:
        return {}
    grid = TUNING_GRIDS.get(config.model.algorithm)
    if not grid:
        _log.info(
            "no tuning grid for this algorithm; using configured params",
            extra={"algorithm": config.model.algorithm},
        )
        return {}

    scoring = "f1_macro" if _is_classification(config) else "neg_root_mean_squared_error"
    probe = pd.DataFrame(matrix, columns=matrix.columns)
    probe["__target__"] = target
    folds = cross_validation_folds(probe, config.split, target="__target__")
    if not folds:
        return {}

    combinations = 1
    for values in grid.values():
        combinations *= len(values)

    estimator = build_estimator(config)
    common: dict[str, Any] = {
        "estimator": estimator,
        "scoring": scoring,
        "cv": folds,
        "n_jobs": -1,
        "refit": True,
    }
    try:
        if combinations > 12:
            search: Any = RandomizedSearchCV(
                param_distributions=grid, n_iter=n_iter, random_state=config.seed, **common
            )
        else:
            search = GridSearchCV(param_grid=grid, **common)
        search.fit(matrix, target)
    except Exception as exc:
        _log.warning(
            "hyperparameter search failed; falling back to configured params", extra={"error": str(exc)}
        )
        return {}

    _log.info(
        "tuning complete", extra={"best_score": float(search.best_score_), "best_params": search.best_params_}
    )
    return dict(search.best_params_)


def _calibrate(model: Any, matrix: pd.DataFrame, target: np.ndarray, config: ModuleConfig) -> Any:
    """Wrap a classifier in probability calibration when configured.

    Calibration is an improvement, never a requirement: if it fails (usually
    because a class has too few members for the requested fold count) the
    uncalibrated model is served and the substitution is logged.
    """
    if not config.model.calibrate or not _is_classification(config) or not hasattr(model, "predict_proba"):
        return model
    counts = np.bincount(np.asarray(target, dtype=int))
    smallest = int(counts[counts > 0].min()) if (counts > 0).any() else 2
    folds = max(2, min(config.split.n_splits, smallest))
    try:
        calibrated = CalibratedClassifierCV(model, method=config.model.calibration_method, cv=folds)
        calibrated.fit(matrix, target)
        return calibrated
    except Exception as exc:
        _log.warning("calibration failed; serving the uncalibrated model", extra={"error": str(exc)})
        return model


def _feature_importances(model: Any, columns: list[str]) -> dict[str, float]:
    raw = getattr(model, "feature_importances_", None)
    if raw is None and hasattr(model, "calibrated_classifiers_"):
        try:
            raw = np.mean([c.estimator.feature_importances_ for c in model.calibrated_classifiers_], axis=0)
        except Exception:
            raw = None
    if raw is None:
        return {}
    raw = np.asarray(raw, dtype="float64").ravel()
    if raw.size != len(columns):
        return {}
    return {name: float(value) for name, value in zip(columns, raw, strict=False)}


def _baseline_metrics(config: ModuleConfig, splits: Any, labels: list[str]) -> dict[str, float]:
    """Score the naive comparator with the same metric code as the real model."""
    target = config.data.target
    assert target is not None
    kind = BaselineKind(config.gates.baseline) if config.gates.baseline else default_baseline(config.task)

    if _is_classification(config):
        y_train = _encode_labels(labels, splits.train[target].to_numpy())
        y_test = _encode_labels(labels, splits.test[target].to_numpy())
        predictions = baseline_predictions(kind, y_train, n_test=len(y_test))
        return compute_metrics(
            _decode_labels(labels, y_test), _decode_labels(labels, predictions), task=config.task
        )

    y_train = splits.train[target].to_numpy(dtype="float64")
    y_test = splits.test[target].to_numpy(dtype="float64")
    last_value = None
    if config.split.time_column and config.split.time_column in splits.train.columns:
        ordered = splits.train.sort_values(config.split.time_column)
        last_value = float(ordered[target].iloc[-1])
    predictions = baseline_predictions(kind, y_train, n_test=len(y_test), last_train_value=last_value)
    return compute_metrics(y_test, predictions, task=config.task)


def train_module(
    module_name: str, *, tune: bool = True, n_iter: int = 12, register: bool = True
) -> dict[str, Any]:
    """Train, evaluate on the held-out partition, register, and enforce the gate.

    Raises AgriError when the gate fails: that exception is what makes the CLI
    exit code usable as a CI gate.
    """
    config = load_module_config(module_name)
    predictor_class = load_predictor_class(module_name)
    if predictor_class is None:
        raise ConfigError(f"No Predictor class found for module '{module_name}'")
    if config.data.target is None:
        raise ConfigError(f"Module '{module_name}' declares no target and cannot be trained")

    set_global_seed(config.seed)
    registry = ModelRegistry()

    # Order matters: read, engineer, then validate. The risk module derives its
    # training label, so the schema check can only be meaningful afterwards.
    frame = read_frame(config)
    frame = predictor_class.prepare_features(frame)
    report = validate_frame(frame, config.data, name=config.data.raw_path.name)
    report.raise_if_invalid()
    _log.info("dataset loaded", extra=report.summary())

    splits, _ = split_and_validate(config, frame)
    target = config.data.target

    # The feature pipeline is built and fitted on the training partition only.
    pipeline: Pipeline = predictor_class.build_feature_pipeline(config, splits.train)
    train_matrix = feature_matrix(pipeline, splits.train)
    test_matrix = feature_matrix(pipeline, splits.test)
    feature_columns = list(train_matrix.columns)

    if _is_classification(config):
        labels = config.data.classes or sorted(pd.Series(splits.train[target]).astype(str).unique().tolist())
        y_train = _encode_labels(labels, splits.train[target].to_numpy())
        y_test = _encode_labels(labels, splits.test[target].to_numpy())
        y_test_raw = splits.test[target].to_numpy()
    else:
        labels = []
        y_train = apply_target_transform(
            splits.train[target].to_numpy(dtype="float64"), config.model.target_transform
        )
        y_test_raw = splits.test[target].to_numpy(dtype="float64")
        y_test = apply_target_transform(y_test_raw, config.model.target_transform)

    best_params = tune_hyperparameters(config, train_matrix, y_train, enable=tune, n_iter=n_iter)

    # Final fit on train plus the validation partition when one exists.
    fit_matrix, fit_target = train_matrix, y_train
    if splits.validation is not None and len(splits.validation):
        valid_matrix = feature_matrix(pipeline, splits.validation)
        fit_matrix = pd.concat([train_matrix, valid_matrix], ignore_index=True)
        if _is_classification(config):
            fit_target = np.concatenate(
                [y_train, _encode_labels(labels, splits.validation[target].to_numpy())]
            )
        else:
            fit_target = np.concatenate(
                [
                    y_train,
                    apply_target_transform(
                        splits.validation[target].to_numpy(dtype="float64"), config.model.target_transform
                    ),
                ]
            )

    model = build_estimator(config, params=best_params)
    model.fit(fit_matrix, fit_target)
    model = _calibrate(model, fit_matrix, fit_target, config)

    # -------------------------------------------------- evaluate on test only
    segment = None
    for candidate in (config.data.group_column, "Crop", "State"):
        if candidate and candidate in splits.test.columns:
            segment = splits.test[candidate]
            break

    if _is_classification(config):
        metrics, detail = evaluate_predictions(config, model, test_matrix, y_test, labels, segment=segment)
    else:
        # Report regression metrics in the target's real units, not the
        # transformed space, so a log-transform does not flatter the numbers.
        predictions = invert_target_transform(
            np.asarray(model.predict(test_matrix)).ravel(), config.model.target_transform
        )
        metrics, detail = evaluate_predictions(
            config, _IdentityModel(predictions), test_matrix, y_test_raw, [], segment=segment
        )

    baseline = _baseline_metrics(config, splits, labels)
    primary = config.gates.primary_metric
    if primary in baseline:
        metrics["baseline_score"] = float(baseline[primary])
        denominator = abs(baseline[primary]) if abs(baseline[primary]) > 1e-12 else 1e-12
        metrics["relative_lift_over_baseline"] = float((metrics[primary] - baseline[primary]) / denominator)

    gate = evaluate_gate(metrics, config.gates)
    _log.info("gate evaluated", extra=gate.to_dict())

    residual_std = None
    if not _is_classification(config):
        residuals = np.asarray(y_test_raw, dtype="float64") - invert_target_transform(
            np.asarray(model.predict(test_matrix)).ravel(), config.model.target_transform
        )
        residual_std = float(np.std(residuals))

    bundle = TabularBundle(
        model=model,
        feature_pipeline=pipeline,
        feature_columns=feature_columns,
        classes=labels,
        target_transform=config.model.target_transform,
        residual_std=residual_std,
        trained_metrics={k: float(v) for k, v in metrics.items()},
        feature_importances=_feature_importances(model, feature_columns),
    )

    data_digest = sha256_file(config.data.raw_path)
    if register:
        registry.save(
            config.name,
            config.version,
            bundle,
            metrics={k: float(v) for k, v in metrics.items()},
            config_digest=config.config_digest,
            data_digest=data_digest,
            dataset=config.data.raw_path.name,
            params=best_params or config.model.params,
            artifact_file=BUNDLE_FILENAME,
            extra={
                "split": config.split.model_dump(),
                "n_train": len(splits.train),
                "n_test": len(splits.test),
                "n_rows": len(frame),
                "baseline_metrics": _jsonable(baseline),
                "gate": gate.to_dict(),
                "feature_columns": feature_columns,
                "classes": labels,
            },
        )

    write_metrics_report(
        config,
        {k: float(v) for k, v in metrics.items()},
        detail,
        gate.to_dict(),
        manifest={"git_sha": git_sha(), "data_digest": data_digest},
        extra={
            "baseline_metrics": _jsonable(baseline),
            "best_params": best_params,
            "n_train": len(splits.train),
            "n_test": len(splits.test),
            "feature_columns": feature_columns,
        },
    )

    if _is_classification(config):
        plot_confusion(detail, config.name, FIGURES_DIR / f"{config.name}_confusion.png")
    else:
        predictions = invert_target_transform(
            np.asarray(model.predict(test_matrix)).ravel(), config.model.target_transform
        )
        plot_predictions(
            np.asarray(y_test_raw, dtype="float64"),
            predictions,
            config.name,
            FIGURES_DIR / f"{config.name}_predictions.png",
        )

    if not gate.passed:
        raise AgriError(
            f"Metric gate failed for {config.name}: {', '.join(gate.failed_checks)}",
            context={
                "primary_metric": gate.primary_metric,
                "value": gate.primary_value,
                "messages": gate.messages,
            },
        )

    return {
        "module": config.name,
        "version": config.version,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "gate": gate.to_dict(),
    }


def evaluate_module(module_name: str, *, version: str | None = None) -> dict[str, Any]:
    """Re-evaluate a registered artefact against a freshly rebuilt test split.

    Refuses to run when the dataset on disk no longer matches the digest recorded
    at training time, because the resulting metric would be attached to the wrong
    data and could not be reproduced.
    """
    config = load_module_config(module_name)
    predictor_class = load_predictor_class(module_name)
    if predictor_class is None:
        raise ConfigError(f"No Predictor class found for module '{module_name}'")
    if config.data.target is None:
        raise ConfigError(f"Module '{module_name}' declares no target and cannot be evaluated")

    set_global_seed(config.seed)
    registry = ModelRegistry()
    entry = registry.get(module_name, version)

    recorded = entry.manifest.get("data_digest")
    current = sha256_file(config.data.raw_path) if config.data.raw_path.is_file() else None
    if recorded and current and recorded != current:
        raise AgriError(
            f"Dataset digest changed since {module_name} v{entry.version} was trained; refusing to evaluate",
            context={"recorded": recorded, "current": current, "dataset": config.data.raw_path.name},
        )

    frame = read_frame(config)
    frame = predictor_class.prepare_features(frame)
    splits, _ = split_and_validate(config, frame)
    target = config.data.target

    bundle: TabularBundle = entry.load()
    test_matrix = feature_matrix(bundle.feature_pipeline, splits.test)

    segment = None
    for candidate in (config.data.group_column, "Crop", "State"):
        if candidate and candidate in splits.test.columns:
            segment = splits.test[candidate]
            break

    if _is_classification(config):
        labels = bundle.classes
        y_test = _encode_labels(labels, splits.test[target].to_numpy())
        metrics, detail = evaluate_predictions(
            config, bundle.model, test_matrix, y_test, labels, segment=segment
        )
    else:
        y_true = splits.test[target].to_numpy(dtype="float64")
        predictions = invert_target_transform(
            np.asarray(bundle.model.predict(test_matrix)).ravel(), bundle.target_transform
        )
        metrics, detail = evaluate_predictions(
            config, _IdentityModel(predictions), test_matrix, y_true, [], segment=segment
        )

    gate = evaluate_gate(metrics, config.gates)
    write_metrics_report(
        config,
        {k: float(v) for k, v in metrics.items()},
        detail,
        gate.to_dict(),
        manifest=entry.manifest,
        extra={"evaluated_from_registry": True, "version": entry.version},
    )
    _log.info(
        "evaluation complete",
        extra={"module": module_name, "version": entry.version, "passed": gate.passed},
    )
    if not gate.passed:
        raise AgriError(
            f"Metric gate failed for {module_name} v{entry.version}: {', '.join(gate.failed_checks)}",
            context={"metrics": _jsonable(metrics), "gate": gate.to_dict()},
        )
    return {"module": module_name, "version": entry.version, "metrics": metrics, "gate": gate.to_dict()}


class _IdentityModel:
    """Adapter letting pre-computed predictions flow through the metric helpers."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = np.asarray(predictions)

    def predict(self, _matrix: Any) -> np.ndarray:
        return self._predictions


def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (bool, str)) or value is None:
            out[key] = value
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        else:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def add_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--no-tune", action="store_true", help="skip hyperparameter search")
    parser.add_argument("--n-iter", type=int, default=12, help="randomised search iterations")
    parser.add_argument("--no-register", action="store_true", help="train without writing to the registry")
    return parser
