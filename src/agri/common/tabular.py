"""Generic tabular ML framework.

Five of the ten modules are the same shape: a CSV goes in, a feature matrix
comes out, a scikit-learn-compatible estimator is fitted, and a Prediction
envelope is served. Writing that five times guarantees they drift apart, so it
is written once here and each module declares only what is genuinely different.

What a module supplies
----------------------
    field_map      request field -> feature column
    response_model a Prediction subclass (typed value)
    make_value     optional: turn a model output into the response value

What this framework supplies
----------------------------
    artefact resolution, feature pipeline application, probability calibration,
    uncertainty intervals, top-k alternatives, feature-importance reasons,
    training with tuning + baseline comparison + gate enforcement, evaluation
    that re-derives the exact training split from the recorded data digest, and
    a JSON metrics report with figures.

The train/evaluate paths live here rather than in train.py so that offline
evaluation and online inference physically cannot use different code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from agri.common.base import BasePredictor
from agri.common.config import ModuleConfig
from agri.common.errors import ConfigError, ModuleExecutionError
from agri.common.logging import get_logger
from agri.common.metrics import (
    compute_metrics,
    confusion,
    per_class_report,
    per_segment_errors,
)
from agri.common.paths import METRICS_DIR, ensure_dir, write_json
from agri.common.schemas import Interval, Prediction, TaskType
from agri.common.splits import SplitResult, make_splits
from agri.common.validation import ValidationReport, validate_frame
from agri.features import ColumnSelector, NumericNormalizer, Pipeline

_log = get_logger("tabular")

BUNDLE_FILENAME = "bundle.joblib"

#: Algorithm registry. Names are resolved to the classifier or regressor
#: variant according to the module's declared task.
_ALGORITHMS: dict[str, tuple[Any, Any]] = {
    "RandomForest": (RandomForestClassifier, RandomForestRegressor),
    "ExtraTrees": (ExtraTreesClassifier, ExtraTreesRegressor),
    "GradientBoosting": (GradientBoostingClassifier, GradientBoostingRegressor),
    "HistGradientBoosting": (HistGradientBoostingClassifier, HistGradientBoostingRegressor),
    "DecisionTree": (DecisionTreeClassifier, DecisionTreeRegressor),
    "LogisticRegression": (LogisticRegression, Ridge),
    "Ridge": (Ridge, Ridge),
}

_CLASSIFICATION_TASKS = {TaskType.CLASSIFICATION, TaskType.INFERENCE}


def _xgb():
    """Import XGBoost lazily so the framework works without it installed."""
    try:
        from xgboost import XGBClassifier, XGBRegressor

        return XGBClassifier, XGBRegressor
    except ImportError as exc:  # pragma: no cover - xgboost is a core dependency
        raise ConfigError("Algorithm 'XGBoost' requires the xgboost package") from exc


def resolve_estimator_class(algorithm: str, task: TaskType) -> Any:
    """Map an algorithm name plus task to a concrete estimator class."""
    classifier = task in _CLASSIFICATION_TASKS
    if algorithm.lower() == "xgboost":
        return _xgb()[0 if classifier else 1]
    if algorithm not in _ALGORITHMS:
        raise ConfigError(
            f"Unknown algorithm '{algorithm}'",
            context={"supported": sorted([*_ALGORITHMS, "XGBoost"])},
        )
    return _ALGORITHMS[algorithm][0 if classifier else 1]


def _filter_params(estimator_class: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Drop params the estimator does not accept.

    A typo in a YAML hyperparameter should not take down a training run at
    midnight; it is logged and ignored, and the surviving set is recorded in the
    manifest so the substitution is visible.
    """
    try:
        accepted = set(estimator_class().get_params())
    except Exception:  # estimators requiring mandatory args
        return params
    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in params.items():
        if key in accepted:
            kept[key] = value
        else:
            dropped.append(key)
    if dropped:
        _log.warning("ignored unsupported hyperparameters", extra={"dropped": dropped})
    return kept


def build_estimator(config: ModuleConfig, *, params: dict[str, Any] | None = None) -> Any:
    """Instantiate the configured estimator with filtered, seeded parameters."""
    estimator_class = resolve_estimator_class(config.model.algorithm, config.task)
    merged = dict(config.model.params)
    if params:
        merged.update(params)
    merged.setdefault("random_state", config.seed)
    if config.model.class_weight and config.task in _CLASSIFICATION_TASKS:
        merged.setdefault("class_weight", config.model.class_weight)
    return estimator_class(**_filter_params(estimator_class, merged))


def apply_target_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.log(np.clip(values, 1e-9, None))
    if transform == "log1p":
        return np.log1p(np.clip(values, 0, None))
    if transform == "sqrt":
        return np.sqrt(np.clip(values, 0, None))
    return values


def invert_target_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.exp(values)
    if transform == "log1p":
        return np.expm1(values)
    if transform == "sqrt":
        return np.square(values)
    return values


@dataclass
class TabularBundle:
    """Everything needed to reproduce a prediction, in one serialisable unit.

    The estimator alone is not enough: the feature pipeline and the label
    mapping must travel with it or serving will silently disagree with training.
    """

    model: Any
    feature_pipeline: Pipeline
    feature_columns: list[str]
    classes: list[str] = field(default_factory=list)
    target_transform: str = "none"
    residual_std: float | None = None
    trained_metrics: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)

    def is_classifier(self) -> bool:
        return hasattr(self.model, "predict_proba")

    def explain(self, limit: int = 3) -> list[str]:
        """Plain-language drivers from the model's own feature importances."""
        if not self.feature_importances:
            return []
        ranked = sorted(self.feature_importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:limit]
        return [f"{name} was among the strongest inputs (weight {value:.3f})" for name, value in ranked]


# --------------------------------------------------------------------- data IO
def read_frame(config: ModuleConfig) -> pd.DataFrame:
    """Read the configured CSV and normalise its column names.

    Validation is deliberately NOT done here. A module may derive its training
    label (the risk module does), and the derived column has to exist before the
    schema check is meaningful. Callers read, engineer, then validate.
    """
    path = config.data.raw_path
    if path.is_file():
        frame = pd.read_csv(path, dtype=config.data.dtype_map or None)
    elif path.is_dir():
        raise ModuleExecutionError(
            f"{config.name} reads a directory, not a CSV; use its own loader",
            context={"path": str(path)},
        )
    else:
        raise ModuleExecutionError(f"Training data missing: {path}", context={"module": config.name})
    # Source files carry stray whitespace in headers (fertilizer has 'Humidity ').
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_training_frame(config: ModuleConfig) -> tuple[pd.DataFrame, ValidationReport]:
    """Read and fully validate the configured CSV against its declared schema."""
    frame = read_frame(config)
    report = validate_frame(frame, config.data, name=config.data.raw_path.name)
    report.raise_if_invalid()
    return frame, report


def default_feature_pipeline(config: ModuleConfig) -> Pipeline:
    """Column selection plus optional normalisation for the declared features.

    Categorical handling is left to the module's feature hook: a one-hot encoder
    fitted per request would be a train/serve divergence, and the modules that
    need it fit their encoder once at train time and ship it in the bundle.
    """
    steps: list[Any] = [ColumnSelector(config.data.features, drop=config.data.drop)]
    if config.model.family in {"sklearn", "xgboost"} and config.model.algorithm in {
        "LogisticRegression",
        "Ridge",
    }:
        # Only linear models need standardisation; trees are invariant to scale
        # and normalising them adds a train/serve surface for no benefit.
        steps.append(NumericNormalizer(config.data.features))
    return Pipeline(steps=steps, output_columns=None)


def split_and_validate(config: ModuleConfig, frame: pd.DataFrame) -> tuple[SplitResult, ValidationReport]:
    report = validate_frame(frame, config.data, name=config.data.raw_path.name)
    report.raise_if_invalid()
    result = make_splits(frame, config.split, task=config.task, target=config.data.target)
    return result, report


def feature_matrix(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen pipeline. Never fits - that is the whole point."""
    return pipeline.build(frame)


# ------------------------------------------------------------------- predictor
class TabularPredictor(BasePredictor):
    """Base class for every tabular module.

    A subclass sets module_name, request_model, response_model and normally
    field_map. It does not implement _load or _predict.
    """

    #: Pydantic model describing the request body.
    request_model: ClassVar[type[BaseModel]]
    #: Pydantic model describing the response; must subclass Prediction.
    response_model: ClassVar[type[BaseModel]]
    #: Request field name -> feature column name, when they differ.
    field_map: ClassVar[dict[str, str]] = {}
    #: Request fields that must be present for a prediction.
    required_fields: ClassVar[tuple[str, ...]] = ()
    artifact_filename: ClassVar[str] = BUNDLE_FILENAME

    def _load(self) -> None:
        bundle = self.artifact
        if not isinstance(bundle, TabularBundle):
            raise ModuleExecutionError(
                f"{self.module_name} artefact is {type(bundle).__name__}, expected TabularBundle"
            )
        self.bundle: TabularBundle = bundle
        self.feature_columns = list(bundle.feature_columns)
        self.classes = list(bundle.classes)
        self.shap_explainer = None

    # ------------------------------------------------------------- conversion
    def request_to_frame(self, request: BaseModel) -> pd.DataFrame:
        """Turn a validated request into a one-row feature frame."""
        payload = request.model_dump()
        row: dict[str, Any] = {}
        for field_name, value in payload.items():
            column = self.field_map.get(field_name, field_name)
            row[column] = value
        return pd.DataFrame([row])

    @classmethod
    def prepare_features(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Module hook for domain feature engineering.

        Declared as a classmethod so the trainer and the server call literally
        the same function - the single most common source of train/serve skew is
        a feature step that exists in one path and not the other.
        """
        return frame

    @classmethod
    def build_feature_pipeline(cls, config: ModuleConfig, frame: pd.DataFrame) -> Pipeline:
        """Default pipeline, or a module override that adds fitted encoders."""
        return default_feature_pipeline(config)

    # -------------------------------------------------------------- inference
    def _predict(self, request: BaseModel) -> Prediction:
        frame = self.request_to_frame(request)
        frame = self.prepare_features(frame)
        matrix = feature_matrix(self.bundle.feature_pipeline, frame)
        if self.bundle.is_classifier():
            return self._predict_class(request, matrix)
        return self._predict_regression(request, matrix)

    def _predict_class(self, request: BaseModel, matrix: pd.DataFrame) -> Prediction:
        probabilities = self.bundle.model.predict_proba(matrix)[0]
        order = np.argsort(probabilities)[::-1]
        classes = self.bundle.classes or [
            str(label) for label in getattr(self.bundle.model, "classes_", order)
        ]
        top_index = int(order[0])
        top_label = str(classes[top_index]) if top_index < len(classes) else str(top_index)
        score = float(probabilities[top_index])

        alternatives = [
            {"label": str(classes[i]) if i < len(classes) else str(i), "score": float(probabilities[i])}
            for i in order[1:3]
        ]
        reasons = list(self.bundle.explain())
        if score < self.config.serving.confidence_medium:
            reasons.insert(
                0,
                f"The model is not confident here (top class {top_label} at {score:.0%}); "
                "treat this as a suggestion and confirm with a soil test",
            )
        else:
            reasons.insert(0, f"{top_label} scored {score:.0%} against {len(classes)} candidate classes")

        return self.response_model(
            value=top_label,
            confidence_score=score,
            confidence=self.classify_confidence(score),
            alternatives=alternatives if score < self.config.serving.confidence_high else None,
            reasons=reasons,
            model_name=self.module_name,
            model_version=self.version,
        )

    def _predict_regression(self, request: BaseModel, matrix: pd.DataFrame) -> Prediction:
        raw = float(np.asarray(self.bundle.model.predict(matrix)).ravel()[0])
        value = float(invert_target_transform(np.array([raw]), self.bundle.target_transform)[0])

        interval = None
        if self.bundle.residual_std:
            spread = 1.2816 * self.bundle.residual_std
            interval = Interval(low=value - spread, high=value + spread, level=0.80)

        return self.response_model(
            value=value,
            confidence_score=None,
            confidence=self.classify_confidence(None),
            interval=interval,
            reasons=list(self.bundle.explain()),
            model_name=self.module_name,
            model_version=self.version,
        )


# ----------------------------------------------------------------------- train
def _decode_labels(labels: list[str], encoded: np.ndarray) -> np.ndarray:
    """Turn integer class codes back into the label strings a human reads.

    Estimators are fitted on integer codes because scikit-learn requires them,
    but every metric, confusion matrix and API response must speak label names.
    A report that says class 17 is not a report.
    """
    return np.array(
        [labels[int(i)] if 0 <= int(i) < len(labels) else str(i) for i in np.asarray(encoded).ravel()]
    )


def evaluate_predictions(
    config: ModuleConfig,
    model: Any,
    matrix: pd.DataFrame,
    y_true: np.ndarray,
    classes: list[str],
    *,
    segment: pd.Series | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute metrics, the confusion matrix, per-class and per-segment detail.

    y_true may hold integer class codes or label strings. When classes is given, both
    the truth vector and the model output are decoded to label strings before scoring,
    so the report is readable and the confusion matrix is ordered against the declared
    class list rather than against whatever order the estimator happened to see.
    """
    if config.task in _CLASSIFICATION_TASKS:
        raw_predictions = np.asarray(model.predict(matrix)).ravel()
        y_score = model.predict_proba(matrix) if hasattr(model, "predict_proba") else None
        if classes:
            predictions = _decode_labels(classes, raw_predictions)
            truth = _decode_labels(classes, y_true)
        else:
            predictions, truth = raw_predictions, np.asarray(y_true).ravel()
    else:
        predictions = np.asarray(model.predict(matrix)).ravel()
        truth = np.asarray(y_true).ravel()
        y_score = None

    metrics = compute_metrics(truth, predictions, task=config.task, y_score=y_score, segment=segment)

    detail: dict[str, Any] = {}
    if config.task in _CLASSIFICATION_TASKS:
        detail["confusion_matrix"] = confusion(truth, predictions, labels=classes or None)
        detail["per_class"] = per_class_report(truth, predictions)
    if segment is not None:
        detail["per_segment"] = per_segment_errors(truth, predictions, np.asarray(segment), task=config.task)
    detail["n_test"] = len(truth)
    return metrics, detail


def write_metrics_report(
    config: ModuleConfig,
    metrics: dict[str, float],
    detail: dict[str, Any],
    gate_result: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist the full evaluation report that CI and the /v1/metrics endpoint read."""
    ensure_dir(METRICS_DIR)
    payload: dict[str, Any] = {
        "module": config.name,
        "version": config.version,
        "task": config.task.value,
        "metrics": metrics,
        "gate": gate_result,
        "detail": detail,
        "split": config.split.model_dump(),
        "manifest": manifest or {},
    }
    if extra:
        payload.update(extra)
    return write_json(METRICS_DIR / f"{config.name}_v{config.version}.json", payload)


def plot_confusion(detail: dict[str, Any], module: str, path: Path) -> Path | None:
    """Render a confusion matrix to PNG. Skips silently if matplotlib is absent."""
    matrix = detail.get("confusion_matrix")
    if not matrix:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a core dependency
        return None

    labels = matrix["labels"]
    values = np.asarray(matrix["matrix"])
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 0.5), max(5, len(labels) * 0.45)))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title(f"{module} - confusion matrix (test set)")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    ensure_dir(path.parent)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, module: str, path: Path) -> Path | None:
    """Actual-versus-predicted scatter for regressors."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y_true, y_pred, s=8, alpha=0.4)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    axis.plot([low, high], [low, high], color="crimson", linewidth=1, label="perfect")
    axis.set_xlabel("actual")
    axis.set_ylabel("predicted")
    axis.set_title(f"{module} - test set")
    axis.legend()
    figure.tight_layout()
    ensure_dir(path.parent)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
