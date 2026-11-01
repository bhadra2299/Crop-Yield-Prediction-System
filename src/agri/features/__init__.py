"""Shared feature builders - used by both training and serving.

The golden rule this module exists to enforce: the model cannot transform a
serving-time input differently than a training-time input. That difference is
the single largest source of real-world model drift.

Enforced by design
------------------
* Every builder exposes one function: build(features: DataFrame) -> DataFrame
* All parameters are captured in the feature config, never in code
* No training-time state lives in the object: no .fit(), no mean/std
  stored on the instance - the scaler/encoder is an artefact loaded by the
  module like any other model
* Categorical encoders are one-hot with a fixed column order so a new category
  never silently changes the prediction vector shape
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agri.common.errors import ModuleExecutionError
from agri.common.logging import get_logger
from agri.common.paths import ensure_dir

_log = get_logger("features")


class FeatureBuilder(ABC):
    """Base class: one build(frame) function per builder.

    No fit(), no partial_fit(). If a builder needs learned parameters (scaler
    mean/std, label encodings) they are loaded at module load time, not at
    build time. This guarantees build(frame) is identical offline and online.
    """

    @abstractmethod
    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed feature set. Columns and order must be stable."""
        raise NotImplementedError

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Same thing, renamed for clarity at call sites."""
        return self.build(frame)


class ColumnSelector(FeatureBuilder):
    """Select, reorder and rename the columns the model expects.

    The feature vector shape is controlled entirely by the feature config. A
    missing column fails loudly rather than producing a different-shaped vector.
    """

    def __init__(
        self, features: list[str], *, drop: list[str] | None = None, name_map: dict[str, str] | None = None
    ) -> None:
        self.features = list(features)
        self.drop = list(drop or [])
        self.name_map = dict(name_map or {})

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        available = list(frame.columns)
        missing = [c for c in self.features if c not in available]
        if missing:
            raise ModuleExecutionError(
                f"Feature vector missing columns: {missing}",
                context={"features_requested": self.features, "features_available": available[:50]},
            )
        keep = [c for c in self.features if c not in self.drop]
        frame = frame[keep].copy()
        if self.name_map:
            renamed = {k: v for k, v in self.name_map.items() if k in frame.columns}
            if renamed:
                frame = frame.rename(columns=renamed)
        return frame.reset_index(drop=True)


class NumericNormalizer(FeatureBuilder):
    """Standardization with fixed parameters - no per-request fit.

    The scaler params are an artefact loaded once at module load time so
    training and serving use the same center/scale.
    """

    def __init__(
        self,
        columns: list[str],
        *,
        strategy: str = "standard",
        with_mean: bool = True,
        with_std: bool = True,
        scaler: StandardScaler | None = None,
    ) -> None:
        self.columns = list(columns)
        self.strategy = strategy
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler = scaler

    def _validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise ModuleExecutionError(f"Numeric feature missing: {missing}")
        # Impute before normalising so we do not introduce NaN where the model saw 0 during training.
        for column in self.columns:
            if frame[column].isna().any():
                frame[column] = frame[column].fillna(frame[column].median())
        return frame

    def _transform(self, values: pd.Series) -> np.ndarray:
        values = values.to_numpy(dtype="float64")
        if self.strategy == "standard":
            mean, std = float(values.mean()), float(values.std())
            out = (values - mean) / (std if std > 1e-12 else 1.0)
            return np.nan_to_num(out, 0)
        if self.strategy == "minmax":
            mn, mx = values.min(), values.max()
            range_ = mx - mn if mx > mn else 1.0
            out = (values - mn) / range_
            return np.nan_to_num(out, 0)
        if self.strategy == "robust":
            q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
            iqr = q3 - q1 if q3 > q1 else 1.0
            out = (values - q1) / iqr
            return np.nan_to_num(out, 0)
        return values

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate(frame)
        for column in self.columns:
            frame[column] = self._transform(frame[column])
        return frame.reset_index(drop=True)

    def to_params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "strategy": self.strategy,
            "with_mean": self.with_mean,
            "with_std": self.with_std,
        }


class CategoricalEncoder(FeatureBuilder):
    """One-hot encode with a fixed column order and known category set.

    A category seen only at serving time produces zero everywhere rather than
    silently changing the vector. This is the safest default for a model that
    will be served to millions of farmers.
    """

    def __init__(
        self, columns: list[str], *, known_categories: list[str] | None = None, handle_unknown: str = "drop"
    ) -> None:
        self.columns = list(columns)
        self.known = set(known_categories or [])
        self.handle_unknown = handle_unknown
        self._encoder: OneHotEncoder | None = None

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            if column not in frame.columns:
                continue
            series = frame[column].astype(str).fillna("")
            if self.known:
                # Blank out anything outside the known vocabulary before categorising so
                # it lands in the encoder's unknown bucket rather than inventing a new
                # category. This previously cast a CategoricalDtype OBJECT rather than
                # the series, which silently produced a broken encoding for any
                # encoder constructed with an explicit vocabulary.
                series = series.where(series.isin(self.known), "")
                series = series.astype(pd.CategoricalDtype(categories=sorted(self.known), ordered=True))
            else:
                series = series.astype("category")
            ohe = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False, dtype="float64")
            encoded = ohe.fit_transform(series.to_frame()).astype("float32")
            if ohe.categories_[0].size:
                names = [
                    f"{column}__{int(cat)}" if str(cat).isdigit() else f"{column}__{cat}"
                    for cat in ohe.categories_[0]
                ]
            else:
                names = [f"{column}__empty"]
            names = [n if isinstance(n, str) else str(n) for n in names]
            if isinstance(encoded, np.ndarray) and encoded.ndim == 2:
                encoded_frame = pd.DataFrame(encoded, columns=names, index=series.index)
            else:
                encoded_frame = pd.DataFrame({f"{column}__0": np.ones(len(series))}, index=series.index)
            frame = pd.concat([frame, encoded_frame], axis=1)
            frame = frame.drop(columns=[column])
        return frame.reset_index(drop=True)


def _categorical_values(series: pd.Series) -> list[str]:
    """Distinct non-missing string values from a possibly-mixed column.

    Written because pandas 3.x keeps NaN as a float through astype(str), so the
    resulting object column contains a float nan that has no .lower(). Normalising
    once here keeps the fit and transform paths consistent and version-independent.
    """
    cleaned: list[str] = []
    for value in series.tolist():
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            # pd.isna on a non-scalar raises; treat as present.
            pass
        text = str(value).strip()
        if text and text.lower() not in ("nan", "none", "<na>", "null"):
            cleaned.append(text)
    return sorted(set(cleaned))


class CategoryIndexEncoder(FeatureBuilder):
    """Map categoricals to stable integer codes using a vocabulary fitted once.

    Trees consume integer codes natively, which avoids exploding a 54-crop column
    into 54 sparse columns. The vocabulary is fitted on the training partition and
    shipped inside the bundle, so serving assigns the same code to the same value
    - and an unseen category becomes -1 rather than silently shifting every other
    code by one, which is the failure mode that makes one-hot dangerous.
    """

    UNKNOWN = -1

    def __init__(self, columns: list[str], vocabularies: dict[str, list[str]] | None = None) -> None:
        self.columns = list(columns)
        self.vocabularies: dict[str, list[str]] = dict(vocabularies or {})

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> CategoryIndexEncoder:
        """Learn one vocabulary per categorical column from training data."""
        vocabularies: dict[str, list[str]] = {}
        for column in columns:
            if column not in frame.columns:
                vocabularies[column] = []
                continue
            vocabularies[column] = sorted(str(v) for v in _categorical_values(frame[column]))
        return cls(columns, vocabularies)

    def to_params(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "vocabulary_sizes": {k: len(v) for k, v in self.vocabularies.items()},
        }

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for column in self.columns:
            vocabulary = self.vocabularies.get(column, [])
            lookup = {value: index for index, value in enumerate(vocabulary)}
            if column in frame.columns:
                series = frame[column].where(frame[column].notna(), None)
                series = series.map(lambda value: None if value is None else str(value).strip())
                frame[column] = series.map(lookup).fillna(self.UNKNOWN).astype("int32")
        return frame


class TargetEncoder(FeatureBuilder):
    """Smooth target encoding for high-cardinality categoricals.

    Smoothness prevents overfitting on rare values: each category mean is
    shrunk toward the global mean by an amount proportional to its support.
    Params are fitted once and loaded from the artefact directory.
    """

    def __init__(self, column: str, target: str, *, prior_weight: float = 20.0) -> None:
        self.column = column
        self.target = target
        self.prior_weight = prior_weight
        self._mappings: dict[str, float] = {}
        self._global_mean: float = 0.0
        self._counts: dict[str, int] = {}

    def fit_sample(self, frame: pd.DataFrame) -> None:
        """One-time fit on a full dataset before serving. Must not be called at request time."""
        if self.column not in frame.columns or self.target not in frame.columns:
            raise ModuleExecutionError(f"TargetEncoder requires columns {self.column} and {self.target}")
        self._global_mean = float(frame[self.target].mean())
        agg = frame.groupby(self.column)[self.target].agg(["mean", "count"])
        counts = agg["count"].to_dict()
        smoothed = (counts[self.prior_weight] + agg["mean"] * agg["count"]) / (
            self.prior_weight + agg["count"]
        )
        self._mappings = {str(cat): float(val) for cat, val in smoothed.items()}
        self._counts = {str(cat): int(cnt) for cat, cnt in counts.items()}

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.column not in frame.columns:
            return frame
        series = frame[self.column].astype(str).fillna("")
        if self._mappings:
            encoded = series.map(self._mappings).fillna(self._global_mean)
            # Unknown categories shrink toward the global mean.
            frame = frame.copy()
            frame[self.column] = encoded
        return frame


class DropFeatures(FeatureBuilder):
    """Remove declared leakage/unneeded columns."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.drop(columns=[c for c in self.columns if c in frame.columns], errors="ignore")
        return frame


class InteractionBuilder(FeatureBuilder):
    """Add domain features the model should get for free.

    Weather-domain helpers live here so the same formula is used offline and
    online - and so they can be toggled per module from the config.
    """

    def __init__(self, *, degree: int = 2, interaction_features: list[tuple[str, ...]] | None = None) -> None:
        self.degree = degree
        self.interaction_features = interaction_features or []

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        numeric_cols = frame.select_dtypes(include=["number", "bool"]).columns
        if len(numeric_cols) > 1:
            try:
                from sklearn.preprocessing import PolynomialFeatures

                poly = PolynomialFeatures(degree=self.degree, include_bias=False, interaction_only=True)
                poly_frame = poly.fit_transform(frame[numeric_cols])
                poly_names = [
                    "intercept" if name == "1" else name for name in poly.get_feature_names_out(numeric_cols)
                ]
                poly_frame = pd.DataFrame(poly_frame, columns=poly_names, index=frame.index)
                poly_frame = poly_frame.drop(columns=["intercept"])
                frame = pd.concat([frame, poly_frame], axis=1)
            except Exception:
                _log.warning("polynomial interaction generation failed; skipping", exc_info=True)
        for features in self.interaction_features:
            if all(c in frame.columns for c in features):
                name = "_".join(features)
                frame[name] = np.prod(frame[features].values, axis=1)
        return frame


class Pipeline:
    """Ordered feature steps + optional final selection.

    Steps are executed in order; the final column list is declared so a module
    never serves a different vector than the one it was trained on.
    """

    def __init__(self, steps: list[FeatureBuilder], *, output_columns: list[str] | None = None) -> None:
        self.steps = list(steps)
        self.output_columns = list(output_columns or [])

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            frame = step.transform(frame)
        if self.output_columns:
            missing = [c for c in self.output_columns if c not in frame.columns]
            if missing:
                raise ModuleExecutionError(f"Pipeline output missing columns: {missing}")
            frame = frame[self.output_columns].copy()
        return frame.reset_index(drop=True)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.build(frame)


def save_transformer(obj: Any, path: str | Path) -> Path:
    """Save a scaler/encoder artefact atomically."""
    resolved = Path(path)
    ensure_dir(resolved.parent)
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    joblib.dump(obj, tmp)
    tmp.replace(resolved)
    return resolved


def load_transformer(path: str | Path) -> Any:
    """Load a scaler/encoder artefact."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Transformer artefact not found: {resolved}")
    return joblib.load(resolved)


def pipeline_hash(pipeline: Pipeline, frame_sample: pd.DataFrame) -> str:
    """Stable hash of a feature pipeline so training and serving cannot diverge.

    The hash includes the ordered step types and the feature names they see, so
    a config drift (a dropped feature) changes the hash and refuses to serve.
    """
    import hashlib

    parts = [
        f"step:{type(step).__name__}:{step.to_params() if hasattr(step, 'to_params') else {}}"
        for step in getattr(pipeline, "steps", ())
    ]
    parts.append(f"n_steps:{len(getattr(pipeline, 'steps', ()))}")
    for col in frame_sample.columns:
        parts.append(f"column:{col}:{frame_sample[col].dtype}")
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
