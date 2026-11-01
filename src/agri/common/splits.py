"""Dataset partitioning.

Two rules are enforced here rather than left to each module:

1. A time-ordered dataset is never randomly split. For yield and market price
   the test set is strictly later in time than the training set, which is the
   only way to avoid lookahead leakage.
2. The test partition is materialised once and never touched during model
   selection. Hyperparameter search runs on train/validation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from agri.common.config import SplitSpec
from agri.common.errors import ConfigError, DataValidationError
from agri.common.logging import get_logger
from agri.common.schemas import TaskType

_log = get_logger("splits")


@dataclass
class SplitResult:
    """Materialised partitions plus the cross-validation folds when requested."""

    train: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame | None = None
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None
    strategy: str = "stratified"

    @property
    def sizes(self) -> dict[str, int]:
        out = {"train": len(self.train), "test": len(self.test)}
        if self.validation is not None:
            out["validation"] = len(self.validation)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "sizes": self.sizes,
            "n_folds": len(self.folds) if self.folds else 0,
        }


def _stratify_vector(frame: pd.DataFrame, target: str | None) -> pd.Series | None:
    """Stratify only when every class has enough members for the requested split."""
    if target is None or target not in frame.columns:
        return None
    counts = frame[target].value_counts()
    return frame[target] if len(counts) > 1 and counts.min() >= 2 else None


def make_splits(
    frame: pd.DataFrame,
    spec: SplitSpec,
    *,
    task: TaskType,
    target: str | None = None,
) -> SplitResult:
    """Partition *frame* according to *spec*.

    Time-aware strategies sort by the configured time column and cut on value,
    so the test window contains only periods after the training window.
    """
    if frame.empty:
        raise DataValidationError("Cannot split an empty dataset")
    if spec.strategy == "none":
        return SplitResult(train=frame.copy(), test=frame.copy(), strategy="none")

    if spec.strategy == "time":
        return _time_split(frame, spec)
    if spec.strategy == "group":
        return _group_split(frame, spec, target)
    if spec.strategy == "repeated_kfold":
        return _repeated_kfold(frame, spec, target)
    return _random_split(frame, spec, task, target)


def _random_split(frame: pd.DataFrame, spec: SplitSpec, task: TaskType, target: str | None) -> SplitResult:
    stratify = _stratify_vector(frame, target) if spec.strategy == "stratified" else None
    if spec.strategy == "stratified" and stratify is None:
        _log.warning("stratification requested but not possible; falling back to a random split")

    holdout = spec.val_size + spec.test_size
    train_part, rest = train_test_split(
        frame, test_size=holdout, random_state=42, shuffle=spec.shuffle, stratify=stratify
    )

    validation = None
    if spec.val_size > 0:
        relative_val = spec.val_size / holdout
        rest_stratify = _stratify_vector(rest, target) if stratify is not None else None
        validation, test = train_test_split(
            rest, test_size=1 - relative_val, random_state=42, stratify=rest_stratify
        )
    else:
        test = rest
    return SplitResult(train=train_part, test=test, validation=validation, strategy=spec.strategy)


def _time_split(frame: pd.DataFrame, spec: SplitSpec) -> SplitResult:
    column = spec.time_column or ""
    if column not in frame.columns:
        raise ConfigError(f"time split requires column '{column}'", context={"columns": list(frame.columns)})

    ordered = frame.sort_values(column).reset_index(drop=True)
    series = ordered[column]
    numeric = pd.to_numeric(series, errors="coerce")

    if spec.train_end is not None and spec.test_start is not None:
        train_end, test_start = float(spec.train_end), float(spec.test_start)
        train = ordered[numeric <= train_end]
        test = ordered[numeric >= test_start]
    elif spec.holdout_years is not None:
        cutoff = float(numeric.max()) - spec.holdout_years
        train = ordered[numeric <= cutoff]
        test = ordered[numeric > cutoff]
    else:
        # Fall back to the configured fraction, computed on row count, not at random.
        cut = int(len(ordered) * (1.0 - spec.test_size))
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]

    if train.empty or test.empty:
        raise DataValidationError(
            "Time split produced an empty partition",
            context={"train_rows": len(train), "test_rows": len(test), "column": column},
        )
    _log.info(
        "time split",
        extra={
            "column": column,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_max": str(train[column].max()),
            "test_min": str(test[column].min()),
        },
    )
    return SplitResult(train=train, test=test, strategy="time")


def _group_split(frame: pd.DataFrame, spec: SplitSpec, target: str | None) -> SplitResult:
    column = spec.group_column if hasattr(spec, "group_column") else None
    if not column or column not in frame.columns:
        raise ConfigError("group split requires a group column present in the data")
    splitter = GroupShuffleSplit(n_splits=1, test_size=spec.test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(frame, groups=frame[column]))
    return SplitResult(
        train=frame.iloc[train_idx].reset_index(drop=True),
        test=frame.iloc[test_idx].reset_index(drop=True),
        strategy="group",
    )


def _repeated_kfold(frame: pd.DataFrame, spec: SplitSpec, target: str | None) -> SplitResult:
    stratify = _stratify_vector(frame, target)
    if stratify is None:
        raise DataValidationError("repeated_kfold requires a target with at least two members per class")
    splitter = RepeatedStratifiedKFold(n_splits=spec.n_splits, n_repeats=spec.n_repeats, random_state=42)
    folds = list(splitter.split(frame, frame[target]))
    train_idx, test_idx = folds[0]
    return SplitResult(
        train=frame.iloc[train_idx].reset_index(drop=True),
        test=frame.iloc[test_idx].reset_index(drop=True),
        folds=folds,
        strategy="repeated_kfold",
    )


def cross_validation_folds(
    frame: pd.DataFrame, spec: SplitSpec, *, target: str | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds for hyperparameter search. Stratified whenever the target allows it."""
    n_splits = min(spec.n_splits, max(2, len(frame) // 2))
    stratify = _stratify_vector(frame, target)
    if stratify is not None:
        splitter: Any = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(frame, frame[target]))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kfold.split(frame))
