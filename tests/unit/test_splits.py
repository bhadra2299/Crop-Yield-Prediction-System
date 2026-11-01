"""Dataset partitioning tests.

What this protects against
--------------------------
Lookahead leakage and test-set reuse. A time-ordered dataset that receives a
random split produces an offline score that cannot be reproduced in the field,
and a fold list that is not disjoint poisons hyperparameter search. The tests
assert structure only - disjoint partitions, ordered time windows, the declared
fold count - and never assert a model accuracy.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from agri.common.config import SplitSpec
from agri.common.errors import ConfigError, DataValidationError
from agri.common.schemas import TaskType
from agri.common.splits import cross_validation_folds, make_splits

pytestmark = pytest.mark.unit


@pytest.fixture
def split_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    size = 200
    return pd.DataFrame(
        {
            "row_id": range(size),
            "feature": rng.normal(size=size),
            "year": np.repeat(np.arange(2000, 2010), 20),
            "group": np.repeat(np.arange(20), 10),
            "label": np.array(["a", "b"] * (size // 2)),
        }
    )


def row_ids(frame: pd.DataFrame) -> set[int]:
    return set(frame["row_id"].tolist())


def test_stratified_split_is_disjoint_and_keeps_both_classes(split_frame: pd.DataFrame) -> None:
    result = make_splits(
        split_frame,
        SplitSpec(strategy="stratified", test_size=0.25),
        task=TaskType.CLASSIFICATION,
        target="label",
    )
    assert result.sizes == {"train": 150, "test": 50}
    assert row_ids(result.train).isdisjoint(row_ids(result.test))
    assert set(result.train["label"]) == {"a", "b"}
    assert set(result.test["label"]) == {"a", "b"}


def test_random_split_is_disjoint(split_frame: pd.DataFrame) -> None:
    result = make_splits(
        split_frame,
        SplitSpec(strategy="random", test_size=0.25, shuffle=True),
        task=TaskType.CLASSIFICATION,
        target="label",
    )
    assert result.sizes == {"train": 150, "test": 50}
    assert row_ids(result.train).isdisjoint(row_ids(result.test))


def test_time_split_respects_train_end_and_test_start(split_frame: pd.DataFrame) -> None:
    spec = SplitSpec(strategy="time", time_column="year", train_end=2004, test_start=2005)
    result = make_splits(split_frame, spec, task=TaskType.CLASSIFICATION, target="label")

    assert result.train["year"].max() <= 2004
    assert result.test["year"].min() >= 2005
    assert row_ids(result.train).isdisjoint(row_ids(result.test))
    assert len(result.train) > 0
    assert len(result.test) > 0


def test_time_strategy_without_a_time_column_is_rejected_when_the_spec_is_built() -> None:
    with pytest.raises(ValidationError):
        SplitSpec(strategy="time")


def test_time_split_rejects_a_declared_column_that_is_absent(split_frame: pd.DataFrame) -> None:
    spec = SplitSpec(strategy="time", time_column="not_present", train_end=1, test_start=2)
    with pytest.raises(ConfigError):
        make_splits(split_frame, spec, task=TaskType.CLASSIFICATION, target="label")


def test_time_split_holdout_years_keeps_the_recent_period_for_testing(split_frame: pd.DataFrame) -> None:
    spec = SplitSpec(strategy="time", time_column="year", holdout_years=2)
    result = make_splits(split_frame, spec, task=TaskType.CLASSIFICATION, target="label")
    assert result.test["year"].min() > result.train["year"].max()


def test_group_split_is_disjoint(split_frame: pd.DataFrame) -> None:
    # SplitSpec has no group_column field, so the group code path is exercised
    # with the minimal object make_splits actually reads for that strategy.
    spec = SimpleNamespace(strategy="group", test_size=0.25, group_column="group")
    result = make_splits(split_frame, spec, task=TaskType.CLASSIFICATION, target="label")

    assert row_ids(result.train).isdisjoint(row_ids(result.test))
    assert set(result.train["group"]).isdisjoint(set(result.test["group"]))


def test_repeated_kfold_returns_n_splits_times_n_repeats_folds(split_frame: pd.DataFrame) -> None:
    spec = SplitSpec(strategy="repeated_kfold", n_splits=3, n_repeats=2)
    result = make_splits(split_frame, spec, task=TaskType.CLASSIFICATION, target="label")

    assert len(result.folds) == 6
    for train_index, test_index in result.folds:
        assert set(train_index.tolist()).isdisjoint(set(test_index.tolist()))


def test_repeated_kfold_requires_a_stratifiable_target(split_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        make_splits(
            split_frame,
            SplitSpec(strategy="repeated_kfold", n_splits=3),
            task=TaskType.CLASSIFICATION,
            target=None,
        )


def test_strategy_none_returns_the_whole_frame_for_both_partitions(split_frame: pd.DataFrame) -> None:
    result = make_splits(split_frame, SplitSpec(strategy="none"), task=TaskType.PHYSICS, target=None)
    assert result.train.equals(split_frame)
    assert result.test.equals(split_frame)


def test_splitting_an_empty_frame_is_rejected() -> None:
    with pytest.raises(DataValidationError):
        make_splits(
            pd.DataFrame(),
            SplitSpec(strategy="random", test_size=0.5),
            task=TaskType.CLASSIFICATION,
            target=None,
        )


def test_cross_validation_folds_returns_the_declared_number(split_frame: pd.DataFrame) -> None:
    folds = cross_validation_folds(split_frame, SplitSpec(n_splits=5), target="label")
    assert len(folds) == 5
    for train_index, test_index in folds:
        assert set(train_index.tolist()).isdisjoint(set(test_index.tolist()))
