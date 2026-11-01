"""Feature pipeline and encoder contract tests.

What this protects against
--------------------------
Train and serve skew. The encoder and the pipeline are fitted once at training
time and shipped inside the bundle, so serving must assign the same integer code
to the same value and must never silently change the feature vector shape. An
unseen category has to become the sentinel -1 rather than shifting every other
code by one, which is the exact failure mode the integer encoder exists to
prevent.

A missing value must take the same sentinel, never crash the fit and never
silently renumber the vocabulary. The pipeline_hash tests pin the train/serve
skew guard: the digest has to be stable for one pipeline and must change when a
step parameter changes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agri.common.errors import ModuleExecutionError
from agri.features import (
    CategoryIndexEncoder,
    ColumnSelector,
    NumericNormalizer,
    Pipeline,
    pipeline_hash,
)

pytestmark = pytest.mark.unit


def test_category_index_encoder_maps_unseen_categories_to_minus_one() -> None:
    encoder = CategoryIndexEncoder.fit(pd.DataFrame({"crop": ["rice", "maize", "rice"]}), ["crop"])
    assert encoder.vocabularies["crop"] == ["maize", "rice"]

    encoded = encoder.build(pd.DataFrame({"crop": ["rice", "wheat", "maize"]}))
    assert encoded["crop"].tolist() == [1, CategoryIndexEncoder.UNKNOWN, 0]
    assert CategoryIndexEncoder.UNKNOWN == -1


def test_category_index_encoder_is_deterministic() -> None:
    encoder = CategoryIndexEncoder.fit(pd.DataFrame({"crop": ["rice", "maize"]}), ["crop"])
    frame = pd.DataFrame({"crop": ["rice", "wheat", "maize"]})
    assert encoder.build(frame).equals(encoder.build(frame))


def test_category_index_encoder_leaves_an_absent_column_untouched() -> None:
    encoder = CategoryIndexEncoder(["crop"], {"crop": ["rice"]})
    frame = pd.DataFrame({"other": [1, 2]})
    assert encoder.build(frame).equals(frame)


def test_category_index_encoder_ignores_blank_vocabulary_entries() -> None:
    encoder = CategoryIndexEncoder.fit(pd.DataFrame({"crop": ["rice", "", "   "]}), ["crop"])
    assert encoder.vocabularies["crop"] == ["rice"]


def test_category_index_encoder_tolerates_missing_values() -> None:
    encoder = CategoryIndexEncoder.fit(pd.DataFrame({"crop": ["rice", "", float("nan")]}), ["crop"])
    assert encoder.vocabularies["crop"] == ["rice"]

    encoded = encoder.build(pd.DataFrame({"crop": ["rice", "", float("nan"), "wheat"]}))
    assert encoded["crop"].tolist() == [0, CategoryIndexEncoder.UNKNOWN, -1, -1]
    assert encoded["crop"].dtype.kind == "i"


def test_category_index_encoder_keeps_case_distinct_and_unknowns_at_minus_one() -> None:
    frame = pd.DataFrame({"soil": ["Loam", None, "Clay", float("nan"), "loam"]})
    encoder = CategoryIndexEncoder.fit(frame, ["soil"])
    assert encoder.vocabularies["soil"] == ["Clay", "Loam", "loam"]
    assert encoder.build(frame)["soil"].tolist() == [1, -1, 0, -1, 2]


def test_column_selector_reorders_and_drops_declared_columns() -> None:
    built = ColumnSelector(["a", "b", "c"], drop=["c"]).build(pd.DataFrame({"c": [3], "b": [2], "a": [1]}))
    assert list(built.columns) == ["a", "b"]


def test_column_selector_fails_loudly_on_a_missing_feature() -> None:
    with pytest.raises(ModuleExecutionError):
        ColumnSelector(["a", "missing"]).build(pd.DataFrame({"a": [1]}))


def test_numeric_normalizer_standardises_with_population_statistics() -> None:
    built = NumericNormalizer(["a"]).build(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    spread = (2.0 / 3.0) ** 0.5
    assert built["a"].tolist() == pytest.approx([-1.0 / spread, 0.0, 1.0 / spread])


def test_numeric_normalizer_imputes_missing_values_before_scaling() -> None:
    built = NumericNormalizer(["a"]).build(pd.DataFrame({"a": [1.0, float("nan"), 3.0]}))
    spread = (2.0 / 3.0) ** 0.5
    assert not built["a"].isna().any()
    assert built["a"].tolist() == pytest.approx([-1.0 / spread, 0.0, 1.0 / spread])


def test_pipeline_applies_steps_in_order_and_can_select_the_output_columns() -> None:
    pipeline = Pipeline([ColumnSelector(["a", "b"]), NumericNormalizer(["a"])], output_columns=["a"])
    built = pipeline.build(pd.DataFrame({"a": [1.0, 3.0], "b": [5, 6]}))
    assert list(built.columns) == ["a"]
    assert built["a"].tolist() == pytest.approx([-1.0, 1.0])


def test_pipeline_rejects_a_declared_output_column_that_is_missing() -> None:
    pipeline = Pipeline([ColumnSelector(["a"])], output_columns=["a", "nope"])
    with pytest.raises(ModuleExecutionError):
        pipeline.build(pd.DataFrame({"a": [1.0]}))


def test_pipeline_transform_is_an_alias_for_build() -> None:
    pipeline = Pipeline([ColumnSelector(["a"])])
    frame = pd.DataFrame({"a": [1.0]})
    assert pipeline.transform(frame).equals(pipeline.build(frame))


def test_pipeline_parameters_change_when_a_parameter_changes() -> None:
    standard = Pipeline([NumericNormalizer(["a"], strategy="standard")])
    minmax = Pipeline([NumericNormalizer(["a"], strategy="minmax")])
    assert standard.steps[0].to_params() != minmax.steps[0].to_params()


def test_pipeline_hash_changes_when_params_change() -> None:
    sample = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    standard = Pipeline([NumericNormalizer(["a"], strategy="standard")])
    minmax = Pipeline([NumericNormalizer(["a"], strategy="minmax")])
    assert pipeline_hash(standard, sample) != pipeline_hash(minmax, sample)


def test_pipeline_hash_is_stable_for_one_pipeline() -> None:
    sample = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pipeline = Pipeline([NumericNormalizer(["a"])])
    digest = pipeline_hash(pipeline, sample)
    assert digest == pipeline_hash(pipeline, sample)
    assert len(digest) == 64


def test_pipeline_hash_handles_a_pipeline_without_steps() -> None:
    assert len(pipeline_hash(Pipeline([]), pd.DataFrame({"a": [1.0]}))) == 64
