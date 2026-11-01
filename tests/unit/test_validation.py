"""Dataset validation tests.

What this protects against
--------------------------
A corrupted or silently replaced dataset reaching training. Validation is the
only check standing between a CSV on disk and a registered model, so a missing
required column and an impossible physical value must each be reported as a
blocking error. The leakage detector must refuse post-harvest columns, which
would make an offline score look excellent while the served prediction is
useless in the field.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.common.errors import DataValidationError
from agri.common.validation import detect_leakage, validate_frame

pytestmark = pytest.mark.unit


def test_a_valid_frame_reports_no_errors(crop_frame: pd.DataFrame, crop_data_spec) -> None:
    report = validate_frame(crop_frame, crop_data_spec, name="crop.csv")
    assert report.ok is True
    assert report.errors == []
    assert report.rows == len(crop_frame)
    assert report.duplicate_rows == 0
    assert report.summary()["errors"] == 0


def test_missing_required_column_is_a_blocking_error(crop_frame: pd.DataFrame, crop_data_spec) -> None:
    report = validate_frame(crop_frame.drop(columns=["rainfall"]), crop_data_spec, name="crop.csv")
    assert report.ok is False
    assert "required_columns" in {issue.check for issue in report.errors}
    with pytest.raises(DataValidationError):
        report.raise_if_invalid()


def test_out_of_range_value_is_a_blocking_error(crop_frame: pd.DataFrame, crop_data_spec) -> None:
    frame = crop_frame.copy()
    frame.loc[0, "ph"] = 20.0
    report = validate_frame(frame, crop_data_spec, name="crop.csv")
    assert report.ok is False
    assert "range_max" in {issue.check for issue in report.errors}


def test_below_range_value_is_a_blocking_error(crop_frame: pd.DataFrame, crop_data_spec) -> None:
    frame = crop_frame.copy()
    frame.loc[0, "temperature"] = -100.0
    report = validate_frame(frame, crop_data_spec, name="crop.csv")
    assert "range_min" in {issue.check for issue in report.errors}


def test_non_finite_value_is_a_blocking_error(crop_frame: pd.DataFrame, crop_data_spec) -> None:
    frame = crop_frame.copy()
    frame.loc[0, "rainfall"] = np.inf
    report = validate_frame(frame, crop_data_spec, name="crop.csv")
    assert "finite_values" in {issue.check for issue in report.errors}


def test_duplicate_rows_are_reported(crop_frame: pd.DataFrame, crop_data_spec) -> None:
    frame = pd.concat([crop_frame, crop_frame.iloc[[0]]], ignore_index=True)
    report = validate_frame(frame, crop_data_spec, name="crop.csv")
    assert report.duplicate_rows == 1
    assert "duplicates" in {issue.check for issue in report.issues}


def test_an_undeclared_class_label_is_a_blocking_error(crop_frame: pd.DataFrame, crop_data_spec) -> None:
    frame = crop_frame.copy()
    frame.loc[0, "label"] = "not_a_declared_crop"
    report = validate_frame(frame, crop_data_spec, name="crop.csv")
    assert "unknown_classes" in {issue.check for issue in report.errors}


def test_detect_leakage_flags_production_and_yield_features(crop_data_spec) -> None:
    spec = crop_data_spec.model_copy(
        update={
            "target": "Yield",
            "features": ["Production", "Yield", "Area"],
            "required_columns": ["Production", "Yield", "Area"],
        }
    )
    frame = pd.DataFrame({"Production": [1.0], "Yield": [2.0], "Area": [3.0]})
    issues = detect_leakage(frame, spec)
    assert {issue.context["column"] for issue in issues} == {"Production", "Yield"}
    assert all(issue.severity == "error" for issue in issues)


def test_detect_leakage_accepts_a_clean_feature_set(crop_frame: pd.DataFrame, crop_data_spec) -> None:
    assert detect_leakage(crop_frame, crop_data_spec) == []
