"""Dataset schema validation.

Runs on every load, in training and in CI. The goal is that a silently corrupted
or silently replaced dataset fails loudly and early, rather than producing a
model that looks fine and is wrong.

Checks performed
----------------
* file exists and is non-empty
* every declared required column is present
* row count meets the module minimum
* missing-value fraction per column is within budget
* numeric columns are finite (no inf, no silent text coercion)
* declared value ranges hold
* duplicate rows are reported
* class labels match the configured set and every class has enough members
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agri.common.config import DataSpec
from agri.common.errors import DataValidationError
from agri.common.logging import get_logger

_log = get_logger("validation")


class ValidationIssue(BaseModel):
    """One problem found in a dataset. 'error' blocks training, 'warning' does not."""

    severity: str = Field(..., pattern="^(error|warning)$")
    check: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Result of validating one dataset."""

    dataset: str
    rows: int
    columns: int
    issues: list[ValidationIssue] = Field(default_factory=list)
    duplicate_rows: int = 0
    memory_mb: float = 0.0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise DataValidationError(
                f"Dataset '{self.dataset}' failed validation with {len(self.errors)} error(s)",
                context={"issues": [issue.model_dump() for issue in self.errors]},
            )

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "columns": self.columns,
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": round(self.memory_mb, 2),
        }


@dataclass
class RangeRule:
    """Declared physical bound for a column. Catches impossible values."""

    column: str
    minimum: float | None = None
    maximum: float | None = None
    allow_null: bool = True
    note: str = ""


#: Reusable physical ranges for the agriculture domain. Modules extend these.
COMMON_RANGES: dict[str, RangeRule] = {
    "N": RangeRule("N", 0, 200, note="kg/ha soil nitrogen"),
    "P": RangeRule("P", 0, 200, note="kg/ha soil phosphorus"),
    "K": RangeRule("K", 0, 250, note="kg/ha soil potassium"),
    "ph": RangeRule("ph", 0, 14, note="soil pH is dimensionless"),
    "temperature": RangeRule("temperature", -30, 60, note="degrees C"),
    "humidity": RangeRule("humidity", 0, 100, note="percent"),
    "rainfall": RangeRule("rainfall", 0, 6000, note="mm"),
    "moisture": RangeRule("moisture", 0, 5000, note="raw sensor units"),
    "Area": RangeRule("Area", 0, None, note="hectares"),
    "Production": RangeRule("Production", 0, None, note="tonnes"),
    "Yield": RangeRule("Yield", 0, None, note="tonnes per hectare"),
    "Annual_Rainfall": RangeRule("Annual_Rainfall", 0, 12000, note="mm per year"),
    "Fertilizer": RangeRule("Fertilizer", 0, None, note="kg"),
    "Pesticide": RangeRule("Pesticide", 0, None, note="kg"),
    "Crop_Year": RangeRule("Crop_Year", 1960, 2100, note="cropping year"),
}


def validate_frame(
    frame: pd.DataFrame,
    spec: DataSpec,
    *,
    name: str,
    ranges: dict[str, RangeRule] | None = None,
    min_class_members: int = 2,
    strict: bool = False,
) -> ValidationReport:
    """Validate a loaded dataframe against its declared spec.

    strict=True turns warnings into errors; CI uses it for release branches.
    """
    issues: list[ValidationIssue] = []
    report = ValidationReport(
        dataset=name,
        rows=len(frame),
        columns=frame.shape[1],
        memory_mb=frame.memory_usage(deep=True).sum() / 1e6,
    )

    # ---------------------------------------------------------- shape and columns
    if frame.empty:
        issues.append(ValidationIssue(severity="error", check="non_empty", message="Dataset has zero rows"))
    if len(frame) < spec.min_rows:
        issues.append(
            ValidationIssue(
                severity="error",
                check="min_rows",
                message=f"Dataset has {len(frame)} rows, minimum is {spec.min_rows}",
            )
        )

    expected = set(spec.required_columns or []) | set(spec.features) | {spec.target} - {None}
    missing = sorted(column for column in expected if column not in frame.columns)
    if missing:
        issues.append(
            ValidationIssue(
                severity="error",
                check="required_columns",
                message=f"Missing required column(s): {', '.join(missing)}",
                context={"missing": missing, "present": list(frame.columns)[:50]},
            )
        )

    # ------------------------------------------------------------ missing values
    if not frame.empty:
        missing_fraction = frame.isna().mean()
        budget = spec.max_missing_fraction
        for column, fraction in missing_fraction.items():
            if fraction <= 0:
                continue
            if column in spec.allow_missing or column in spec.drop or column in spec.id_columns:
                continue
            severity = "error" if fraction > max(budget, 1e-9) else "warning"
            issues.append(
                ValidationIssue(
                    severity=severity,
                    check="missing_values",
                    message=f"Column '{column}' is {fraction:.2%} null (budget {budget:.0%})",
                    context={"column": str(column), "null_fraction": float(fraction)},
                )
            )

    # ----------------------------------------------------------- numeric sanity
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].to_numpy(dtype="float64", na_value=np.nan)
        finite = np.isfinite(values)
        invalid = int((~finite & ~np.isnan(values)).sum())
        if invalid:
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="finite_values",
                    message=f"Column '{column}' contains {invalid} non-finite value(s)",
                    context={"column": str(column), "count": invalid},
                )
            )

    # ------------------------------------------------------------------ ranges
    rules = {**(ranges or {}), **COMMON_RANGES}
    for column, values in frame.items():
        rule = rules.get(str(column))
        if rule is None or not pd.api.types.is_numeric_dtype(values):
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if rule.minimum is not None:
            violations = int((numeric < rule.minimum).sum())
            if violations:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="range_min",
                        message=f"Column '{column}' has {violations} value(s) below {rule.minimum} ({rule.note})",
                        context={"column": str(column), "count": violations, "min": rule.minimum},
                    )
                )
        if rule.maximum is not None:
            violations = int((numeric > rule.maximum).sum())
            if violations:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="range_max",
                        message=f"Column '{column}' has {violations} value(s) above {rule.maximum} ({rule.note})",
                        context={"column": str(column), "count": violations, "max": rule.maximum},
                    )
                )
        if not rule.allow_null and numeric.isna().any():
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="null_not_allowed",
                    message=f"Column '{column}' contains nulls but nulls are not permitted",
                )
            )

    # -------------------------------------------------------------- duplicates
    report.duplicate_rows = int(frame.duplicated().sum())
    if report.duplicate_rows:
        issues.append(
            ValidationIssue(
                severity="warning" if report.duplicate_rows < len(frame) * 0.05 else "error",
                check="duplicates",
                message=f"{report.duplicate_rows} fully duplicated row(s) ({report.duplicate_rows / max(len(frame), 1):.2%})",
                context={"duplicates": report.duplicate_rows},
            )
        )

    # ------------------------------------------------------------ class labels
    if spec.target and spec.classes and spec.target in frame.columns:
        observed = set(frame[spec.target].astype(str).unique())
        unexpected = sorted(observed - set(spec.classes))
        absent = sorted(set(spec.classes) - observed)
        if unexpected:
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="unknown_classes",
                    message=f"Target contains undeclared classes: {', '.join(unexpected)}",
                    context={"unexpected": unexpected},
                )
            )
        if absent:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    check="absent_classes",
                    message=f"Target is missing declared classes: {', '.join(absent)}",
                    context={"absent": absent},
                )
            )
        counts = frame[spec.target].value_counts()
        thin = counts[counts < min_class_members]
        if not thin.empty:
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="class_support",
                    message=f"{len(thin)} class(es) have fewer than {min_class_members} samples",
                    context={"classes": {str(k): int(v) for k, v in thin.items()}},
                )
            )

    report.issues = issues
    if strict:
        for issue in report.issues:
            if issue.severity == "warning":
                issue.severity = "error"

    log = _log.info if report.ok else _log.warning
    log("validated dataset", extra=report.summary())
    return report


def validate_file(
    path: str | Path, spec: DataSpec, *, name: str | None = None, strict: bool = False
) -> ValidationReport:
    """Load a CSV and validate it. Used by 'agri data validate' and by tests."""
    resolved = Path(path)
    if not resolved.is_file():
        raise DataValidationError(f"Dataset file not found: {resolved}")
    if resolved.stat().st_size == 0:
        raise DataValidationError(f"Dataset file is empty: {resolved}")
    frame = pd.read_csv(resolved)
    return validate_frame(frame, spec, name=name or resolved.name, strict=strict)


def detect_leakage(frame: pd.DataFrame, spec: DataSpec) -> list[ValidationIssue]:
    """Flag columns that cannot be known at prediction time.

    Production and Production-derived columns are the classic yield leakage: they
    are recorded after harvest, so a model using them scores well offline and is
    useless in the field.
    """
    issues: list[ValidationIssue] = []
    suspicious = {"Production", "Yield", "yield", "production"}
    for column in frame.columns:
        if column in spec.features and column in suspicious:
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="target_leakage",
                    message=f"Feature '{column}' is recorded after harvest and leaks the target",
                    context={"column": str(column)},
                )
            )
    return issues


@dataclass
class SchemaContract:
    """Minimal structural contract a module can assert in a regression test."""

    columns: list[str]
    dtypes: dict[str, str] = field(default_factory=dict)

    def matches(self, frame: pd.DataFrame) -> tuple[bool, list[str]]:
        problems: list[str] = []
        missing = sorted(set(self.columns) - set(frame.columns))
        if missing:
            problems.append(f"missing columns: {missing}")
        for column, expected in self.dtypes.items():
            if column in frame.columns and str(frame[column].dtype) != expected:
                problems.append(f"{column}: expected {expected}, got {frame[column].dtype}")
        return not problems, problems
