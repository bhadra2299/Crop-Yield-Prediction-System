"""Drift and data-quality monitoring.

Two questions, deliberately kept separate:

1. Has the distribution of served requests moved away from what the model was
   trained on? That is drift, measured with the Population Stability Index.

2. Is the data itself still valid - are the ranges sane, are required fields
   arriving? That is quality, and it is a different failure with a different fix.

A PSI needs a reference distribution. The reference here is the training feature
distribution, reconstructed from reports/metrics feature statistics when present,
or from the dataset itself. When neither is available the check reports
"unknown" rather than inventing a baseline.

The standard PSI reading is used, and stated in the output so nobody has to
remember it: below 0.1 is stable, 0.1 to 0.25 is a moderate shift worth watching,
above 0.25 is a significant shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from agri.common.logging import get_logger
from agri.common.paths import METRICS_DIR, read_json

_log = get_logger("monitoring.drift")

PSI_STABLE = 0.10
PSI_MODERATE = 0.25


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, *, bins: int = 10, epsilon: float = 1e-6
) -> float:
    """PSI between a reference and a current distribution.

    Bin edges come from the reference quantiles so each reference bin holds
    roughly equal mass. Bins with zero mass are floored at epsilon rather than
    skipped, because a category that disappears entirely is exactly the signal
    the index is meant to catch.
    """
    expected = np.asarray(expected, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < bins or len(actual) == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # A near-constant reference has nothing to compare against.
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_share = expected_counts / max(len(expected), 1)
    actual_share = actual_counts / max(len(actual), 1)

    expected_share = np.clip(expected_share, epsilon, None)
    actual_share = np.clip(actual_share, epsilon, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def classify_psi(psi: float) -> str:
    if psi != psi:  # NaN
        return "unknown"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate_shift"
    return "significant_shift"


@dataclass
class DriftReport:
    """Per-feature drift verdict across one module."""

    module: str
    generated_at: str
    reference_size: int
    current_size: int
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        shifted = [name for name, item in self.features.items() if item["verdict"] == "significant_shift"]
        return {
            "module": self.module,
            "generated_at": self.generated_at,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "features": self.features,
            "significantly_shifted": shifted,
            "warnings": self.warnings,
            "reading_guide": "PSI < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.",
        }


def drift_report_for(module: str, current: pd.DataFrame) -> DriftReport:
    """Compare currently served feature values against the reference distribution.

    The reference is the training dataset declared in the module config. That is
    the correct baseline: it is what the model actually learned from.
    """
    generated = datetime.now(UTC).isoformat()
    try:
        from agri.common.config import load_module_config

        config = load_module_config(module)
    except Exception as exc:
        return DriftReport(module, generated, 0, len(current), {}, [f"configuration unreadable: {exc}"])

    report = DriftReport(module=module, generated_at=generated, reference_size=0, current_size=len(current))
    path = config.data.raw_path

    if not path.is_file():
        report.warnings.append(
            f"The reference dataset for {module} is not present at {path.name}, so no drift baseline "
            "could be built. Drift is reported as unknown rather than assumed absent."
        )
        return report

    try:
        reference = pd.read_csv(path, low_memory=False)
    except (OSError, ValueError) as exc:
        report.warnings.append(f"Reference dataset unreadable: {exc}")
        return report

    reference.columns = [str(column).strip() for column in reference.columns]
    report.reference_size = len(reference)
    report.warnings.append(
        "Drift is measured against the training dataset on disk, not against the data the model was "
        "fitted on at the time. If the dataset changed, retrain before trusting this reading."
    )

    for column in config.data.features:
        if column not in reference.columns or column not in current.columns:
            continue
        expected = pd.to_numeric(reference[column], errors="coerce").dropna().to_numpy()
        actual = pd.to_numeric(current[column], errors="coerce").dropna().to_numpy()
        if len(expected) < 10 or len(actual) < 10:
            report.features[column] = {
                "psi": None,
                "verdict": "unknown",
                "reason": "insufficient numeric values on one side of the comparison",
            }
            continue
        psi = population_stability_index(expected, actual)
        report.features[column] = {
            "psi": None if psi != psi else round(psi, 4),
            "verdict": classify_psi(psi),
            "reference_mean": round(float(np.mean(expected)), 4),
            "current_mean": round(float(np.mean(actual)), 4),
            "reference_n": len(expected),
            "current_n": len(actual),
        }

    significant = [name for name, item in report.features.items() if item["verdict"] == "significant_shift"]
    _log.info(
        "drift report generated",
        extra={"module": module, "features": len(report.features), "significant": significant},
    )
    return report


def quality_check(module: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Validate a batch of incoming values against the module's declared schema."""
    from agri.common.config import load_module_config
    from agri.common.validation import validate_frame

    try:
        config = load_module_config(module)
    except Exception as exc:
        return {"module": module, "ok": False, "error": f"configuration unreadable: {exc}"}

    report = validate_frame(frame, config.data, name=f"{module}:incoming")
    return {
        "module": module,
        "ok": report.ok,
        "errors": [issue.to_dict() for issue in report.errors],
        "warnings": [issue.to_dict() for issue in report.warnings],
    }


def metrics_report_for(module: str) -> dict[str, Any] | None:
    """The most recent persisted metrics report for a module, if any."""
    if not METRICS_DIR.is_dir():
        return None
    candidates = sorted(METRICS_DIR.glob(f"{module}_*.json"))
    if not candidates:
        return None
    return read_json(max(candidates, key=lambda path: path.stat().st_mtime))
