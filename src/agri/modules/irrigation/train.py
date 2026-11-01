"""Irrigation has no training step.

It is a deterministic FAO-56 calculation, so "training" would mean fitting a model
to replace a published equation with something less defensible. The command is
kept so the module contract is uniform, and it verifies that the physics engine
reproduces its reference values instead of pretending to learn.
"""

from __future__ import annotations

import sys

import numpy as np

from agri.common.logging import get_logger, setup_logging
from agri.common.paths import METRICS_DIR, write_json

_log = get_logger("modules.irrigation")


def verify() -> int:
    """Check the physics engine against FAO-56 reference values.

    Fails when a reference value moves, which is the only meaningful gate a
    physics module can have.
    """
    from agri.features.physics import (
        atmospheric_pressure,
        compute_raw,
        compute_taw,
        extraterrestrial_radiation,
        saturation_vapour_pressure,
        vapour_pressure_slope,
    )

    checks = {
        "e0(20C)_kPa": (saturation_vapour_pressure(20.0), 2.338),
        "e0(30C)_kPa": (saturation_vapour_pressure(30.0), 4.243),
        "delta(20C)_kPa_per_C": (vapour_pressure_slope(20.0), 0.1440),
        "P(0m)_kPa": (atmospheric_pressure(0.0), 101.3),
        "P(1000m)_kPa": (atmospheric_pressure(1000.0), 89.9),
        "Ra_equator_equinox": (extraterrestrial_radiation(0.0, 81), 37.6),
        "TAW_loam": (compute_taw(0.32, 0.14, 0.6), 108.0),
        "RAW_loam_p0.55": (compute_raw(compute_taw(0.32, 0.14, 0.6), 0.55), 59.4),
    }
    failures = []
    for name, (actual, expected) in checks.items():
        tolerance = max(abs(expected) * 0.01, 0.005)
        if not np.isclose(actual, expected, atol=tolerance):
            failures.append({"check": name, "actual": round(float(actual), 5), "expected": expected})

    payload = {
        "module": "irrigation",
        "kind": "physics_verification",
        "checks": len(checks),
        "failures": failures,
        "passed": not failures,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(METRICS_DIR / "irrigation_physics.json", payload)
    for name, (actual, expected) in checks.items():
        _log.info(
            "physics check", extra={"check": name, "actual": round(float(actual), 5), "expected": expected}
        )
    if failures:
        _log.error("physics verification failed", extra={"failures": failures})
        return 1
    _log.info("physics verification passed", extra={"checks": len(checks)})
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(verify())
