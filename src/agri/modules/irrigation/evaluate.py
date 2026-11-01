"""Irrigation evaluation.

Because there is no field ground truth, evaluate reports physical consistency:
the FAO-56 reference checks plus the water-balance closure identity over a
synthetic season. It deliberately does not report an accuracy percentage.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from agri.common.logging import setup_logging
from agri.features.physics import DailyWeather, SoilProfile, compute_water_balance
from agri.modules.irrigation.train import verify


def closure_check() -> dict[str, float]:
    """Run a season-long balance and report the closure residual.

    Conservation of mass is a property the implementation must satisfy exactly;
    any non-zero residual is a bug in the accounting, not a modelling error.
    """
    soil = SoilProfile(field_capacity=0.32, wilting_point=0.14, root_depth=0.6, depletion_fraction=0.55)
    series = [
        DailyWeather(t_max=33.0 + (i % 5), t_min=23.0, humidity_mean=50.0, rainfall=0.0, wind_speed_2m=2.0)
        for i in range(60)
    ]
    result = compute_water_balance(series, soil, latitude_deg=17.385, start_date=date(2025, 4, 1))
    return {
        "closure_error_mm": result.closure_error_mm,
        "total_etc": result.total_etc,
        "stress_days": result.stress_days,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Evaluate the irrigation physics engine").parse_args(argv)
    setup_logging()
    physics_ok = verify() == 0
    closure = closure_check()
    payload = {
        "ok": physics_ok and abs(closure["closure_error_mm"]) < 1e-6,
        "module": "irrigation",
        "physics_verified": physics_ok,
        "water_balance": closure,
        "note": (
            "Irrigation has no field ground truth, so this gate reports physical consistency "
            "rather than an accuracy percentage. No accuracy claim should be made for this module."
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
