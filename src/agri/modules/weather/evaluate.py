"""Weather evaluation.

The only meaningful automated check is that the ingestion boundary converts
provider units correctly and that the module degrades instead of crashing. Live
provider checks are marked network-dependent and are not part of the offline gate.
"""

from __future__ import annotations

import json
import sys

from agri.common.logging import setup_logging
from agri.features.physics import DailyWeather, compute_et0


def check_et0_range() -> dict[str, float]:
    """ET0 for a hot dry day must land in a physically plausible band.

    A value outside 0-15 mm/day means a unit conversion is wrong - most often
    Kelvin leaking through as Celsius, which inflates ET0 by an order of magnitude.
    """
    hot_dry = DailyWeather(t_max=42.0, t_min=28.0, humidity_mean=20.0, wind_speed_2m=3.0, elevation=200.0)
    mild = DailyWeather(t_max=26.0, t_min=16.0, humidity_mean=70.0, wind_speed_2m=2.0, elevation=200.0)
    return {
        "et0_hot_dry_mm": round(compute_et0(hot_dry, 17.385, 120), 3),
        "et0_mild_mm": round(compute_et0(mild, 17.385, 120), 3),
    }


def main() -> int:
    setup_logging()
    values = check_et0_range()
    plausible = (
        0.0 < values["et0_hot_dry_mm"] < 15.0
        and 0.0 < values["et0_mild_mm"] < 15.0
        and values["et0_hot_dry_mm"] > values["et0_mild_mm"]
    )
    payload = {
        "ok": plausible,
        "module": "weather",
        "checks": values,
        "note": (
            "Offline gate checks that ET0 is physically plausible and that a hot dry day drives more "
            "demand than a mild humid one. Live provider checks require network access and are not "
            "part of this gate."
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0 if plausible else 1


if __name__ == "__main__":
    sys.exit(main())
