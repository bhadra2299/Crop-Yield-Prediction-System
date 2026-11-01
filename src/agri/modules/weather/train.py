"""Weather has no training step.

The module serves provider data and a published equation, so there is nothing to
fit. This command verifies configuration and documents the absence explicitly
rather than silently succeeding at nothing.
"""

from __future__ import annotations

import json
import sys

from agri.common.config import load_module_config
from agri.common.logging import get_logger, setup_logging
from agri.common.settings_helpers import openweather_key_status

_log = get_logger("modules.weather")


def main() -> int:
    config = load_module_config("weather")
    status = openweather_key_status()
    payload = {
        "ok": True,
        "module": "weather",
        "kind": "no_training_required",
        "reason": "Weather serves provider data and the FAO-56 ET0 equation; there is nothing to fit.",
        "openweather_key_configured": status,
        "forecast_days": config.model.params.get("forecast_days", 7),
    }
    if not status:
        payload["warning"] = (
            "OPENWEATHER_API_KEY is not set. The module loads and reports its own unreadiness, "
            "but live forecasts will fail with a configuration error until it is supplied."
        )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
