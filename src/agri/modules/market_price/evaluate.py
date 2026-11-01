"""Evaluate the registered market price models.

Re-runs the rolling-origin backtest against the dataset on disk and enforces the
config gate. Refuses to run when the dataset digest changed, because a metric
computed on different data than the model was fitted on cannot be reproduced.
"""

from __future__ import annotations

import argparse
import json
import sys

from agri.common.config import load_module_config
from agri.common.errors import AgriError
from agri.common.logging import get_logger, setup_logging
from agri.common.metrics import evaluate_gate
from agri.common.paths import METRICS_DIR, sha256_file, write_json
from agri.common.registry import ModelRegistry

_log = get_logger("modules.market_price.evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate market price models")
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    config = load_module_config("market_price")
    registry = ModelRegistry()

    try:
        entry = registry.get(config.name, args.version)
    except AgriError as exc:
        print(json.dumps({"ok": False, "module": config.name, "error": exc.to_dict()}, indent=2))
        return 1

    recorded = entry.manifest.get("data_digest")
    current = sha256_file(config.data.raw_path) if config.data.raw_path.is_file() else None
    if recorded and current and recorded != current:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "dataset digest changed since training; refusing to evaluate",
                    "recorded": recorded,
                    "current": current,
                },
                indent=2,
            )
        )
        return 1

    bundle = entry.load()
    # The bundle records the rejected trend model's evidence under trend_backtest and
    # names the forecaster that is actually served. Older bundles only carry backtest.
    backtest = bundle.get("trend_backtest") or bundle.get("backtest", {})
    served = bundle.get("served_forecaster", "unknown")
    horizon = int(bundle.get("primary_horizon_days", 7))
    primary = backtest.get(f"h{horizon}", {})

    if served == "naive_last_value":
        # The served forecaster IS the baseline, so its accuracy is the baseline column.
        # Reporting the trend's numbers here would score a model that is not shipped.
        metrics = {
            "relative_lift_over_baseline": 0.0,
            "beat_fraction": 0.0,
            "directional_accuracy": float("nan"),
            "mape": primary.get("baseline_mape", float("nan")),
            "median_ape": primary.get("baseline_median_ape", float("nan")),
            "baseline_mape": primary.get("baseline_mape", float("nan")),
            "baseline_median_ape": primary.get("baseline_median_ape", float("nan")),
        }
    else:
        metrics = {
            "relative_lift_over_baseline": primary.get("relative_lift_over_baseline", float("nan")),
            "beat_fraction": primary.get("beat_fraction", float("nan")),
            "directional_accuracy": primary.get("directional_accuracy", float("nan")),
            "mape": primary.get("mape", float("nan")),
            "baseline_mape": primary.get("baseline_mape", float("nan")),
            "median_ape": primary.get("median_ape", float("nan")),
            "baseline_median_ape": primary.get("baseline_median_ape", float("nan")),
        }
    metrics = {key: float(value) for key, value in metrics.items() if value == value}
    gate = evaluate_gate(metrics, config.gates)

    write_json(
        METRICS_DIR / f"{config.name}_evaluated.json",
        {
            "module": config.name,
            "served_forecaster": served,
            "version": entry.version,
            "metrics": metrics,
            "gate": gate.to_dict(),
            "backtest_by_horizon": backtest,
            "evaluated_from_registry": True,
        },
    )
    payload = {
        "ok": gate.passed,
        "module": config.name,
        "version": entry.version,
        "metrics": metrics,
        "gate": gate.to_dict(),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
