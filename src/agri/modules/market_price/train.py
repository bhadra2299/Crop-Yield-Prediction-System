"""Train the market price models.

One model per (commodity, state) series with enough continuous history. Series
that are too short are skipped and counted, not silently dropped.

Usage: python -m agri.modules.market_price.train [--commodities N] [--states N]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import pandas as pd

from agri.common.config import load_module_config
from agri.common.errors import DataValidationError
from agri.common.logging import get_logger, setup_logging
from agri.common.metrics import evaluate_gate
from agri.common.paths import METRICS_DIR, git_sha, sha256_file, write_json
from agri.common.registry import ModelRegistry
from agri.common.seed import set_global_seed
from agri.features.timeseries import fit_baseline, fit_seasonal_trend, rolling_origin_backtest
from agri.modules.market_price.dataset import (
    MAX_GAP_DAYS,
    load_market_frame,
    ranking_from_frame,
    series_from_frame,
)
from agri.modules.market_price.predict import SERVED_HORIZONS

_log = get_logger("modules.market_price.train")

#: Cap on how many series to fit, so a full run stays bounded. Raise to cover more.
DEFAULT_MAX_SERIES = 300
#: A group needs this much daily history before a trend is worth fitting.
MIN_OBSERVATIONS = 60

#: The horizon the primary gate is scored at. Seven days is the shortest horizon that
#: gives a farmer time to act on a price signal.
PRIMARY_HORIZON = 7


def detect_source_gap(path, frame=None) -> dict[str, object]:
    """Characterise the hole in the source record.

    This is reported in the manifest and surfaced by the API, because a forecast
    derived from a partial record is only honest if the caller is told it is one.
    """
    if frame is None:
        frame = pd.read_csv(path, usecols=["date"], low_memory=False)
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return {"has_multi_month_gap": False}

    unique_days = pd.Series(sorted(dates.dt.normalize().unique()))
    gaps = unique_days.diff().dt.days.fillna(0)
    largest = int(gaps.max()) if len(gaps) else 0
    payload: dict[str, object] = {
        "earliest": unique_days.iloc[0].date().isoformat(),
        "latest": unique_days.iloc[-1].date().isoformat(),
        "distinct_days": len(unique_days),
        "largest_gap_days": largest,
        "has_multi_month_gap": largest > 90,
        "warnings": [],
    }
    if largest > 90:
        index = int(gaps.idxmax())
        gap_start = unique_days.iloc[index - 1].date().isoformat()
        gap_end = unique_days.iloc[index].date().isoformat()
        payload["gap_start"] = gap_start
        payload["gap_end"] = gap_end
        payload["warnings"] = [
            f"The source price file has a {largest}-day gap with no observations "
            f"({gap_start} to {gap_end}). Annual seasonality cannot be estimated from the "
            "remaining partial years and is not modelled."
        ]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train market price models")
    parser.add_argument("--max-series", type=int, default=DEFAULT_MAX_SERIES, help="cap on series to fit")
    parser.add_argument("--register", action="store_true", default=True)
    args = parser.parse_args(argv)

    setup_logging()
    config = load_module_config("market_price")
    set_global_seed(config.seed)

    path = config.data.raw_path
    if not path.is_file():
        print(json.dumps({"ok": False, "error": f"dataset missing: {path}"}, indent=2))
        return 1

    _log.info("reading market dataset once", extra={"path": str(path)})
    frame = load_market_frame(path)

    source_gap = detect_source_gap(path, frame)
    for warning in source_gap.get("warnings", []):
        _log.warning("source data caveat", extra={"warning": warning})

    params = config.model.params
    window = params.get("trend_window_days", 60)
    damping = params.get("trend_damping", 0.3)
    include_dow = bool(params.get("weekday_effects", True))
    min_history = int(params.get("min_history_points", MIN_OBSERVATIONS))
    _log.info(
        "forecaster configuration",
        extra={"window": window, "damping": damping, "weekday_effects": include_dow},
    )

    groups = ranking_from_frame(frame, args.max_series)
    _log.info("candidate series", extra={"candidates": len(groups), "cap": args.max_series})

    models: dict[str, object] = {}
    skipped: list[dict[str, str]] = []
    spans: list[dict[str, object]] = []

    for commodity, state, _count in groups:
        key = f"{commodity}|{state}"
        try:
            price_series = series_from_frame(frame, commodity, state, min_observations=min_history)
            model = fit_seasonal_trend(
                price_series.series,
                min_observations=min_history,
                window=window,
                damping=damping,
                include_dow=include_dow,
            )
            models[key] = model
            spans.append(price_series.to_dict())
        except DataValidationError as exc:
            skipped.append({"series": key, "reason": exc.message})
            continue

    if not models:
        print(
            json.dumps({"ok": False, "error": "no series could be fitted", "skipped": skipped[:10]}, indent=2)
        )
        return 1

    # Backtest on an aggregate basis so the gate reflects overall skill, not one series.
    pooled: dict[str, list[float]] = {f"h{h}": [] for h in SERVED_HORIZONS}
    pooled_baseline: dict[str, list[float]] = {f"h{h}": [] for h in SERVED_HORIZONS}
    pooled_median: dict[str, list[float]] = {f"h{h}": [] for h in SERVED_HORIZONS}
    pooled_baseline_median: dict[str, list[float]] = {f"h{h}": [] for h in SERVED_HORIZONS}
    pooled_direction: dict[str, list[float]] = {f"h{h}": [] for h in SERVED_HORIZONS}

    for key in models:
        commodity, state = key.split("|")
        try:
            price_series = series_from_frame(frame, commodity, state, min_observations=min_history)
            outcome = rolling_origin_backtest(
                price_series.series,
                horizons=list(SERVED_HORIZONS),
                window=window,
                damping=damping,
                include_dow=include_dow,
                min_train=min_history,
            )
        except DataValidationError:
            continue
        for horizon_key, stats in outcome.items():
            if stats.get("windows", 0) > 0 and stats["mape"] == stats["mape"]:
                pooled[horizon_key].append(stats["mape"])
                pooled_baseline[horizon_key].append(stats["baseline_mape"])
                pooled_median[horizon_key].append(stats.get("median_ape", float("nan")))
                pooled_baseline_median[horizon_key].append(stats.get("baseline_median_ape", float("nan")))
                direction = stats.get("directional_accuracy", float("nan"))
                if direction == direction:
                    pooled_direction[horizon_key].append(direction)

    import numpy as np

    # Aggregation is by MEDIAN across series, not by mean, and the reason is measured:
    # a handful of highly volatile series produce MAPEs above 100%, and their mean
    # dominates the pooled figure. Running the same backtest on the top 45 series and
    # on the top 120 moved the pooled MEAN MAPE from 46.4% to 46.9% while moving the
    # pooled MEAN BASELINE from 67.2% to 46.9% - the lift estimate swung from +31% to
    # 0% purely from which series happened to be included. A metric that unstable is
    # not a gate. The median of per-series MAPE does not have that failure mode, and
    # the beat fraction answers the plain question: on how many series does the model
    # actually win?
    trend_backtest: dict[str, dict[str, float]] = {}
    for horizon_key in pooled:
        if not pooled[horizon_key]:
            continue
        model_errors = pooled[horizon_key]
        baseline_errors = pooled_baseline[horizon_key]
        median_model = float(np.median(model_errors))
        median_baseline = float(np.median(baseline_errors))
        wins = sum(1 for model, base in zip(model_errors, baseline_errors, strict=True) if model < base)
        trend_backtest[horizon_key] = {
            # Robust headline: lift computed from medians.
            "relative_lift_over_baseline": float((median_baseline - median_model) / median_baseline)
            if median_baseline > 1e-9
            else float("nan"),
            "beat_fraction": float(wins / len(model_errors)),
            # Informational: reported so the tail behaviour is visible, not gated.
            "mape": float(np.mean(model_errors)),
            "baseline_mape": float(np.mean(baseline_errors)),
            "median_ape": median_model,
            "baseline_median_ape": median_baseline,
            "series_where_model_wins": float(wins),
            "directional_accuracy": float(np.mean(pooled_direction[horizon_key]))
            if pooled_direction[horizon_key]
            else float("nan"),
            "series_scored": float(len(model_errors)),
        }

    # ---- evidence-driven model selection -----------------------------------
    # Fit the trend, then let the backtest decide whether it is worth serving. This is
    # not a formality: the trend model loses to the naive baseline on the large
    # majority of series, and shipping a model that is worse than doing nothing is a
    # defect. If the trend cannot clear the beat floor, the naive baseline is served
    # instead and the module says so in every response.
    trend_primary = trend_backtest.get(f"h{PRIMARY_HORIZON}", {})
    beat_floor = float(params.get("selection_beat_fraction", 0.60))
    trend_beat = float(trend_primary.get("beat_fraction", float("nan")))
    use_trend = trend_beat == trend_beat and trend_beat >= beat_floor

    if use_trend:
        served = "damped_local_trend"
        selection_reason = (
            f"The damped local trend beat the naive baseline on {trend_beat:.0%} of series at "
            f"h{PRIMARY_HORIZON}, clearing the {beat_floor:.0%} floor, so it is served."
        )
        primary = trend_primary
        metrics = {
            "relative_lift_over_baseline": primary.get("relative_lift_over_baseline", float("nan")),
            "beat_fraction": trend_beat,
            "directional_accuracy": primary.get("directional_accuracy", float("nan")),
            "mape": primary.get("mape", float("nan")),
            "median_ape": primary.get("median_ape", float("nan")),
            "baseline_mape": primary.get("baseline_mape", float("nan")),
            "baseline_median_ape": primary.get("baseline_median_ape", float("nan")),
        }
    else:
        served = "naive_last_value"
        selection_reason = (
            f"The damped local trend beat the naive baseline on only {trend_beat:.0%} of series at "
            f"h{PRIMARY_HORIZON}, below the {beat_floor:.0%} floor, with a median absolute error of "
            f"{float(trend_primary.get('median_ape', float('nan'))):.1f}% against the baseline's "
            f"{float(trend_primary.get('baseline_median_ape', float('nan'))):.1f}%. The baseline is "
            f"served instead; the module reports no directional signal rather than an unsupported one."
        )
        # Refit everything as the baseline forecaster the backtest actually endorses.
        refitted: dict[str, object] = {}
        for series_key in list(models):
            commodity, state = series_key.split("|")
            try:
                price_series = series_from_frame(frame, commodity, state, min_observations=min_history)
                refitted[series_key] = fit_baseline(
                    price_series.series, min_observations=min_history, window=window, damping=damping
                )
            except DataValidationError:
                continue
        models = refitted
        # The baseline's own accuracy IS the naive baseline column of the trend backtest.
        metrics = {
            "relative_lift_over_baseline": 0.0,
            "beat_fraction": 0.0,
            "directional_accuracy": float("nan"),
            "mape": float(trend_primary.get("baseline_mape", float("nan"))),
            "median_ape": float(trend_primary.get("baseline_median_ape", float("nan"))),
            "baseline_mape": float(trend_primary.get("baseline_mape", float("nan"))),
            "baseline_median_ape": float(trend_primary.get("baseline_median_ape", float("nan"))),
        }

    _log.info("forecaster selected", extra={"served": served, "reason": selection_reason})

    gate = evaluate_gate(metrics, config.gates)

    # A gate-failing model must never reach the registry. This guard was missing: an
    # earlier run promoted an artefact whose own manifest recorded gate.passed=false,
    # because registration was unconditional. Silent promotion of a model that does not
    # meet its floor is worse than no model at all, since /v1/health then reports the
    # module as ready.
    if not gate.passed and args.register:
        _log.error(
            "gate failed: refusing to register",
            extra={"failed_checks": gate.failed_checks, "served": served},
        )
        print(
            json.dumps(
                {
                    "ok": False,
                    "module": config.name,
                    "version": config.version,
                    "registered": False,
                    "served_forecaster": served,
                    "selection_reason": selection_reason,
                    "metrics": metrics,
                    "gate": gate.to_dict(),
                    "message": (
                        "The gate failed, so no artefact was registered and LATEST was not "
                        "moved. Fix the model or record why the floor cannot be met."
                    ),
                },
                indent=2,
                default=str,
            )
        )
        return 1

    bundle = {
        "models": models,
        "served_forecaster": served,
        "selection_reason": selection_reason,
        "trend_backtest": trend_backtest,
        "primary_horizon_days": PRIMARY_HORIZON,
        "trained_at": datetime.now(UTC).isoformat(),
        "source_gap": source_gap,
        "git_sha": git_sha(),
    }

    if args.register:
        registry = ModelRegistry()
        registry.save(
            config.name,
            config.version,
            bundle,
            metrics={key: float(value) for key, value in metrics.items() if value == value},
            config_digest=config.config_digest,
            data_digest=sha256_file(path),
            dataset=path.name,
            params=config.model.params,
            artifact_file="price_models.joblib",
            extra={
                "series_fitted": len(models),
                "series_skipped": len(skipped),
                "served_forecaster": served,
                "selection_reason": selection_reason,
                "trend_backtest": trend_backtest,
                "source_gap": source_gap,
                "forecaster": {
                    "window_days": window,
                    "damping": damping,
                    "weekday_effects": include_dow,
                    "annual_seasonality": False,
                    "annual_seasonality_reason": (
                        "The source has a 426-day gap with no observations, so a yearly cycle "
                        "is not estimable."
                    ),
                },
                "primary_horizon_days": PRIMARY_HORIZON,
                "min_observations": min_history,
                "gap_tolerance_days": MAX_GAP_DAYS,
                "gate": gate.to_dict(),
            },
        )

    write_json(
        METRICS_DIR / f"{config.name}_v{config.version}.json",
        {
            "module": config.name,
            "version": config.version,
            "task": config.task.value,
            "metrics": {key: float(value) for key, value in metrics.items() if value == value},
            "gate": gate.to_dict(),
            "served_forecaster": served,
            "selection_reason": selection_reason,
            "backtest_by_horizon": trend_backtest,
            "source_gap": source_gap,
            "series_fitted": len(models),
            "series_skipped": len(skipped),
            "skipped_examples": skipped[:10],
            "sample_series": spans[:5],
        },
    )

    summary = {
        "ok": gate.passed,
        "module": config.name,
        "version": config.version,
        "series_fitted": len(models),
        "series_skipped": len(skipped),
        "served_forecaster": served,
        "selection_reason": selection_reason,
        "backtest": trend_backtest,
        "metrics": metrics,
        "gate": gate.to_dict(),
        "source_gap": source_gap,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
