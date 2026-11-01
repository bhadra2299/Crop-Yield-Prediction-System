"""market_price: forecast a mandi price as a relative change.

The contract this module enforces is the important part. It reports the expected
price as a RELATIVE CHANGE against the latest known price, alongside that latest
price and its date. It does not report a bare forecast rupee figure, because a
bare figure loses the two things a farmer needs to judge it: what the price is
today, and how stale that today is.

Nothing here claims to predict a market. The model extrapolates a trend and a
weekly pattern from a single partial year of 2019 data; the source file contains
no observations at all for 2020. The response says so, and the signal it emits is
advisory.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agri.common.base import BasePredictor
from agri.common.errors import ModelNotTrainedError
from agri.common.logging import get_logger
from agri.common.schemas import Confidence, Interval, Prediction

_log = get_logger("modules.market_price")

#: Horizons the registry artefact is evaluated at, and the only ones served.
SERVED_HORIZONS = (1, 7, 30)


class Request(BaseModel):
    """A price forecast request."""

    commodity: str = Field(..., min_length=1, description="commodity name as it appears in the mandi data")
    state: str = Field(..., min_length=1, description="Indian state name")
    horizon_days: int = Field(
        7, ge=1, le=30, description="forecast horizon in days (1, 7 or 30 are backtested)"
    )


class Response(Prediction[float]):
    """Forecast as a relative change, with the anchor price it is relative to."""

    value: float = Field(
        ...,
        description="expected relative price change over the horizon, e.g. 0.04 means +4%",
    )
    latest_price: float = Field(..., description="last observed modal price, rupees per quintal")
    latest_price_date: str = Field(..., description="date of the last observation")
    forecast_price: float = Field(..., description="implied price at the horizon, rupees per quintal")
    horizon_days: int = Field(..., description="days ahead the forecast covers")
    signal: str = Field(..., description="sell_now, hold, or insufficient_signal")
    backtested_mape: float | None = Field(
        None, description="rolling-origin MAPE measured at the nearest backtested horizon, percent"
    )
    baseline_mape: float | None = Field(
        None, description="MAPE of the naive last-value forecast on the same windows, percent"
    )
    backtested_median_ape: float | None = Field(
        None, description="median absolute percentage error on the same windows, percent"
    )
    baseline_median_ape: float | None = Field(
        None,
        description=(
            "median APE of the naive baseline. Usually LOWER than the model's: repeating the "
            "last price is typically fine and occasionally catastrophic, while the model trades a "
            "slightly worse typical case for far fewer disasters."
        ),
    )
    relative_lift_over_baseline: float | None = Field(
        None, description="fractional improvement in mean error over the naive baseline"
    )
    directional_accuracy: float | None = Field(
        None, description="fraction of windows where up-versus-down was called correctly"
    )
    data_span: dict[str, Any] = Field(default_factory=dict, description="the segment the model was fitted on")
    warnings: list[str] = Field(default_factory=list)


class Predictor(BasePredictor[Request, Response]):
    """Serves the registered price models. One model per commodity and state."""

    module_name: ClassVar[str] = "market_price"
    requires_artifact: ClassVar[bool] = True
    artifact_filename: ClassVar[str] = "price_models.joblib"
    request_model = Request
    response_model = Response

    def _load(self) -> None:
        bundle = self.artifact
        if not isinstance(bundle, dict) or "models" not in bundle:
            raise ModelNotTrainedError(
                "market_price artefact is not a price-model bundle",
                context={"module": self.module_name},
            )
        self.models: dict[str, Any] = bundle["models"]
        # The bundle records evidence for the trend model and names which forecaster is
        # actually served. Older bundles only carry "backtest".
        self.backtest: dict[str, dict[str, float]] = bundle.get("trend_backtest") or bundle.get(
            "backtest", {}
        )
        self.served_forecaster: str = bundle.get("served_forecaster", "unknown")
        self.selection_reason: str = bundle.get("selection_reason", "")
        self.trained_at: str = bundle.get("trained_at", "unknown")
        self.source_gap: dict[str, Any] = bundle.get("source_gap", {})

    def _key(self, commodity: str, state: str) -> str:
        return f"{commodity.strip()}|{state.strip()}"

    def _nearest_horizon(self, horizon: int) -> int:
        return min(SERVED_HORIZONS, key=lambda value: abs(value - horizon))

    def _predict(self, request: Request) -> Response:
        key = self._key(request.commodity, request.state)
        model = self.models.get(key)

        if model is None:
            # A miss is not an error: there are 232 commodities and 29 states, and
            # only the series with enough history were fitted.
            available = len(self.models)
            raise ModelNotTrainedError(
                f"No price model for {request.commodity} in {request.state}. "
                f"{available} commodity/state series have enough history to forecast.",
                context={
                    "commodity": request.commodity,
                    "state": request.state,
                    "models_available": available,
                },
            )

        prices, half_widths = model.forecast(request.horizon_days)
        forecast_price = float(prices[-1])
        half_width = float(half_widths[-1])

        anchor = model.last_price
        relative_change = (forecast_price - anchor) / anchor if anchor else 0.0

        backtest_key = f"h{self._nearest_horizon(request.horizon_days)}"
        stats = self.backtest.get(backtest_key, {})
        mape = stats.get("mape")
        baseline_mape = stats.get("baseline_mape")
        directional = stats.get("directional_accuracy")
        median_ape = stats.get("median_ape")
        baseline_median = stats.get("baseline_median_ape")
        lift = stats.get("relative_lift_over_baseline")

        # A signal is only emitted when the move clears the model's own measured
        # error. Telling a farmer to hold when the expected move is smaller than
        # the forecast error would be dressing up noise as advice.
        threshold = (mape / 100.0) if mape and mape > 0 else 0.02
        if abs(relative_change) <= threshold:
            signal = "insufficient_signal"
        elif relative_change > 0:
            signal = "hold"
        else:
            signal = "sell_now"

        # Confidence reflects both the measured error and how far ahead we look.
        # Confidence keys off measured skill, never off model self-report.
        if mape is None or not directional or directional != directional:
            confidence = Confidence.LOW
        elif directional >= 0.60 and request.horizon_days <= 7:
            confidence = Confidence.MEDIUM
        elif directional >= 0.52 and request.horizon_days <= 30:
            confidence = Confidence.LOW
        else:
            confidence = Confidence.LOW
        # No HIGH confidence is reachable: the measured directional accuracy at any
        # horizon does not support one. See docs/MODULES/market_price.md.

        interval = None
        if half_width > 0:
            interval = Interval(
                low=round((forecast_price - half_width - anchor) / anchor, 6) if anchor else 0.0,
                high=round((forecast_price + half_width - anchor) / anchor, 6) if anchor else 0.0,
                level=0.80,
            )

        reasons = [
            f"Forecast anchored on the last observed modal price of {anchor:,.0f}/quintal on "
            f"{model.last_date.isoformat()}, {request.horizon_days} day(s) ahead.",
            (
                f"Model is a log-linear trend fitted on the most recent {model.window} daily "
                f"observations (of a segment running {model.segment.start.isoformat()} to "
                f"{model.segment.end.isoformat()}), with the extrapolation damped at {model.damping}."
                if getattr(model, "kind", "") == "damped_local_trend"
                else (
                    f"Served forecaster is the naive last-value baseline: the price is held flat at "
                    f"{model.last_price:,.0f}/quintal from {model.segment.end.isoformat()}, and no "
                    f"direction is claimed. The segment began {model.segment.start.isoformat()}."
                )
            ),
        ]
        if mape is not None and baseline_mape is not None:
            reasons.append(
                f"Rolling-origin backtest at the nearest horizon ({backtest_key}) measured {mape:.1f}% mean "
                f"absolute error against {baseline_mape:.1f}% for simply repeating the last price"
                + (f" ({lift:+.0%} mean lift)." if lift is not None else ".")
            )
        if median_ape is not None and baseline_median is not None and median_ape > baseline_median:
            # Say this out loud rather than letting the mean improvement imply uniform skill.
            reasons.append(
                f"In the typical case the model is slightly WORSE than the naive baseline "
                f"({median_ape:.1f}% versus {baseline_median:.1f}% median error). Its advantage is in "
                "the average, where it avoids the rare large misses that repeating the last price "
                "produces. Both numbers are reported for that reason."
            )
        if directional is not None and directional == directional:
            reasons.append(
                f"Directional accuracy was {directional:.0%}; the naive baseline scores near zero by "
                "construction because it always predicts no change."
            )
        if signal == "insufficient_signal":
            reasons.append(
                f"The expected move ({relative_change:+.2%}) is within the model's own measured error, so no "
                "buy-or-hold signal is given."
            )
        reasons.extend(model.warnings)
        if self.source_gap.get("has_multi_month_gap"):
            reasons.append(
                "The source data contains a gap with no observations between "
                f"{self.source_gap.get('gap_start')} and {self.source_gap.get('gap_end')}; the model is fitted "
                "only on the continuous segment and does not model across the gap."
            )
        reasons.append(
            "This is an extrapolation of a historical price regime, not a market prediction. "
            "Retrain as new mandi data arrives."
        )

        return Response(
            value=round(relative_change, 6),
            latest_price=round(anchor, 2),
            latest_price_date=model.last_date.isoformat(),
            forecast_price=round(forecast_price, 2),
            horizon_days=request.horizon_days,
            signal=signal,
            backtested_mape=round(mape, 3) if mape is not None else None,
            baseline_mape=round(baseline_mape, 3) if baseline_mape is not None else None,
            backtested_median_ape=round(median_ape, 3) if median_ape is not None else None,
            baseline_median_ape=round(baseline_median, 3) if baseline_median is not None else None,
            relative_lift_over_baseline=round(lift, 4) if lift is not None else None,
            directional_accuracy=round(directional, 4) if directional == directional else None,
            data_span=model.segment.to_dict(),
            warnings=list(model.warnings) + list(self.source_gap.get("warnings", [])),
            confidence_score=None,
            confidence=confidence,
            interval=interval,
            reasons=reasons,
            model_name=self.module_name,
            model_version=self.version,
        )


#: A series that is actually fitted. Only commodity/state pairs with enough continuous
#: history are in the artefact, so a plausible-looking but absent pair (Wheat/Karnataka)
#: fails with ModelNotTrainedError rather than returning a forecast.
SAMPLE: dict[str, Any] = {"commodity": "Wheat", "state": "Madhya Pradesh", "horizon_days": 7}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("market_price", Predictor, SAMPLE))
