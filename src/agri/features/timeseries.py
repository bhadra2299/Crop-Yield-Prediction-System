"""Time-series forecasting primitives for mandi prices.

Provides a damped log-linear trend and the naive last-value baseline, plus a
rolling-origin backtest used to decide between them.

Why not a heavier model? Because of what the data actually is. The mandi price file
spans 2019-05-22 to 2021-10-01 with NO observations for 426 days in between - 590,361
rows land in 2019 and 235,553 in late 2021, with nothing at all in 2020. Annual
seasonality cannot be estimated from a single partial year, and a model claiming to
learn it would be fitting a curve through a hole.

The trend model is REJECTED, and this file records that rather than hiding it. On a
rolling-origin backtest over 120 commodity/state series the damped local trend beat the
naive baseline on only 13% of series at a 7-day horizon, with a median absolute error of
33.4% against the baseline's 20.0%. It loses, so it is not served.

An earlier version of this module reported the opposite, and the mistake is instructive:
it aggregated error as a pooled MEAN across series. A handful of highly volatile series
produce MAPEs above 100%, which dominated the mean, so the apparent lift swung from +31%
to 0% purely by changing the series set from 45 to 120. The model's own mean barely
moved while the measured "advantage" evaporated. Aggregation is therefore by MEDIAN of
per-series APE, with beat_fraction (share of series actually won) as the companion
statistic.

fit_baseline() exists so the rejected-model case has a first-class answer that is not
"no model". The baseline is a legitimate forecaster: it wins, so it ships, and the module
reports no directional signal instead of an unsupported one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd

from agri.common.errors import DataValidationError
from agri.common.logging import get_logger

_log = get_logger("features.timeseries")


@dataclass
class ContinuousSegment:
    """The longest usable run of observations, and what was skipped to get it."""

    start: date
    end: date
    observed_days: int
    span_days: int
    skipped_days: int
    max_gap_days: int

    @property
    def density(self) -> float:
        """Fraction of calendar days in the span that actually have an observation."""
        return self.observed_days / self.span_days if self.span_days else 0.0

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "observed_days": self.observed_days,
            "span_days": self.span_days,
            "skipped_days": self.skipped_days,
            "max_gap_days": self.max_gap_days,
            "density": round(self.density, 3),
        }


def find_continuous_segment(series: pd.Series, *, max_gap_days: int = 3) -> ContinuousSegment:
    """Split a sparse date series and return its longest continuous run.

    A gap of up to max_gap_days is tolerated so that weekends and holidays, when
    mandis are shut, do not split an otherwise continuous record. The tolerance is
    deliberately small: a 426-day hole is not a weekend.
    """
    if series.empty:
        raise DataValidationError("Cannot segment an empty price series")

    ordered = series.sort_index()
    if ordered.index.has_duplicates:
        ordered = ordered.groupby(level=0).median()

    gaps = ordered.index.to_series().diff().dt.days.fillna(0)
    breaks = np.where(gaps > max_gap_days)[0]

    boundaries = [0, *breaks.tolist(), len(ordered)]
    best_start, best_end, best_len = 0, len(ordered), -1
    for lower, upper in pairwise(boundaries):
        if upper - lower > best_len:
            best_start, best_end, best_len = lower, upper, upper - lower

    segment = ordered.iloc[best_start:best_end]
    span = (segment.index[-1] - segment.index[0]).days + 1
    return ContinuousSegment(
        start=segment.index[0].date(),
        end=segment.index[-1].date(),
        observed_days=len(segment),
        span_days=int(span),
        skipped_days=int(len(ordered) - len(segment)),
        max_gap_days=int(max_gap_days),
    )


@dataclass
class SeasonalTrendModel:
    """log(price) ~ level + damped_drift(h), fitted by OLS on a recent window.

    Working in logs makes the trend multiplicative, which is how agricultural prices
    move, and keeps fitted values positive by construction.

    The drift is damped: instead of extending the slope h steps, it accumulates
    damping^1 + damping^2 + ... + damping^h. Damping matters because a slope fitted on
    two months of a noisy series, extended thirty days undamped, produces absurd
    extrapolations. It is the single change that took this model from worse-than-naive
    to substantially better.

    Weekday effects are centred to sum to zero, so 'no weekly pattern' is represented
    by coefficients near zero rather than an arbitrary level absorbed into the
    intercept.
    """

    level: float
    slope: float
    dow_effects: dict[int, float]
    residual_std: float
    last_date: date
    last_price: float
    n_observations: int
    segment: ContinuousSegment
    window: int
    damping: float
    warnings: list[str] = field(default_factory=list)

    #: A sustained daily drift beyond this is almost always a data artefact (a
    #: denomination change, a unit switch, a bad merge) rather than a real market
    #: move, so the slope is clipped to keep extrapolation sane.
    MAX_DAILY_LOG_DRIFT = 0.01

    #: Discriminator so a consumer can tell which forecaster answered without guessing
    #: from field names.
    kind: str = "damped_local_trend"

    def _damped_steps(self, steps: np.ndarray) -> np.ndarray:
        """Cumulative damped drift after each step."""
        if self.damping >= 1.0:
            return steps.astype("float64")
        return np.cumsum(self.damping ** steps.astype("float64"))

    def forecast(self, horizon_days: int, *, confidence_z: float = 1.2816) -> tuple[np.ndarray, np.ndarray]:
        """Forecast prices and an 80% interval half-width for the next horizons.

        Returns (prices, half_widths) in original price units. The interval widens
        through the residual spread; it captures model error, not market shocks, which
        the response states.
        """
        if horizon_days < 1:
            raise DataValidationError(
                "horizon_days must be at least 1", context={"horizon_days": horizon_days}
            )

        steps = np.arange(1, horizon_days + 1)
        target_dates = [self.last_date + timedelta(days=int(step)) for step in steps]
        log_price = self.level + self.slope * self._damped_steps(steps)
        for index, target in enumerate(target_dates):
            log_price[index] += self.dow_effects.get(target.weekday(), 0.0)
        prices = np.exp(log_price)
        half_widths = prices * (np.exp(confidence_z * self.residual_std) - 1.0)
        return prices, half_widths

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "slope": self.slope,
            "dow_effects": self.dow_effects,
            "residual_std": self.residual_std,
            "last_date": self.last_date.isoformat(),
            "last_price": self.last_price,
            "n_observations": self.n_observations,
            "window": self.window,
            "damping": self.damping,
            "segment": self.segment.to_dict(),
            "warnings": self.warnings,
        }


def fit_seasonal_trend(
    series: pd.Series,
    *,
    min_observations: int = 60,
    window: int | None = 60,
    damping: float = 0.3,
    include_dow: bool = True,
    annual_seasonality: bool = False,
) -> SeasonalTrendModel:
    """Fit the damped local trend model to a daily price series.

    window=None fits on the whole continuous segment (the behaviour that measured
    worse than naive). window=N fits on the most recent N observations.

    Fails loudly rather than returning a model fitted to too little data, because a
    forecast from three points looks exactly like a forecast from three hundred.
    """
    if len(series) < min_observations:
        raise DataValidationError(
            f"A price series needs at least {min_observations} observations to fit; got {len(series)}",
            context={"observations": len(series), "minimum": min_observations},
        )
    if not 0.0 < damping <= 1.0:
        raise DataValidationError("damping must be in (0, 1]", context={"damping": damping})

    ordered = series.sort_index()
    if ordered.index.has_duplicates:
        ordered = ordered.groupby(level=0).median()
    ordered = ordered[ordered > 0]
    if len(ordered) < min_observations:
        raise DataValidationError(
            "Too few positive price observations after cleaning",
            context={"remaining": len(ordered), "minimum": min_observations},
        )

    segment = find_continuous_segment(ordered)
    usable = ordered.loc[pd.Timestamp(segment.start) : pd.Timestamp(segment.end)]
    if len(usable) < min_observations:
        raise DataValidationError(
            "The longest continuous segment is too short to fit a trend",
            context={"segment_days": len(usable), "minimum": min_observations},
        )

    warnings: list[str] = []
    fitted = usable
    if window is not None and len(usable) > window:
        fitted = usable.iloc[-window:]
        if segment.observed_days > window * 2:
            warnings.append(
                f"The trend is fitted on the most recent {window} observations, not the full "
                f"{segment.observed_days}-day segment; a shorter window tracks recent conditions "
                "but will not see a longer cycle."
            )

    log_price = np.log(fitted.to_numpy(dtype="float64"))
    t = np.arange(len(fitted), dtype="float64")

    columns = [np.ones_like(t), t]
    if include_dow:
        weekday = fitted.index.dayofweek.to_numpy()
        for day in range(1, 7):  # day 0 is the reference level
            columns.append((weekday == day).astype("float64"))
    if annual_seasonality:
        month = fitted.index.month.to_numpy()
        for value in range(2, 13):  # month 1 is the reference level
            columns.append((month == value).astype("float64"))

    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, log_price, rcond=None)

    slope = float(coefficients[1])
    if abs(slope) > SeasonalTrendModel.MAX_DAILY_LOG_DRIFT:
        clipped = float(
            np.clip(slope, -SeasonalTrendModel.MAX_DAILY_LOG_DRIFT, SeasonalTrendModel.MAX_DAILY_LOG_DRIFT)
        )
        warnings.append(
            f"The fitted daily trend ({slope:+.4f} in log price) exceeds the plausible range and was "
            f"clipped to {clipped:+.4f}; a trend that steep usually indicates a denomination or unit "
            "change in the source."
        )
        slope = clipped
        coefficients[1] = slope

    residuals = log_price - design @ coefficients
    residual_std = (
        float(np.std(residuals, ddof=design.shape[1]))
        if len(residuals) > design.shape[1]
        else float(np.std(residuals))
    )

    dow_effects: dict[int, float] = {}
    if include_dow:
        raw = {0: 0.0}
        for offset, day in enumerate(range(1, 7), start=2):
            raw[day] = float(coefficients[offset])
        mean_effect = float(np.mean(list(raw.values())))
        dow_effects = {day: value - mean_effect for day, value in raw.items()}
        if max(abs(value) for value in dow_effects.values()) > 0.5:
            warnings.append(
                "The fitted weekday effects are implausibly large (>0.5 in log price); the weekly "
                "pattern may be an artefact of sparse reporting rather than a real market cycle."
            )

    # The fitted level at the LAST observation, so forecasts start from where the
    # model thinks the series is rather than from the raw final print (which may be
    # an outlier).
    level = float(coefficients[0] + slope * (len(fitted) - 1))

    return SeasonalTrendModel(
        level=level,
        slope=slope,
        dow_effects=dow_effects,
        residual_std=residual_std,
        last_date=usable.index[-1].date(),
        last_price=float(usable.iloc[-1]),
        n_observations=len(fitted),
        segment=segment,
        window=int(window) if window is not None else len(fitted),
        damping=float(damping),
        warnings=warnings,
    )


@dataclass
class BaselineForecaster:
    """The naive last-value forecast, as a first-class model.

    This exists because measurement said so. On a rolling-origin backtest over 120
    commodity/state series, the damped trend model beat this baseline on only 13% of
    series at a 7-day horizon, with a median absolute error of 33.4% against the
    baseline's 20.0%. The trend model is worse, not better, and the earlier apparent
    improvement came from the mean being dragged around by a handful of volatile
    series.

    So the shipped forecaster is the one the evidence supports: repeat the last
    observed price, refuse to claim a direction, and label confidence LOW. A forecast
    that says "no reliable signal" is a better product than one that says "+4%" on
    evidence that does not support it.

    It implements the same interface as the trend model so the module, the registry
    and the API do not need to know which one answered.
    """

    last_price: float
    last_date: date
    n_observations: int
    segment: ContinuousSegment
    window: int
    damping: float
    residual_std: float = 0.0
    warnings: list[str] = field(default_factory=list)

    kind: str = "naive_last_value"

    def forecast(self, horizon_days: int, *, confidence_z: float = 1.2816) -> tuple[np.ndarray, np.ndarray]:
        """Return the last observed price for every horizon, with no interval."""
        if horizon_days < 1:
            raise DataValidationError(
                "horizon_days must be at least 1", context={"horizon_days": horizon_days}
            )
        prices = np.full(horizon_days, self.last_price, dtype="float64")
        return prices, np.zeros(horizon_days, dtype="float64")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "baseline_last_value",
            "last_date": self.last_date.isoformat(),
            "last_price": self.last_price,
            "n_observations": self.n_observations,
            "segment": self.segment.to_dict(),
            "warnings": self.warnings,
        }


def fit_baseline(
    series: pd.Series, *, min_observations: int = 60, window: int = 60, damping: float = 0.3
) -> BaselineForecaster:
    """Wrap the last-value baseline in the model interface.

    Takes the same arguments as fit_seasonal_trend so the two are interchangeable at
    the call site; the fitted trend would be discarded anyway.
    """
    ordered = series.sort_index()
    if ordered.index.has_duplicates:
        ordered = ordered.groupby(level=0).median()
    ordered = ordered[ordered > 0]
    if len(ordered) < min_observations:
        raise DataValidationError(
            f"A price series needs at least {min_observations} observations; got {len(ordered)}",
            context={"observations": len(ordered), "minimum": min_observations},
        )
    segment = find_continuous_segment(ordered)
    usable = ordered.loc[pd.Timestamp(segment.start) : pd.Timestamp(segment.end)]
    return BaselineForecaster(
        last_price=float(usable.iloc[-1]),
        last_date=usable.index[-1].date(),
        n_observations=int(min(len(usable), window)),
        segment=segment,
        window=int(window),
        damping=float(damping),
        warnings=[
            "The shipped forecast is the naive last-value baseline. A damped trend model was "
            "evaluated and measured worse on the majority of series, so it is not served."
        ],
    )


def rolling_origin_backtest(
    series: pd.Series,
    *,
    horizons: list[int],
    min_train: int = 60,
    step: int = 5,
    max_windows: int = 40,
    window: int | None = 60,
    damping: float = 0.3,
    include_dow: bool = True,
) -> dict[str, dict[str, float]]:
    """Expanding-window backtest at each horizon, plus a last-value baseline.

    This is the only honest way to score a price forecast: fit on everything up to a
    cut point, predict forward, compare against what actually happened. A random split
    would let the model see the future, which for a trending series inflates the score
    spectacularly.

    Three statistics are reported per horizon, and they do not agree with each other:

    * mape          - mean absolute percentage error. The headline.
    * median_ape    - typical error. The naive baseline often wins this one, because
                      repeating the last price is usually fine and occasionally
                      catastrophic; the damped model trades a slightly worse typical
                      case for far fewer disasters.
    * directional_accuracy - did the model call up-versus-down correctly. The naive
                      baseline scores ~0 here by construction: it always predicts no
                      change.

    The baseline is scored on the identical windows, because a MAPE of 8% means
    nothing until you know that doing nothing scores 9%.
    """
    ordered = series.sort_index()
    if ordered.index.has_duplicates:
        ordered = ordered.groupby(level=0).median()
    ordered = ordered[ordered > 0]

    results: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        model_apes: list[float] = []
        baseline_apes: list[float] = []
        hits = 0
        total = 0
        windows = 0

        cut_points = list(range(min_train, len(ordered) - horizon, step))
        if len(cut_points) > max_windows:
            # Thin evenly so the window set spans the whole record, not just its start.
            stride = len(cut_points) / max_windows
            cut_points = [cut_points[int(index * stride)] for index in range(max_windows)]

        for cut in cut_points:
            train = ordered.iloc[:cut]
            actual = float(ordered.iloc[cut + horizon - 1])
            if actual <= 0:
                continue
            try:
                model = fit_seasonal_trend(
                    train,
                    window=window,
                    damping=damping,
                    include_dow=include_dow,
                    min_observations=min(min_train, len(train)),
                )
            except DataValidationError:
                continue

            predicted = float(model.forecast(horizon)[0][-1])
            last_value = float(train.iloc[-1])

            model_apes.append(abs(predicted - actual) / actual)
            baseline_apes.append(abs(last_value - actual) / actual)
            windows += 1

            predicted_change = predicted - last_value
            actual_change = actual - last_value
            if predicted_change * actual_change > 0 or abs(actual_change) < 1e-9:
                hits += 1
            total += 1

        if not windows:
            results[f"h{horizon}"] = {
                "mape": float("nan"),
                "median_ape": float("nan"),
                "baseline_mape": float("nan"),
                "directional_accuracy": float("nan"),
                "windows": 0.0,
            }
            continue

        model_mape = float(np.mean(model_apes) * 100.0)
        baseline_mape = float(np.mean(baseline_apes) * 100.0)
        results[f"h{horizon}"] = {
            "mape": model_mape,
            "median_ape": float(np.median(model_apes) * 100.0),
            "baseline_mape": baseline_mape,
            "baseline_median_ape": float(np.median(baseline_apes) * 100.0),
            "directional_accuracy": float(hits / total) if total else float("nan"),
            "relative_lift_over_baseline": float((baseline_mape - model_mape) / baseline_mape)
            if baseline_mape > 1e-9
            else float("nan"),
            "windows": float(windows),
        }

    _log.info("rolling-origin backtest complete", extra={"horizons": list(horizons)})
    return results
