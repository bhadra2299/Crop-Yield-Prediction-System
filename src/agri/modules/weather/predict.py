"""weather: live weather and FAO-56 reference ET for a location.

This module has no model and no artefact. Its job is to fetch, normalise and
cache the weather that every other irrigation decision depends on, and to report
honestly when it could not.

Two design decisions worth stating:

* The output includes a computed ET0 per forecast day. Computing ET0 here rather
  than in the irrigation module means the number that drives scheduling is
  produced once, from the same equation, no matter which caller asked.

* A provider failure is a typed ExternalServiceError, not a crash. The
  orchestrator is expected to degrade - an irrigation recommendation computed
  from the last cached forecast is more useful than an error page, as long as the
  response says which forecast it used and how old it was.
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agri.common.base import BasePredictor
from agri.common.config import get_settings
from agri.common.errors import ExternalServiceError
from agri.common.logging import get_logger
from agri.common.schemas import Confidence, Prediction
from agri.features.physics import DailyWeather, compute_et0
from agri.ingestion.providers import fetch_nasa_power, fetch_openweather

_log = get_logger("modules.weather")


class Request(BaseModel):
    """A weather lookup for one location."""

    latitude: float = Field(17.385, ge=-90, le=90, description="latitude, degrees")
    longitude: float = Field(78.4867, ge=-180, le=180, description="longitude, degrees")
    elevation_m: float = Field(0.0, ge=-430, le=6000, description="elevation, metres")
    history_days: int = Field(30, ge=0, le=365, description="days of NASA POWER history to include")
    include_history: bool = Field(False, description="fetch the historical series as well as the forecast")
    force_refresh: bool = Field(False, description="bypass the on-disk cache")


class ForecastDay(BaseModel):
    """One forecast day with its computed reference evapotranspiration."""

    day: str
    t_max: float
    t_min: float
    humidity_mean: float | None = None
    rainfall: float = 0.0
    wind_speed_2m: float = 2.0
    et0_mm: float = Field(..., description="FAO-56 reference evapotranspiration for the day, mm")


class Response(Prediction[float]):
    """Current conditions plus a forecast with per-day ET0."""

    value: float = Field(..., description="today's FAO-56 reference ET0 in mm/day")
    source: str = Field(..., description="which provider answered: openweather or nasa_power")
    current: dict[str, float] = Field(default_factory=dict)
    forecast: list[ForecastDay] = Field(default_factory=list)
    history_days: int = Field(0, description="number of historical days returned")
    cached: bool = Field(False, description="whether the payload came from the on-disk cache")


class Predictor(BasePredictor[Request, Response]):
    """Serves normalised weather with FAO-56 ET0. No trained artefact."""

    module_name: ClassVar[str] = "weather"
    requires_artifact: ClassVar[bool] = False
    request_model = Request
    response_model = Response

    def _load(self) -> None:
        # Settings are read through the accessor, not from an instance attribute:
        # BasePredictor owns config and artefact state, not process settings. Reading
        # self.settings here raised AttributeError and took the module down.
        settings = get_settings()
        params = self.config.model.params
        self.api_key = settings.openweather_api_key
        self.cache_ttl_hours = float(params.get("cache_ttl_hours", 6))
        self.nasa_base = str(params.get("nasa_power_base_url", "")) or None
        self.forecast_days = int(params.get("forecast_days", 7))

    def _predict(self, request: Request) -> Response:
        snapshot = fetch_openweather(
            request.latitude,
            request.longitude,
            api_key=self.api_key,
            cache_ttl_hours=self.cache_ttl_hours,
            force_refresh=request.force_refresh,
        )

        warnings = list(snapshot.warnings)
        _log.info(
            "weather fetched",
            extra={"source": snapshot.source, "cached": snapshot.cached, "days": len(snapshot.forecast)},
        )

        current_block = snapshot.current or {}
        current_weather = DailyWeather(
            t_max=float(current_block.get("t_max", 30.0)),
            t_min=float(current_block.get("t_min", 20.0)),
            humidity_mean=current_block.get("humidity"),
            wind_speed_2m=float(current_block.get("wind_speed_2m", 2.0)),
            rainfall=float(current_block.get("rainfall", 0.0)),
            elevation=request.elevation_m,
        )
        today = date.today()
        et0_today = compute_et0(current_weather, request.latitude, today.timetuple().tm_yday)

        forecast: list[ForecastDay] = []
        for entry in snapshot.forecast[: self.forecast_days]:
            day_value = date.fromisoformat(entry["day"])
            weather = DailyWeather(
                t_max=float(entry["t_max"]),
                t_min=float(entry["t_min"]),
                humidity_mean=entry.get("humidity_mean"),
                wind_speed_2m=float(entry.get("wind_speed_2m", 2.0)),
                rainfall=float(entry.get("rainfall", 0.0)),
                elevation=request.elevation_m,
            )
            forecast.append(
                ForecastDay(
                    day=entry["day"],
                    t_max=weather.t_max,
                    t_min=weather.t_min,
                    humidity_mean=weather.humidity_mean,
                    rainfall=weather.rainfall,
                    wind_speed_2m=weather.wind_speed_2m,
                    et0_mm=round(compute_et0(weather, request.latitude, day_value.timetuple().tm_yday), 3),
                )
            )

        history_count = 0
        if request.include_history and request.history_days > 0:
            from agri.ingestion.providers import default_window

            window_start, window_end = default_window(request.history_days)
            try:
                rows = fetch_nasa_power(
                    request.latitude,
                    request.longitude,
                    start=window_start,
                    end=window_end,
                    base_url=self.nasa_base or "https://power.larc.nasa.gov/api/temporal/daily/point",
                    cache_ttl_hours=float(self.config.model.params.get("cache_ttl_hours", 6)) * 4,
                )
                history_count = len(rows)
            except ExternalServiceError as exc:
                # History is supplementary: its absence degrades the response, not the request.
                warnings.append(f"Historical weather unavailable: {exc.message}")
                _log.warning("nasa power history unavailable", extra={"reason": exc.message})

        # Confidence is about data provenance, not model certainty.
        if snapshot.cached:
            confidence = Confidence.MEDIUM
        elif forecast:
            confidence = Confidence.HIGH
        else:
            confidence = Confidence.LOW

        reasons = [
            f"Conditions came from {snapshot.source}"
            + (" (served from the on-disk cache)" if snapshot.cached else " (live fetch)"),
            f"Reference evapotranspiration today is {et0_today:.2f} mm/day, computed with the "
            "FAO-56 Penman-Monteith equation from the reported temperature, humidity and wind.",
        ]
        if forecast:
            total_rain = sum(day.rainfall for day in forecast)
            reasons.append(
                f"{len(forecast)} forecast day(s) available with {total_rain:.1f} mm of rain expected in total."
            )
        if warnings:
            reasons.extend(warnings)

        return Response(
            value=round(et0_today, 3),
            source=snapshot.source,
            current={k: round(float(v), 3) for k, v in current_block.items()},
            forecast=forecast,
            history_days=history_count,
            cached=snapshot.cached,
            confidence_score=None,
            confidence=confidence,
            reasons=reasons,
            model_name=self.module_name,
            model_version=self.version,
        )


SAMPLE: dict[str, Any] = {
    "latitude": 17.385,
    "longitude": 78.4867,
    "elevation_m": 542,
    "include_history": False,
}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("weather", Predictor, SAMPLE))
