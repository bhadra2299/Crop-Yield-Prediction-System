"""Concrete weather providers.

Two sources, deliberately: a forecast and a history. OpenWeather gives the next
few days; NASA POWER gives a multi-decade daily record. They are not
interchangeable, so the module reports which one answered.

Both are cached on disk keyed by (provider, location, window). Weather APIs are
rate limited and metered, and re-fetching the same day's forecast on every request
is both slow and rude. The cache is a courtesy to the provider, not a correctness
mechanism - a stale forecast is a real risk, so the TTL is short and explicit.

Neither provider is allowed to take the service down. Every call is wrapped so
that a missing key, a network failure or a rate limit produces a typed
ExternalServiceError that the orchestrator can degrade around, rather than a
traceback that kills the whole recommendation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from agri.common.errors import ConfigError, ExternalServiceError
from agri.common.logging import get_logger
from agri.common.paths import CACHE_DIR, ensure_dir

_log = get_logger("ingestion.providers")

OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"


@dataclass
class WeatherSnapshot:
    """A normalised weather record in the units FAO-56 expects.

    Provider payloads are converted at the boundary so nothing downstream has to
    know that OpenWeather reports temperature in Kelvin and wind in m/s while
    NASA POWER reports temperature in Celsius and wind in m/s at 10 m.
    """

    source: str
    latitude: float
    longitude: float
    fetched_at: str
    current: dict[str, float] = field(default_factory=dict)
    forecast: list[dict[str, float]] = field(default_factory=list)
    history: list[dict[str, float]] = field(default_factory=list)
    #: True when the payload came from disk rather than the provider.
    cached: bool = False
    #: Non-fatal problems worth surfacing (e.g. rain field absent on free tier).
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "fetched_at": self.fetched_at,
            "current": {k: round(v, 3) for k, v in self.current.items()},
            "forecast": [{k: round(v, 3) for k, v in day.items()} for day in self.forecast],
            "history_days": len(self.history),
            "cached": self.cached,
            "warnings": self.warnings,
        }


def _cache_file(provider: str, latitude: float, longitude: float, tag: str) -> Any:
    """Cache path for a provider call, rounded so nearby fields share an entry."""
    directory = ensure_dir(CACHE_DIR / "weather")
    key = f"{provider}_{latitude:.3f}_{longitude:.3f}_{tag}.json"
    return directory / key


def _read_cache(path: Any, ttl_hours: float) -> dict[str, Any] | None:
    import json

    if not path.is_file():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if age_hours > ttl_hours:
        _log.info("weather cache expired", extra={"provider": path.stem, "age_hours": round(age_hours, 2)})
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _log.warning("weather cache unreadable; refetching", extra={"path": str(path), "error": str(exc)})
        return None


def _write_cache(path: Any, payload: dict[str, Any]) -> None:
    import json

    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        # A cache write failure must never fail the request.
        _log.warning("could not write weather cache", extra={"path": str(path), "error": str(exc)})


def _http_get(
    url: str, params: dict[str, Any], *, timeout: float = 12.0, attempts: int = 3
) -> dict[str, Any]:
    """GET with bounded retries. Raises ExternalServiceError, never a bare error."""
    import httpx

    last: str = "unknown"
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.get(url, params=params, timeout=timeout)
            if response.status_code == 401:
                raise ExternalServiceError(
                    f"{url} rejected the credentials (401); check the configured API key",
                    context={"provider": url, "status": 401},
                )
            if response.status_code == 429:
                last = "rate limited (429)"
            elif response.status_code >= 400:
                last = f"HTTP {response.status_code}"
            else:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ExternalServiceError(
                        "weather provider returned a non-object payload",
                        context={"provider": url},
                    )
                return payload
        except ExternalServiceError:
            raise
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        except ValueError as exc:
            last = f"malformed JSON: {exc}"
        if attempt < attempts:
            # Exponential backoff; keeps a transient provider wobble from surfacing.
            time.sleep(0.5 * (2 ** (attempt - 1)))
    raise ExternalServiceError(
        f"Weather provider unavailable after {attempts} attempts ({last})",
        context={"provider": url, "last_error": last},
    )


def fetch_openweather(
    latitude: float,
    longitude: float,
    *,
    api_key: str | None,
    cache_ttl_hours: float = 6.0,
    force_refresh: bool = False,
) -> WeatherSnapshot:
    """Current conditions plus a multi-day forecast from OpenWeather.

    The free tier returns 3-hourly forecast steps; they are aggregated to daily
    values because the FAO-56 daily step is what the water balance uses.
    """
    if not api_key:
        raise ConfigError(
            "OPENWEATHER_API_KEY is not configured",
            context={"hint": "Set OPENWEATHER_API_KEY in the environment to enable live weather"},
        )

    cache_path = _cache_file("openweather", latitude, longitude, "forecast")
    payload = None if force_refresh else _read_cache(cache_path, cache_ttl_hours)
    cached = payload is not None

    if payload is None:
        common = {"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"}
        current = _http_get(f"{OPENWEATHER_BASE}/weather", common)
        forecast = _http_get(f"{OPENWEATHER_BASE}/forecast", common)
        payload = {"current": current, "forecast": forecast}
        _write_cache(cache_path, payload)

    warnings: list[str] = []
    current_block = payload.get("current", {})
    main = current_block.get("main", {}) or {}
    wind = current_block.get("wind", {}) or {}
    rain = current_block.get("rain", {}) or {}

    # 'rain' is absent on the free tier for dry conditions, not missing data.
    if "rain" not in current_block:
        warnings.append(
            "No precipitation field in the current-conditions payload (OpenWeather omits it when dry); "
            "rainfall is treated as zero."
        )

    current = {
        "temperature": float(main.get("temp", 25.0)),
        "t_max": float(main.get("temp_max", main.get("temp", 25.0))),
        "t_min": float(main.get("temp_min", main.get("temp", 25.0))),
        "humidity": float(main.get("humidity", 60.0)),
        "pressure": float(main.get("pressure", 101.3)),
        "wind_speed_2m": float(wind.get("speed", 2.0)),
        "rainfall": float(rain.get("1h", rain.get("3h", 0.0)) or 0.0),
    }

    # Aggregate 3-hourly steps into daily t_max/t_min/humidity/rainfall.
    daily: dict[str, dict[str, float]] = {}
    for step in payload.get("forecast", {}).get("list", []) or []:
        stamp = step.get("dt")
        if stamp is None:
            continue
        day = datetime.fromtimestamp(int(stamp), tz=UTC).date().isoformat()
        step_main = step.get("main", {}) or {}
        bucket = daily.setdefault(
            day,
            {"t_max": -999.0, "t_min": 999.0, "humidity_sum": 0.0, "humidity_n": 0.0, "rainfall": 0.0},
        )
        bucket["t_max"] = max(bucket["t_max"], float(step_main.get("temp_max", step_main.get("temp", 25.0))))
        bucket["t_min"] = min(bucket["t_min"], float(step_main.get("temp_min", step_main.get("temp", 25.0))))
        bucket["humidity_sum"] += float(step_main.get("humidity", 60.0))
        bucket["humidity_n"] += 1.0
        bucket["rainfall"] += float((step.get("rain", {}) or {}).get("3h", 0.0) or 0.0)

    forecast_days: list[dict[str, float]] = []
    for day in sorted(daily)[:7]:
        bucket = daily[day]
        forecast_days.append(
            {
                "day": day,
                "t_max": bucket["t_max"],
                "t_min": bucket["t_min"],
                "humidity_mean": bucket["humidity_sum"] / max(bucket["humidity_n"], 1.0),
                "rainfall": bucket["rainfall"],
                "wind_speed_2m": current["wind_speed_2m"],
            }
        )

    if not forecast_days:
        warnings.append("Forecast contained no usable daily steps; only current conditions are available.")

    return WeatherSnapshot(
        source="openweather",
        latitude=latitude,
        longitude=longitude,
        fetched_at=datetime.now(UTC).isoformat(),
        current=current,
        forecast=forecast_days,
        cached=cached,
        warnings=warnings,
    )


def fetch_nasa_power(
    latitude: float,
    longitude: float,
    *,
    start: date,
    end: date,
    base_url: str = NASA_POWER_BASE,
    cache_ttl_hours: float = 24.0,
    force_refresh: bool = False,
) -> list[dict[str, float]]:
    """Daily historical weather from NASA POWER.

    Selected parameters are the FAO-56 inputs plus wind at 2 m. POWER reports
    T2M_MAX/T2M_MIN in Celsius and WS2M in m/s, which is exactly what the
    Penman-Monteith implementation consumes, so no unit conversion happens here.
    """
    tag = f"history_{start.isoformat()}_{end.isoformat()}"
    cache_path = _cache_file("nasa_power", latitude, longitude, tag)
    payload = None if force_refresh else _read_cache(cache_path, cache_ttl_hours)

    if payload is None:
        params = {
            "parameters": "T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
        }
        payload = _http_get(base_url, params)
        _write_cache(cache_path, payload)

    properties = (payload.get("properties") or {}).get("parameter") or {}
    if not properties:
        raise ExternalServiceError(
            "NASA POWER returned no parameter block",
            context={"latitude": latitude, "longitude": longitude},
        )

    # POWER returns a map of parameter -> {date: value}, with -999 for missing.
    dates = sorted(properties.get("T2M_MAX", {}).keys())
    rows: list[dict[str, float]] = []
    for day in dates:
        t_max = float(properties.get("T2M_MAX", {}).get(day, -999))
        t_min = float(properties.get("T2M_MIN", {}).get(day, -999))
        if t_max <= -900 or t_min <= -900:
            continue
        humidity = float(properties.get("RH2M", {}).get(day, -999))
        rainfall = float(properties.get("PRECTOTCORR", {}).get(day, -999))
        solar = float(properties.get("ALLSKY_SFC_SW_DWN", {}).get(day, -999))
        wind = float(properties.get("WS2M", {}).get(day, -999))
        rows.append(
            {
                "day": day,
                "t_max": t_max,
                "t_min": t_min,
                "humidity_mean": None if humidity <= -900 else humidity,
                "rainfall": 0.0 if rainfall <= -900 else max(rainfall, 0.0),
                "solar_radiation": None if solar <= -900 else solar,
                "wind_speed_2m": 2.0 if wind <= -900 else max(wind, 0.0),
            }
        )
    return rows


def history_to_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    """Historical rows as a DataFrame with a real datetime index."""
    frame = pd.DataFrame(rows)
    if not frame.empty and "day" in frame.columns:
        frame["day"] = pd.to_datetime(frame["day"], format="%Y%m%d", errors="coerce")
        frame = frame.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)
    return frame


def default_window(days: int = 30) -> tuple[date, date]:
    """A trailing window ending yesterday (POWER has no same-day data)."""
    today = date.today()
    return today - timedelta(days=days), today - timedelta(days=1)
