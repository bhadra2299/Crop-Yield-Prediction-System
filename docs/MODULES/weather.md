# weather

**Question.** What is the weather doing to my water balance right now?

| | |
|---|---|
| Task | live data retrieval + FAO-56 reference ET |
| Sources | OpenWeather (current + 3-hourly forecast), NASA POWER (history) |
| Artefact | **none** |
| Auth | `AGRI_OPENWEATHER_API_KEY` (optional) |
| Gate | ET0 plausibility checks |

## What it does

Fetches current conditions and forecast, aggregates three-hourly readings to daily,
and computes reference evapotranspiration for each day via the same FAO-56 engine the
`irrigation` module uses. Returns today's ET0 as `value`, a per-day forecast, the
`source` actually used, and any `warnings`.

The irrigation chain consumes this so a user does not have to supply weather by hand.

## Design decisions

**Kelvin to Celsius at the boundary.** OpenWeather returns Kelvin. Conversion happens
in the provider, so no downstream code has to remember which unit it holds.

**Three-hourly to daily in the provider.** The API returns 3-hourly data; the rest of
the system thinks in days. Aggregation is done once, at the edge, and the daily
`t_max` / `t_min` / `humidity_mean` / `rainfall` totals are what everything else sees.

**NASA POWER for history.** `-999` is the POWER fill value and is treated as missing,
not as a measurement of minus nine hundred and ninety-nine.

**Graceful degradation.** A history-fetch failure does not fail the request. The
response comes back with the forecast it could get and a warning naming what was
missing. A weather lookup should degrade, not 502.

**Responses are cached.** Both providers cache to `data/cache/` with a bounded TTL,
so a burst of requests does not become a burst of upstream calls.

**401 is not retried.** An invalid API key is a configuration error. Retrying it three
times just makes the user wait three times as long for the same failure. It raises
`ExternalServiceError` immediately, and the key value is never echoed into the error
or the logs.

## Without an API key

The module still loads and still answers. `train` reports
`openweather_key_configured: false` with an explicit warning, and a request returns a
typed error naming the missing configuration rather than a stack trace.

## Honest limits

* **ET0 is a reference value**, not measured evaporation. It is what FAO-56 defines
  it to be: the ET from a well-watered grass reference surface. Crop ET is derived
  through `Kc`, which lives in the irrigation module.
* **Point weather, not field weather.** A grid or station value near the coordinates,
  not an on-farm measurement. Local microclimate can differ substantially.
* **Forecast accuracy degrades with horizon.** The provider caps at 7 days; beyond
  that the module says so rather than extrapolating.
* **Rate limits are the provider's.** Free-tier quotas apply and are not managed here.
* **The NASA POWER history may lag.** Recent days can be missing; that is reported as
  a warning, not silently filled.

## Usage

```bash
python -m agri.modules.weather.predict
```

```bash
curl -X POST localhost:8000/v1/predict/weather \
  -H 'Content-Type: application/json' \
  -d '{"latitude":17.385,"longitude":78.4867}'
```
