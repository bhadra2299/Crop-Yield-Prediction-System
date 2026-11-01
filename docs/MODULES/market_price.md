# market_price

**Question:** what will the mandi pay for this commodity in this state, relative to the
last price we actually observed?

**Status:** shipped, serving the **naive last-value baseline**. The trend model was
built, measured, and **rejected**. This page explains why, because the rejection is a
result and not a failure.

---

## Contract

```python
from agri.modules.market_price import Predictor, Request

predictor = Predictor()
predictor.load()
response = predictor.predict(Request(commodity="Wheat", state="Punjab", horizon_days=7))
```

| Field | Meaning |
|---|---|
| `value` | forecast modal price, rupees per quintal |
| `forecast_price` | same value, explicit |
| `relative_change` | fractional change versus the latest observed price |
| `signal` | `up`, `down`, or `insufficient_signal` |
| `backtested_mape` | measured mean error at the nearest backtested horizon |
| `backtested_median_ape` | measured **typical** error, which is the number to trust |
| `baseline_mape`, `baseline_median_ape` | the same statistics for doing nothing |
| `directional_accuracy` | fraction of windows where up-versus-down was called correctly |

Served horizons are **1, 7 and 30 days**. Any other horizon snaps to the nearest
backtested one, and the response says which.

---

## What the data is

`data/raw/market_price/market_mandi.csv`, 836 977 rows, 232 commodities, May 2019 to
October 2021.

It has a problem that dominates everything else on this page:

| Property | Value |
|---|---|
| Earliest date | 2019-05-22 |
| Latest date | 2021-10-01 |
| Distinct days with any observation | 341 |
| **Largest gap** | **426 days** (2019-12-31 to 2021-03-01) |
| Rows in 2020 | **0** |

The file spans two and a half years but contains **less than one year of actual days**,
with a 14-month hole in the middle. That has two consequences:

1. **Annual seasonality is not estimable.** A yearly cycle cannot be fitted across a
   gap that is itself longer than a year. `annual_seasonality` is therefore `false`,
   and `detect_source_gap()` reports the gap in every training run and manifest.
2. **No interpolation.** The gap is not filled, smoothed or forward-filled. Filling a
   14-month hole and then training on it would manufacture the seasonality the model
   would then appear to find.

---

## What was built

A damped local log-linear trend with weekday effects:

```
log(price) = level + slope * t + weekday_effect[day_of_week]
forecast(h) = level + slope * sum(damping^k for k in 1..h)
```

fitted on the most recent 60 observations (`trend_window_days: 60`) with damping 0.3
(`trend_damping`), so a trend is not extrapolated at full strength indefinitely.

Rejected alternatives, all measured on the same rolling-origin backtest:

| Model | h7 result |
|---|---|
| **naive last value** | **median APE 20.0%, wins on 87% of series** |
| damped local trend, window 60, damping 0.3 | median APE 33.4%, wins on 13% |
| moving average 7 / 14 | materially worse at h7 and h30 |
| global trend over the whole segment | worse than naive at every horizon |
| Prophet | not installed, not a core dependency, and the yearly component is **not estimable** from this data (decision D3) |

---

## How it was measured, and the mistake that was caught

**Rolling-origin backtest.** For each series: fit on an expanding window, forecast
h days, score, step forward 5 days, repeat. Time order is never broken and no future
observation is ever visible to a forecast. This is why `split.strategy` is `none` for
this module — a random split on a time series is a leak, not a validation.

The first attempt aggregated error as a **pooled mean MAPE across series** and reported
a +31% lift over naive at h7. That number was an artefact. Two runs on different series
sets:

| Series set | pooled mean MAPE | pooled mean baseline | apparent lift |
|---|---|---|---|
| top 45 | 46.4% | 67.2% | **+31%** |
| top 120 | 46.9% | 46.9% | **0%** |

The model's own error barely moved while the apparent lift collapsed. The mean was
being carried by a handful of highly volatile series whose MAPE exceeds 100%. **A
metric that swings 31 points from a change of sample is not a metric**, so the
aggregation was replaced:

* headline: **median** of per-series APE
* **`beat_fraction`**: the share of series where the model actually beats naive

On those robust statistics the trend model loses decisively — median APE 33.4% against
20.0%, winning 16 of 120 series. Its *mean* error happened to be slightly better because
it avoids a few catastrophic misses while being worse in the typical case. Reporting
only the mean would have shipped a model that is worse for most farmers.

**Therefore the module serves the baseline**, and `train.py` performs this selection
automatically: fit the trend, backtest it, serve it only if `beat_fraction` clears
`selection_beat_fraction` (0.60). It measures 0.133, so the baseline is served and the
reason is recorded in the manifest and repeated in every response's `reasons`.

---

## The gate

| | |
|---|---|
| Primary metric | `median_ape` |
| Bound | `maximum: 25.0` (measured naive 20.0% plus headroom) |
| `must_beat_baseline` | false — the lift metric is unstable here, see above |
| v1.0.0 measured | **20.0009%** → passes |

The gate deliberately does **not** require beating a baseline, because the shipped model
*is* the baseline. Its job is the guardrail that actually caught a real regression: it
refuses a forecaster materially worse than doing nothing. An earlier build promoted a
trend model that failed this bar — `train.py` now refuses to register any artefact
whose gate fails, and that defect is documented in code.

Secondary metrics (`mape`, `baseline_mape`, `beat_fraction`,
`relative_lift_over_baseline`) are recorded but not floored. Flooring `beat_fraction`
would be flooring a tautology, since the baseline scores 0 against itself.

---

## Honest limits

1. **This module currently provides no directional edge.** It reports the last observed
   price and says `insufficient_signal`. That is the truthful answer on this data, and
   it is more useful than a confident-looking number with no evidence behind it.
2. **`confidence` is never `HIGH`.** It is capped at `MEDIUM` and no horizon reaches
   it, because measured directional accuracy does not support a stronger claim.
3. **The feed is stale.** The last observation is 2021-10-01. Output is framed as a
   *relative change* against the latest dated price, never as an absolute expectation,
   so a stale feed cannot present an old price as today's market.
4. **Most series are thin.** 232 commodities, many with a handful of observations and
   long gaps; only ranks with at least 60 usable observations are fitted at all.
5. **No mandi, district or arrival-volume features.** Those are in the file but a
   per-mandi model would have far too few rows per group to fit honestly.
6. **A 426-day hole is not repaired**, so nothing here generalises across the missing
   period. A model trained on 2019 need not hold in 2021.

---

## Reproduce

```bash
agri train --module market_price     # fits, backtests, selects, registers
agri evaluate --module market_price  # re-scores the registered artefact
```

The manifest records `served_forecaster`, `selection_reason`, `trend_backtest` (the
rejected model's full evidence), `data_digest` and `git_sha`, so the rejection can be
re-derived rather than trusted.
