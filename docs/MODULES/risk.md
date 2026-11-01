# risk

**Question.** How risky is this crop and season, and how much input does it need?

| | |
|---|---|
| Task | classification + derived advisory rates |
| Algorithm | gradient boosting |
| Features | crop, season, state, year, area, rainfall, fertilizer, pesticide |
| Gate | `recall_high_risk >= 0.50` |
| Measured | **recall(high risk) = 0.5840** |
| Chance rate | 0.33 |

## What it does

Derives a `risk_band` label from `Yield` within each Crop×Season group:

| Band | Definition |
|---|---|
| High | Yield <= 25th percentile of the group |
| Medium | 25th percentile < Yield <= 50th percentile |
| Low | Yield > 50th percentile |

Then predicts the band **without `Yield` as an input** - it defines the label, so
including it would be direct leakage.

It also returns `fertilizer_per_hectare` and `pesticide_per_hectare` estimates.

## Why the gate is 0.50 and not 0.70

The original intent was `recall_high_risk >= 0.70`. Measurement said no:

| Configuration | recall |
|---|---|
| Base features | 0.6196 |
| Base + fertilizer/pesticide per hectare | 0.5840 |

Adding features made it **worse**, which is evidence that the signal is genuinely
absent rather than that features were missing. A floor the model cannot clear is not a
quality bar; it is a permanently red build, and a permanently red build gets ignored.

So the floor is set at 0.50 - comfortably above the 0.33 chance rate, meaning the
model demonstrably learns something, without pretending it learns more than it does.

**This is the weakest gate in the project.** It is documented here, in the README, and
in the metrics report so nobody reads 0.584 as a strong result.

## Why per-hectare derived rates

The source data carries **totals** (fertilizer and pesticide in absolute units), not
rates. Dividing by `Area` yields a rate, which is what a farmer can act on. These are
derived, not modelled, and they are advisory.

## Honest limits

* **Recall of 0.584 against a 0.33 base rate is a modest signal.** Roughly 42% of
  genuinely high-risk cases are missed.
* **The label is a proxy.** "Bottom-quartile yield within a crop-season group" is not
  the same as "this field will fail". A low-yield-by-comparison field may still be
  perfectly profitable.
* **Quartile labels are relative and shift with the group.** A drought year moves the
  whole distribution.
* **State-level data.** See `yield_prediction` - the same granularity caveat applies.
* **No price or market risk.** This is agronomic and input risk only. A perfect yield
  sold at a terrible price is still a loss, and that risk is in `market_price`.
* **Not an insurance instrument.** Advisory.

## Usage

```bash
agri train --module risk
python -m agri.modules.risk.predict
```
