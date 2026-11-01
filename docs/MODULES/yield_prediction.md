# yield_prediction

**Question.** How much will this crop yield per hectare?

| | |
|---|---|
| Task | regression |
| Algorithm | gradient boosting |
| Dataset | `data/raw/yield/crop_yield.csv`, 19,689 rows |
| Features | Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide |
| Target | Yield (tonnes per hectare) |
| Split | **time-based on Crop_Year**, train <= 2014, test >= 2015 |
| Gate | `r2 >= 0.85` |
| Measured | **R² = 0.9698** |

## The leakage rule

`Production` is **excluded**, and `detect_leakage()` enforces that it stays excluded.

The target is `Yield = Production / Area`. Including `Production` as a feature would
let the model read the answer off the input, producing a beautiful R² and a model that
is useless, because nobody knows a field's production before the harvest they are
trying to forecast.

## Why a time split

A random split on year-ordered rows lets the model train on 2015 and be tested on
2012. For agricultural yield - which trends with technology, seed varieties and
policy - that is forecasting the past with knowledge of the future.

The time split costs R² and buys a number that means something: 0.9698 is *temporal*
generalisation to unseen years.

## Honest limits

* **The data is state-level, not field-level.** Rows are state aggregates. Part of
  the high R² is the model memorising crop×state baselines rather than modelling
  agronomy. A field-level dataset would score lower and be far more useful.
* **`Area` and `Fertilizer` are totals, not per-hectare rates.** Their coefficients
  therefore reflect scale as much as practice.
* **State granularity cannot express within-state variation.** A farmer in a
  high-rainfall district of a dry state gets the state average.
* **No weather for the growing season.** Only annual rainfall. Drought timing, which
  often matters more than the total, is invisible.
* Predictions are **expected yield under average management**, not a guarantee.

## Usage

```bash
agri train --module yield_prediction
python -m agri.modules.yield_prediction.predict
```
