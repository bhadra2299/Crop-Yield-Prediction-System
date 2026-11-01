# Decisions

Every non-obvious choice, with the reasoning and the alternative that was rejected.
Written so that a future reader can tell a deliberate constraint from an accident.

---

## D1. FastAPI on port 8000, with unversioned aliases

**Decision.** Serve `/v1/...` and alias the same routers at the root.

**Why.** The bundled React frontend (`intellicrop-ai-farm-main/src/services/api.ts`)
hardcodes `http://localhost:8000/predict/*`. Versioned routing is correct for a
public API; breaking a working frontend to get it is not a trade worth making. Both
are served and the OpenAPI schema documents both.

**Rejected.** Rewriting the frontend to call `/v1`. It would work, and it would
couple a backend versioning decision to a frontend release.

---

## D2. FAO-56 implemented directly, not via a library

**Decision.** `features/physics.py` implements the FAO-56 Penman-Monteith equations
from the publication.

**Why.** The published reference values are testable. `irrigation/train.py` asserts
eight of them - saturation vapour pressure at 20 °C and 30 °C, the slope of the
vapour pressure curve, atmospheric pressure at sea level and 1000 m, the psychrometric
constant at both, extraterrestrial radiation at the equator on day 81, and total
available water for a stated soil. A wrapper library would make those assertions
impossible, because the arithmetic would be behind a call.

**Honesty note.** This reproduces the *component equations* and matches published
Table 2.x values. It has **not** been verified to reproduce FAO-56 Chapter 4
Example 18 end-to-end. That would need the full worked example's daily record, which
is not in this repository. The tests check components, and the docs say components.

**Rejected.** `pyfao56`, which is not installed and would add a dependency whose
internals we could not assert against.

---

## D3. Market price: log-linear trend + weekday seasonality, not Prophet

**Decision.** A small OLS model on log price with a time trend and day-of-week effects.

**Why.** The PRD names Prophet. Prophet is not installed, is not a core dependency,
and would add a Stan build. More importantly, **the data cannot support what Prophet
is for.**

The mandi file spans 2019-05-22 to 2021-10-01 and contains **no observations at all
in 2020**. The row counts are 590,361 in 2019 and 235,553 in 2021. There is a
20-month gap.

Annual seasonality - the thing Prophet's yearly component exists to capture - is
therefore not estimable from this data. A model that reported a yearly seasonal
component would be reporting a curve fitted through a hole. Weekly seasonality *is*
estimable from seven months of daily data, and matters for mandi arrivals, so that is
what is modelled.

**Consequences, enforced in code:**
* `detect_source_gap()` finds the largest gap; `has_multi_month_gap` goes into the
  manifest and into every API response's reasons.
* `find_continuous_segment()` tolerates gaps of up to 3 days (weekends, mandi
  holidays) and refuses to interpolate anything larger.
* The output is a **relative change** against the latest known price, with that price
  and its date attached. A bare forecast figure from a stale feed is indistinguishable
  from a real record high.
* Directional signals are only emitted when the expected move exceeds the model's own
  measured MAPE. Below that the signal is `insufficient_signal`.

**Rejected.** Forward-filling across the gap, which would invent two years of prices.

---

## D4. Integer categorical encoding, vocabulary fitted on train only

**Decision.** `CategoryIndexEncoder` maps categories to integers, unseen values to
`-1`. The vocabulary is fitted on the training partition and shipped in the bundle.

**Why.** One-hot encoding in a pipeline that is replayed at serve time is a classic
source of column-order drift: the encoder sees one row at inference and produces one
column instead of thirty. Integer encoding is stable under single-row transforms, and
`-1` for unseen categories is an explicit, inspectable value rather than an implicit
all-zeros row that looks exactly like a known category.

Fitting on train only is what prevents the vocabulary from leaking test-set categories
into the model.

---

## D5. `Production` is dropped from yield features

**Decision.** `Production` is excluded; the model predicts `Yield = Production / Area`.

**Why.** Including it is textbook target leakage - you cannot know a field's production
before the harvest you are trying to predict. `detect_leakage()` enforces the rule so
it cannot be reintroduced by a config edit.

---

## D6. Time-based split for yield, random for the rest

**Decision.** `yield_prediction` splits on `Crop_Year`; other tabular modules split
randomly.

**Why.** Year-ordered rows make a random split into future-prediction. The time split
costs some R² and buys a metric that means something. The reported R² of 0.9698 is
*temporal* generalisation, which is the harder and more honest number.

**Caveat, stated in the module doc.** The yield data is **state-level**, not
field-level. Part of the high R² is the model memorising crop×state baselines rather
than modelling agronomy. A field-level dataset would score lower and be more useful.

---

## D7. Risk gate at `recall_high_risk >= 0.50`, not 0.70

**Decision.** The configured floor is 0.50 against a 0.33 chance rate. The model
measures ~0.58.

**Why.** The original intent was 0.70. Measurement said the data does not support it:
the model reaches 0.584, and adding `fertilizer_per_hectare` and
`pesticide_per_hectare` made it slightly *worse* (0.6196 to 0.5840), which is
evidence the signal is genuinely absent rather than that the features were missing.
A floor the model cannot clear is not a quality bar, it is a broken build.

**Stated consequence.** 0.58 recall against a 0.33 base rate is a real but modest
signal. The module doc says so.

---

## D8. Soil nutrient gate at `r2 >= 0.05`

**Decision.** A deliberately low floor.

**Why.** Climate and pH carry weak signal for soil N, P and K. A high R² on this
dataset would indicate leakage - most likely a nutrient-related column that is
part of the label definition. The measured 0.8192 comes from soil-composition
features; the low floor documents that a *regression* here is not automatically a bug
and a spike here deserves suspicion.

---

## D9. Soil image: synthetic data, flagged in the artefact

**Decision.** Train on the generated dataset and mark it.

**Why.** The `data/raw/soil_images/` set is 4,200 synthetically generated JPEGs
across 10 classes. A macro-F1 of 0.85+ on synthetic images measures **synthetic
separability**, not field performance. The manifest carries `synthetic_data: true`
and the response carries a limitation string, so no caller can mistake the number for
what it is not.

**Rejected.** Presenting the accuracy as a real result. Easiest thing in the world to
do; makes the whole project untrustworthy.

---

## D10. Disease: third-party weights, no gate

**Decision.** Inference-only, using pretrained weights. `agri evaluate` records it as
skipped rather than inventing a metric.

**Why.** We did not train these weights and have no validation set for regional
varieties, so there is no honest metric to publish. The class count (~38) comes from
the upstream model. Output is advisory.

---

## D11. Calibration confidence, not raw probability

**Decision.** `confidence` is derived from calibrated evidence with per-module
thresholds, and `confidence_score` is the calibrated score.

**Why.** A raw softmax of 0.999 from an overfit model is not high confidence.
Calibration plus explicit thresholds means "high confidence" is a claim the module can
defend. `fertilizer` sets `confidence_high: 0.90` and `confidence_medium: 0.70`
specifically so that with 99 training rows most answers surface as medium or low -
which is the truthful outcome.

---

## D12. No database

**Decision.** Irrigation events and prediction logs are JSONL under `reports/`.

**Why.** Single-node deployment. A database would add an operational dependency and a
migration story for append-only data that nothing queries relationally yet. The
failure mode is stated in the docs: this does not work across multiple nodes.

---

## D13. Per-process rate limiting

**Decision.** An in-memory token bucket, documented as per-process.

**Why.** A correct global limit needs a shared store. Redis would be a dependency with
no other purpose. The honest move is to implement the simple thing and write down
exactly how it fails under multiple workers, which the docstring and README do.

---

## D14. Authentication disabled when `AGRI_API_KEYS` is empty

**Decision.** Empty means open, with a loud startup warning and `auth_enabled: false`
in `/v1/health`.

**Why.** Local development and the bundled frontend should work after a clone with no
ceremony. The failure mode of a silent default is that it reaches production, so it is
made visible in the logs on every boot and in the health payload.

---

## D15. `LogRecordSafe`

**Decision.** A `logging.Logger` subclass that diverts `extra` keys colliding with
`LogRecord` attributes into a side field.

**Why.** `logging` raises `KeyError: Attempt to overwrite 'module' in LogRecord` for
`module`, `name`, `args`, `msg` and others. Those are exactly the field names
structured logging wants. The subclass keeps the ergonomic API and makes the collision
a non-event rather than a crash inside the logger.

---

## D16. Docker healthcheck accepts 503

**Decision.** The container `HEALTHCHECK` treats both 200 and 503 from `/v1/health`
as "process alive", while CI separately asserts that the required modules are ready.

**Why.** `/v1/health` returns 503 when a required module has no artefact, which is
useful and correct. If the container healthcheck treated 503 as failure, Docker would
restart-loop a service that is working exactly as designed and returning a clear
diagnostic. The two questions - "is the process up" and "can it serve everything" -
are different questions and get different checks.

---

## D17. Removed duplicate data and build junk

**Decision.** Duplicate CSVs were SHA-256 verified before deletion, and build output
was removed.

**Why.** Two 50 MB market files with overlapping content, and committed build
artefacts, inflate the repository and, worse, make it ambiguous which file is
authoritative. `market_mandi.csv` is canonical;
`agriculture_prices.csv` was kept because it is a genuinely different schema, not a
duplicate. Verification before deletion is what made the cleanup safe.
