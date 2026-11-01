# Architecture

## The problem this shape solves

Ten predictive modules written by ten different people, or by the same person on ten
different days, drift apart. One returns a float, another returns a dict, a third
prints a warning and returns None. The API layer grows a branch per module. Tests
grow a fixture per module. Adding the eleventh module means touching every layer.

The architecture here exists to prevent that. There is **one contract**, and every
module is a hole in the middle of it.

---

## Layers

```
                       +-------------------------------+
   HTTP / CLI  ---->   |  api/          cli/           |
                       |  auth, limits, errors, routing|
                       +---------------+---------------+
                                       |
                       +---------------v---------------+
                       |  orchestrator/                |
                       |  multi-module chains, partial |
                       |  results, degraded answers    |
                       +---------------+---------------+
                                       |
                       +---------------v---------------+
                       |  modules/<name>/              |
                       |  predict.py  train.py  eval   |
                       +---------------+---------------+
                                       |
                       +---------------v---------------+
                       |  common/                      |
                       |  BasePredictor, Registry,     |
                       |  training, tabular, config,   |
                       |  errors, schemas, logging     |
                       +---------------+---------------+
                                       |
                       +---------------v---------------+
                       |  features/                    |
                       |  physics (FAO-56), timeseries |
                       +-------------------------------+
```

Dependencies point **downward only**. `common/` never imports a module. `features/`
never imports a module. A module imports `common/` and `features/`. The API imports
modules only by name, dynamically. That is what makes modules independently
installable and independently testable.

---

## The predictor contract

`BasePredictor` is a template method. It owns everything that is the same in every
module, so a module author writes only what is genuinely different.

```
Predictor.load()
  |- resolve config          (configs/<name>.yaml, validated)
  |- resolve artefact        (registry LATEST, digest-checked)
  |- _load()                 <-- MODULE IMPLEMENTS: deserialise into memory
  |- mark ready / failed

Predictor.predict(request)
  |- reject if not ready     (typed error, not an empty answer)
  |- request_to_frame()      <-- MODULE MAY OVERRIDE
  |- _predict(request)       <-- MODULE IMPLEMENTS: return a Prediction
  |- stamp model version
  |- stamp latency
  |- log one structured line
```

The two hooks a module must implement are `_load()` and `_predict()`. Everything
else - config loading, artefact discovery, readiness, error translation, timing,
logging - is inherited and therefore identical across all ten modules.

Optional hooks:

| Hook | Default | Used by |
|---|---|---|
| `request_to_frame(request)` | builds a one-row DataFrame from declared features | tabular modules |
| `requires_artifact` | True | `irrigation`, `weather` set False - they compute |
| `request_model` / `response_model` | None | every module, so the API can validate generically |

### Why lazy loading matters

`import agri.modules.anything` must not load a model, open a file, or touch the
network. The `__init__.py` uses PEP 562 module-level `__getattr__` to resolve
`Predictor`, `Request` and `Response` on first access. This is what lets the API
import all ten modules at startup and still boot in under a second, and what lets a
test import one module without paying for the other nine.

---

## The training/serving seam

This is the single most common source of production ML bugs: the notebook one-hots a
column, the service forgets to, and the model silently receives the wrong features in
the wrong order.

The fix is structural. A module ships a **bundle**, not a bare model:

```python
TabularBundle(
    model,                  # the fitted estimator
    feature_pipeline,       # the FITTED feature transformer
    feature_columns,        # the exact column order at fit time
    classes,                # label vocabulary, for classification
    target_transform,       # e.g. log1p on a skewed target
    residual_std,           # for prediction intervals
    feature_importances,    # for the reasons field
)
```

`build_feature_pipeline()` and `prepare_features()` are **classmethods shared by the
trainer and the server**. There is no second implementation to drift. At serve time
the pipeline is replayed, never refitted - refitting at serve time would recompute a
vocabulary from one row and destroy the encoding.

```
train:  raw -> prepare_features -> build_feature_pipeline -> fit -> bundle -> registry
serve:  request -> prepare_features -> bundle.feature_pipeline.transform -> predict
                     ^ same code path ^
```

---

## The registry

```
models/registry/<module>/<version>/
    model.joblib | bundle.joblib     the artefact
    manifest.json                    everything needed to reproduce it
models/registry/<module>/LATEST      a pointer file
```

The manifest is the audit trail:

```json
{
  "name": "yield_prediction", "version": "1.0.0",
  "trained_at": "2026-...", "git_sha": "...",
  "dataset": "crop_yield.csv",
  "data_digest": "sha256:...",       "config_digest": "sha256:...",
  "artifact_file": "bundle.joblib",  "artifact_sha256": "...",
  "python": "3.13.16", "platform": "...",
  "libraries": {"scikit-learn": "1.9.1", "xgboost": "3.4.1"},
  "metrics": {"r2": 0.9698},
  "split": {...}, "n_train": 15751, "n_test": 3938,
  "baseline_metrics": {"r2": 0.9},
  "gate": {"passed": true, "checks": [...]},
  "feature_columns": [...],
  "classes": null
}
```

**`data_digest` is enforced, not decorative.** `evaluate_module` refuses to score an
artefact when the dataset on disk no longer matches the digest recorded at training
time. A metric computed on different data than the model was fitted on is not a
metric, it is a coincidence.

---

## Configuration

One YAML per module. Validated on load by pydantic, so a misspelled key fails at
startup naming the field rather than silently taking a default at 3am.

Sections: `data` (path, columns, dtypes, ranges, field map), `split` (strategy and
boundaries), `model` (algorithm, params, tuning grid), `gates` (primary metric and
floors), `serving` (warm-up, timeouts), `task`, `version`, `seed`.

Split strategy is per module because the correct split depends on the data:

| Strategy | Used by | Why |
|---|---|---|
| `random` | crop_recommendation, fertilizer, soil_nutrient | i.i.d. rows |
| `time` | yield_prediction | rows are years; a random split leaks the future |
| `group` | soil_image | must not put the same source image in train and test |
| `none` | irrigation, weather | no fitting; physics + live data |

---

## Errors

One taxonomy, in `common/errors.py`. Every failure is an `AgriError` subclass
carrying `code`, `message`, `http_status` and `context`:

| Error | HTTP | When |
|---|---|---|
| `ConfigError` | 500 | config missing or invalid |
| `DataValidationError` | 422 | input fails schema |
| `ModelNotTrainedError` | 503 | no registered artefact |
| `ArtifactError` | 500 | artefact corrupt or digest mismatch |
| `ExternalServiceError` | 502 | upstream provider failed |
| `BudgetExceeded` | 504 | inference exceeded its time budget |

`except Exception` appears nowhere in module code. Catching broadly hides bugs, so
the only broad catch is the last-resort HTTP handler, which logs the real detail and
returns a generic message plus the request id.

---

## Observability

* **Logging** - structured JSON, one line per event. `LogRecordSafe` exists because
  `logging` raises on any `extra` key that collides with a `LogRecord` attribute
  (`module`, `name`, `msg` are all real collisions). The subclass diverts colliding
  keys into a side field instead of exploding.
* **Request correlation** - `X-Request-ID` is accepted from the caller or generated,
  attached to every log line in the request, and echoed back.
* **Prediction journal** - one JSON line per served prediction at
  `reports/logs/predictions.jsonl`. Image payloads are summarised by length, never
  written whole.
* **Drift** - PSI against the training distribution. Reports `unknown` when no
  baseline can be built, rather than reporting stability it cannot substantiate.

---

## The orchestrator

`/v1/predict/full-recommendation` composes several modules. The rule is **partial
results beat no result**, as long as the caller can see what is missing.

```
weather, market_price, risk   -> run in parallel (network-bound, independent)
crop_recommendation           -> then
irrigation                    -> always, with weather-derived rain carried forward

response: { complete, signals_available, signals_missing, problems, advisory, latency_ms }
```

A module failure is captured into `problems` with its error code and the chain
continues. Only the anchor - irrigation or crop recommendation - produces a hard
failure, and even then the response explains why rather than returning a bare 500.

---

## What was deliberately not built

* **No database.** Events append to JSONL. Right for a single-node deployment, wrong
  for multi-node - stated in the module docs rather than implied.
* **No Prophet.** Named in the original PRD; not installed and not a core dependency.
  The price model is a documented, testable log-linear trend plus weekday seasonality.
  See [DECISIONS.md](DECISIONS.md).
* **No deep learning for tabular data.** On datasets of 99 to 19,689 rows, gradient
  boosting beats a neural network, trains in seconds, and can report feature
  importances. Torch is an optional extra, used only for the image modules.
* **No shared rate-limit store.** The token bucket is per process. A real global limit
  needs Redis, which would be a dependency with no other purpose here.
