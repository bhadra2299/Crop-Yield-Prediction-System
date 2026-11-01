# Smart Crop AI

An agricultural decision platform: ten independent models behind one versioned API,
each answering one question a farmer or agronomist actually asks.

| Question | Module |
|---|---|
| Which crop suits this soil and climate? | `crop_recommendation` |
| How much will this field yield? | `yield_prediction` |
| Which fertilizer blend, and how much? | `fertilizer` |
| What are the soil's N, P and K levels? | `soil_nutrient` |
| Should I irrigate today, and how much? | `irrigation` |
| What is the weather doing to my water balance? | `weather` |
| What price will the mandi pay? | `market_price` |
| How risky is this crop and season? | `risk` |
| What soil type is in this photo? | `soil_image` |
| What is wrong with this leaf? | `disease` |

---

## Read this before you trust a number

This project is built so that a confidence score means something and a limitation is
stated where it applies. The honest summary is:

* **Six modules are trained on real public agricultural data** and are evaluated with
  gates that fail the build when a metric regresses.
* **The irrigation engine is a published physics calculation** (FAO-56), checked
  against published reference values. It has **no field-truth accuracy figure**,
  because no field sensor series exists to compute one from. It is a water-balance
  calculator, not a validated predictor.
* **The soil image model is trained on synthetic images.** Its accuracy measures how
  well it separates *our synthetic classes*. It is a demonstration, and every
  response says so.
* **The disease model uses third-party pretrained weights** (PlantVillage-style,
  ~38 classes). Those weights were not validated on regional varieties.
* **The market price model serves the naive baseline, on purpose.** The source file has
  a 20-month hole with no observations at all, and when a trend model was fitted it
  *measurably lost* to simply repeating the last observed price (median absolute
  percentage error 33.4% versus 20.0%, winning on only 13% of series). So the trend model
  was rejected and the module repeats the last price, reports no directional signal, and
  never returns high confidence. See `docs/MODULES/market_price.md`.
* **The fertilizer dataset has 99 rows.** Its thresholds are set so most answers come
  back as medium or low confidence, which is the correct behaviour for 99 rows.

Where a number is weak, the response says so in its `reasons` field rather than
presenting a clean figure. That is a design decision, not an oversight.

---

## Quick start

Requires Python 3.11+ (developed on 3.13).

```bash
git clone <repo> && cd "Crop Yeild Prediction"

python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"

cp .env.example .env            # then edit; every value is optional for local work

agri doctor                     # environment, data, registry, secrets
agri modules                    # what is configured
agri data --all                 # validate every input file
```

Train and serve:

```bash
agri train --module crop_recommendation
agri train --module market_price --max-series 120

agri health                     # per-module readiness, no server needed
agri serve --port 8000
```

Then:

```bash
curl -s localhost:8000/v1/health | jq
curl -s -X POST localhost:8000/v1/predict/crop-recommendation \
  -H 'Content-Type: application/json' \
  -d '{"N":90,"P":42,"K":43,"temperature":20.88,"humidity":82,"ph":6.5,"rainfall":202.9}' | jq
```

Interactive docs are at `http://localhost:8000/docs`.

The bundled React frontend in `intellicrop-ai-farm-main/` calls
`http://localhost:8000/predict/*`, which this service serves unversioned as well
as under `/v1`. It works unchanged.

---

## How the code is organised

```
src/agri/
  common/        shared machinery - the contract every module obeys
    base.py        BasePredictor: config, artefact, timing, errors, logging
    training.py    train_module / evaluate_module / tuning / calibration
    tabular.py     TabularPredictor + Bundle: one code path for train and serve
    registry.py    versioned artefacts with manifests and data digests
    config.py      typed configuration, validated on load
    errors.py      the AgriError taxonomy and its HTTP mapping
    logging.py     structured JSON logs that survive LogRecord collisions
    schemas.py     Prediction envelope, Confidence, Interval, HealthStatus
  features/
    physics.py     the FAO-56 engine
    timeseries.py  log-linear trend + weekday seasonality, rolling-origin backtest
  modules/<name>/
    __init__.py    lazy exports (PEP 562)
    predict.py     Request, Response, Predictor, SAMPLE
    train.py       fitting + registration
    evaluate.py    metrics + gate enforcement
  api/             FastAPI app, routers, auth, rate limiting, error mapping
  orchestrator/    the multi-module advisory chain
  monitoring/      prediction logging, PSI drift, data quality
  cli/             the agri command
configs/           one YAML per module - the single source of truth
tests/             unit, integration, regression, fixtures
docker/            Dockerfile + compose
docs/              architecture, per-module notes, decisions
data/raw/          input datasets (not committed)
reports/           metrics, logs, events (not committed)
models/registry/   versioned artefacts (not committed)
```

Two rules explain most of the design:

1. **Training and serving run the same code.** A module does not re-implement feature
   engineering at inference time. It ships the fitted pipeline inside its artefact and
   replays it, which is what makes a training metric predictive of serving behaviour.
2. **Nothing is hardcoded.** Column names, algorithms, hyperparameters, gate
   thresholds and paths all come from `configs/<module>.yaml`.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture.

---

## The module contract

Every module exposes the same three things, so the API, the CLI, the tests and the
orchestrator never special-case a module:

```python
from agri.modules.crop_recommendation import Predictor, Request

predictor = Predictor()
predictor.load()                     # explicit, lazy, never at import time
response = predictor.predict(Request(N=90, P=42, K=43, ...))
print(response.value, response.confidence, response.reasons)
```

Every response is a `Prediction`:

| Field | Meaning |
|---|---|
| `value` | the answer |
| `confidence_score` | 0-1 where the model can produce one |
| `confidence` | high / medium / low, derived from calibrated evidence |
| `interval` | a prediction interval where the model supports one |
| `alternatives` | runner-up answers, so the top-1 is not over-trusted |
| `reasons` | why this answer, in words |
| `model_name`, `model_version` | exactly which artefact answered |
| `latency_ms` | how long it took |

A module that cannot answer raises a typed `AgriError` subclass, which the API maps
to a documented HTTP status. It never prints to stdout or returns a bare `None`.

---

## Command line

| Command | Purpose |
|---|---|
| `agri doctor` | environment, datasets, registry and secret configuration |
| `agri modules` | every configured module with its task, algorithm and gate |
| `agri data --all` | validate every input file against its declared schema |
| `agri train --module X` | fit and register a model |
| `agri train --all` | train everything trainable |
| `agri evaluate --all` | **re-score everything and exit 1 if any gate fails** |
| `agri health` | per-module readiness without starting a server |
| `agri serve` | run the API |
| `agri drift --module X` | PSI drift report against the training distribution |

`agri evaluate --all` is the CI gate. A non-zero exit blocks the merge.

---

## API

Versioned under `/v1`, with unversioned aliases for the bundled frontend.

**Prediction** - `POST /v1/predict/{module}` for each of the ten modules, plus
`POST /v1/predict/full-recommendation` which runs several and degrades per module.

**System** - `GET /v1/health` (503 when a required module is down), `GET /v1/ready/{module}`,
`GET /v1/models`, `GET /v1/metrics/{module}`, `GET /v1/version`.

**Ingest** - `POST /v1/ingest/irrigation-event`, `GET /v1/ingest/irrigation-events`.

Conventions:

* Errors are RFC 9457 problem documents carrying a stable `code`, never a bare 500
  with a stack trace.
* Every response carries `X-Request-ID` and `X-Response-Time-Ms`.
* Authentication is `X-API-Key` against `AGRI_API_KEYS`, and is **disabled when that
  variable is empty** - the service warns loudly at startup when that happens.
* Rate limiting is a per-process token bucket; behind N workers the effective limit is
  N times the configured value. This is stated rather than hidden.

---

## Configuration

`configs/<module>.yaml` is the single source of truth. It declares the data source
and columns, the split strategy, the model family and hyperparameters, the gate
thresholds, and the serving behaviour. Configs are validated on load, so a typo fails
at startup with a message naming the field.

Secrets are read from the environment only, never from a file in the repository.
See `.env.example` for the full list.

> **Rotate the OpenWeather key.** A live key is present in plaintext in
> `smart_crop_ai_PRD.md` and it currently authenticates. It must be revoked. The same
> applies to the Supabase credentials in `intellicrop-ai-farm-main/.env`. Neither file
> is committed, and both are listed in `.gitignore`.

---

## Testing

```bash
pytest -m "not network and not slow"   # what CI runs by default
pytest -m regression                   # metric floors on real data
pytest -m network                      # live provider calls, opt-in
ruff check src tests
```

The suite is split by intent: `unit` for pure logic, `integration` for the API and
registry round-trips, `regression` for published metric floors, `slow` and
`network` for things that should not run on every commit.

---

## Known limitations

These are stated here, in the module docs, and in the API responses.

| Area | Limitation |
|---|---|
| Irrigation | FAO-56 water balance, verified against published reference values. No field-truth accuracy exists. |
| Soil images | Trained on 4 200 procedurally generated images. The 0.9362 macro F1 measures separability of synthetic textures, not field accuracy. |
| Disease | Third-party weights, ~38 classes, not validated on regional varieties. |
| Market price | Source has no 2020 data. The fitted trend model lost to the naive baseline and was rejected: the module repeats the last price, reports no signal, and never claims high confidence. |
| Fertilizer | 99 training rows. Advisory only; thresholds set so most answers are not high confidence. |
| Yield | State-level, not field-level. High R² is partly crop×state structure. |
| Soil nutrient | Climate and pH carry weak signal; a high R² here would indicate leakage, not skill. |
| Risk | The high-risk recall floor is 0.50 against a 0.33 chance rate. A 0.70 floor was intended and the data does not support it. |

## Licence

See [LICENSE](LICENSE) if present; otherwise treat as proprietary.