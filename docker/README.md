# Containerisation - agri platform

Production image, local orchestration and the container smoke test used by CI for
the **Smart Crop AI** agri platform (FastAPI serving + tabular / vision ML modules).

```
docker/
  Dockerfile        multi-stage image: builder -> runtime -> vision -> production (default)
  compose.yaml      api service + one-shot "train" profile service
  .dockerignore     TEMPLATE ONLY - see "The .dockerignore caveat" below
  README.md         this file
```

---

## 1. Requirements

| Tool | Version | Why |
| --- | --- | --- |
| Docker Engine | 24+ (BuildKit on) | `# syntax=docker/dockerfile:1.7`, cache mounts |
| Docker Compose | 2.24+ | `env_file: {path, required: false}` long syntax |
| Python | 3.13 | image base; `pyproject.toml` allows >=3.11 |

**The build context must be the repository root and the Dockerfile path explicit:**

```bash
docker build --file docker/Dockerfile --target production -t agri-platform:latest .
```

Running `docker build docker/` from inside `docker/` does **not** work: the
Dockerfile needs `pyproject.toml`, `README.md`, `src/` and `configs/`, all of
which live in the repository root.

---

## 2. Building

### Default image (tabular modules only, **no torch**)

```bash
docker build --file docker/Dockerfile --target production -t agri-platform:latest .
```

`production` is an alias of the slim `runtime` stage and is the **last** stage,
so a bare `docker build --file docker/Dockerfile .` behaves identically. The
vision extra can therefore never leak into the default image.

### Vision image (adds `torch` + `torchvision`)

Only for the image-classification modules (`soil_image`, `disease`):

```bash
docker build --file docker/Dockerfile --target vision -t agri-platform:vision .
```

```bash
# Force CPU-only wheels explicitly (default) or point at another index:
docker build --file docker/Dockerfile --target vision \
  --build-arg TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cpu \
  -t agri-platform:vision .
```

The vision image is roughly 2 GB with CPU wheels versus ~7 GB with the CUDA
runtime, which is why it is a separate target and not part of the default build.

### Image design

| Aspect | Decision |
| --- | --- |
| Stages | `builder` (compilers) -> `runtime` (slim) -> `vision` -> `production` (default) |
| User | non-root `agri`, **uid/gid 10001**, `/usr/sbin/nologin` |
| Interpreter | `python:3.13-slim-bookworm`, venv at `/opt/venv` copied from the builder |
| Compilers | present only in `builder`; absent from the shipped image |
| PID 1 | `tini` - reaps zombies, forwards SIGTERM so uvicorn shuts down gracefully |
| Healthcheck | pure-stdlib Python probe of `/v1/health` (no curl in slim) |
| Secrets | **none**. All configuration arrives as `AGRI_*` environment variables at run time |

Environment baked into the runtime image:

```
PYTHONUNBUFFERED=1  PYTHONDONTWRITEBYTECODE=1
AGRI_ENV=production  AGRI_LOG_JSON=true
AGRI_PROJECT_ROOT=/app  AGRI_CONFIG_DIR=/app/configs
AGRI_MODEL_REGISTRY=/app/models/registry
AGRI_API_HOST=0.0.0.0  AGRI_API_PORT=8000
```

`AGRI_PROJECT_ROOT` matters: `agri.common.paths` locates the project root by
walking up from its own file until it finds `pyproject.toml`. In a normally
installed venv that file is not an ancestor, so the image pins the root
explicitly and ships `pyproject.toml` + `configs/` at `/app`.

---

## 3. Running

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  agri-platform:latest
```

Then:

```bash
curl -i http://localhost:8000/v1/health
```

### Health semantics - degraded is alive

The service **must boot and serve with no trained model present**. `/v1/health`
returns:

| Status | Meaning | Container health |
| --- | --- | --- |
| `200` | all configured modules loaded | healthy |
| `503` | **degraded** - registry empty or a module failed to load (`AGRI_STRICT_MODELS=false`) | **healthy** (still alive and serving) |
| no response / connection refused | process down | unhealthy |

The `HEALTHCHECK` accepts **200 and 503** and nothing else; it uses
`http.client` because `python:3.13-slim` has neither `curl` nor `wget`.
This is asserted in CI (job `docker`), which starts the container with **no
volumes at all** and requires a 200 or 503 answer.

`AGRI_STRICT_MODELS=true` flips the posture: the process refuses to start
without artefacts. Keep it `false` in production containers unless the
orchestrator handles the restart loop.

---

## 4. Compose

```bash
# api only
docker compose --file docker/compose.yaml up --build --detach
docker compose --file docker/compose.yaml logs --follow api
docker compose --file docker/compose.yaml down

# one-shot training run (profile "train")
docker compose --file docker/compose.yaml --profile train run --rm train
```

```text
services:
  api     long-running FastAPI service, published on 0.0.0.0:8000
  train   one-shot "agri train --all" behind the "train" profile;
          never started by "up", restart policy "no"
```

The `api` service inherits the image `HEALTHCHECK`; `docker compose ps` reports
`(healthy)` even while degraded, by design.

### Volumes

| Volume / mount | Container path | Mode | Contents |
| --- | --- | --- | --- |
| `agri_models_registry` (named) | `/app/models/registry` | rw | versioned bundles, manifests, `LATEST` pointers |
| `agri_reports` (named) | `/app/reports` | rw | `metrics/*.json` and `figures/*` written by train/evaluate |
| `../data/raw` (bind) | `/app/data/raw` | **ro** | raw datasets; a container can never mutate them |

Named volumes survive `compose down` and are only destroyed by
`compose down --volumes`. Both are shared by `api` and `train`, so a training
run becomes visible to a running api after the registry is re-read (restart the
api container to be certain).

### Configuration

Create `.env` from `.env.example` in the repository root. Compose reads it when
present (`required: false`), and it is never baked into an image layer. Anything
sensitive - `AGRI_API_KEYS`, `OPENWEATHER_API_KEY` - belongs in an orchestrator
secret store or `--env-file`, not in `compose.yaml`.

---

## 5. The .dockerignore caveat (important)

`.dockerignore` is read from the **build-context root**, never from the
Dockerfile's directory. With `--file docker/Dockerfile .` the daemon looks for
`<repo-root>/.dockerignore` and ignores `docker/.dockerignore`.

Creating `<repo-root>/.dockerignore` was outside this change-set's write scope
(`docker/` + `.github/`), so:

* `docker/.dockerignore` is a **template** carrying the full ignore list;
* the same list is in the header of `docker/Dockerfile`;
* `.github/DOCKERIGNORE.md` holds ready-to-paste content and the rationale.

**Consequence today:** builds are correct but slower - the working tree,
including `data/raw/` imagery and `.venv/`, is uploaded to the daemon. Nothing
undesirable is copied into the image because every `COPY` is explicit
(`pyproject.toml`, `README.md`, `src/`, `configs/`). `data/`, `models/` and
`.env` are never part of any layer.

**Fix:** `cp docker/.dockerignore .dockerignore` and commit it.

---

## 6. CI

`.github/workflows/ci.yml` runs on push and pull request to `main`:

| Job | Command | Gate |
| --- | --- | --- |
| `lint` | `ruff check` + `ruff format --check` | hard |
| `types` | `mypy src/agri --no-error-summary` | fails only on errors **above** `.github/mypy-baseline.txt` (69 today) |
| `test` | `pytest -m unit\|integration\|regression` with `PYTHONPATH=src` | hard; coverage XML uploaded per marker |
| `docker` | build `--target production`, boot with no volumes, probe `/v1/health` | hard; 200 **or** 503 passes |
| `metric-floors` | `agri evaluate --all` (console entry point) | hard; any non-zero exit blocks the merge |

Reproduce the container gate locally:

```bash
docker build --file docker/Dockerfile --target production -t agri-platform:ci .
docker run --detach --name agri-ci -p 127.0.0.1:8000:8000 agri-platform:ci
sleep 5 && curl -i http://127.0.0.1:8000/v1/health   # 200 or 503 expected
docker rm --force agri-ci
```

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `error: failed to solve ... pyproject.toml not found` | context was not the repo root; use `--file docker/Dockerfile .` |
| Container restarts immediately | `uvicorn agri.api.app:app` must be importable; check `docker logs` |
| Health stays `starting` | `start-period` is 25 s; the probe accepts 200/503, so a failure means the port is not answering |
| `PermissionError` writing to `/app/models/registry` | a bind mount overrode the image directory without uid 10001 ownership; use the named volume or `chown 10001:10001` |
| Build uploads gigabytes | the missing root `.dockerignore` (section 5) |
| Torch download is enormous | you built `--target vision`; leave it out for serving-only images |

---

## 8. Known gaps / assumptions

Verified against the working tree on 2026-10-28. Items 1 and 2 are defects in
`src/`, which is outside this containerisation change-set.

1. **`/v1/health` currently answers HTTP 500, not 200/503.** The service itself
   is resilient - with an empty registry it boots, logs 7 module failures, keeps
   serving, and only `irrigation` comes up ready - but the health endpoint
   raises, so the `docker` CI job (which requires 200 or 503) is **correctly
   red**. Root cause:
   `src/agri/api/routers/system.py:50` and `:64` build
   `ModuleHealth(...)` without the required `ready` field (and pass
   `trained=`, which is not a field; the field is `artifact_present`).
   Schema: `src/agri/common/schemas.py:83-92`; correct construction pattern:
   `src/agri/common/base.py:192-202`.
   Impact beyond CI: Kubernetes/Compose health checks treat HTTP 500 as
   unhealthy, so the container would be restarted despite being alive.
2. **`python -m agri.cli ...` cannot work.** `src/agri/cli/` is a package with
   no `__main__.py`; CPython refuses it (*"No module named agri.cli.__main__"*).
   The console entry point declared in `pyproject.toml`
   (`agri = "agri.cli:main"`) does work, and `evaluate` accepts `--all`, so CI
   calls `agri evaluate --all`. The compose `train` service does the same
   (`agri train --all`). No `src/` change is needed for either, but note that
   two **Makefile** targets still use the broken form
   (`make data` -> `python -m agri.cli data validate`, `make train-all` ->
   `python -m agri.cli train --all`); those need `agri ...` or a new
   `src/agri/cli/__main__.py` (outside this change-set).
3. **The metric gate needs artefacts that git does not carry.** `agri evaluate
   --all` only evaluates modules with a registered artefact and exits 1 when
   there are none; `models/registry/` and `data/raw/` are gitignored. CI has a
   preflight that fails with that diagnosis. Provision them (`dvc pull`,
   restored cache, release artefact) for the gate to be meaningful.
4. **`tests/` exists but has no `regression`-marked tests yet** (unit +
   integration only). The `test` matrix leg for `regression` reports
   "no tests collected" as a warning; it becomes a hard gate as soon as such
   tests land.
5. **`pyproject.toml` declares `requires-python = ">=3.11"`** and ruff/mypy
   target `py311`; the image and CI deliberately pin **3.13**, which is inside
   the supported range.
6. **Docker could not be executed in the authoring environment** (no daemon, no
   WSL distro), so the image build and container boot were validated statically
   and by executing the CI shell logic, not by a real `docker build`. The
   `HEALTHCHECK` semantics were verified against a local server: 200 and 503
   exit 0, while 500/404/connection-refused exit 1.
