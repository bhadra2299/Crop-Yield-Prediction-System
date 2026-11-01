# syntax=docker/dockerfile:1.7
#
# =============================================================================
#  agri - production container image
#  Smart Crop AI platform: FastAPI serving + tabular/vision ML modules
# =============================================================================
#
#  BUILD (context MUST be the repository root, Dockerfile path is explicit):
#
#      docker build --file docker/Dockerfile --target production -t agri-platform:latest .
#
#  Targets / stages
#  ----------------
#    builder        compilers + venv at /opt/venv, installs the core package
#    runtime        slim, non-root (agri, uid/gid 10001), venv copied in, no compilers
#    vision         runtime + the ".[vision]" extra (torch/torchvision) for the
#                   image-classification modules (soil_image, disease)
#    production     DEFAULT target. An alias of "runtime", declared last so a bare
#                   "docker build" can never silently pull torch into the default image.
#
#  !! .DOCKERIGNORE CAVEAT !!
#  -----------------------------------------------------------------------------
#  docker/.dockerignore is NOT read by the daemon: the ignore file is always taken
#  from the *build context root*. With --file docker/Dockerfile and context ".", the
#  daemon looks for <repo-root>/.dockerignore, which this change-set is not allowed
#  to create (write scope is docker/ + .github/). Consequences today:
#    * the full working tree is sent to the daemon (~ GBs of data/raw images and
#      .venv), making builds slow;
#    * nothing unsafe is *copied* into the image (every COPY below is explicit),
#      and data/ + models/ are never baked in regardless.
#  Fix: copy docker/.dockerignore to <repo-root>/.dockerignore (ready-to-paste
#  content is kept in .github/DOCKERIGNORE.md and summarized here).
#  This Dockerfile is written so that the build stays CORRECT without that file,
#  only slower. Keep COPY statements explicit and narrow.
#
#  Repo-root .dockerignore contents (also in docker/.dockerignore / .github/DOCKERIGNORE.md):
#      .git/
#      .github/
#      .venv/
#      venv/
#      **/__pycache__/
#      **/*.py[cod]
#      *.egg-info/
#      build/
#      dist/
#      .pytest_cache/
#      .ruff_cache/
#      .mypy_cache/
#      .coverage
#      .coverage.*
#      coverage*.xml
#      htmlcov/
#      data/
#      models/
#      reports/
#      docs/
#      notebooks/
#      scripts/
#      tests/
#      node_modules/
#      .env
#      .env.*
#      *.pem
#      *.key
#      .DS_Store
#      Thumbs.db
#
# =============================================================================

ARG PYTHON_VERSION=3.13
ARG DEBIAN_SUITE=bookworm

# -----------------------------------------------------------------------------
# Stage 1 - builder: compilers live here and are left behind.
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_SUITE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    PYTHONDONTWRITEBYTECODE=1

# Toolchain for any dependency that has no manylinux wheel for CPython 3.13
# (scipy/statsmodels/xgboost usually ship wheels, but a source fallback must not
# break the build). These packages never reach the runtime stage.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# pyproject.toml declares readme = "README.md", so README.md is required here.
COPY pyproject.toml README.md ./
COPY src ./src

# Self-contained venv. The runtime stage copies it to the same absolute path
# (/opt/venv) from the same base image, so shebangs and pyvenv.cfg stay valid.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Core dependencies only - never the "vision" extra (torch is ~1 GB and belongs
# to the dedicated stage below).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

# -----------------------------------------------------------------------------
# Stage 2 - runtime: slim, non-root, no compilers, no pip cache, no secrets.
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_SUITE} AS runtime

# OCI labels (informational, no secrets).
LABEL org.opencontainers.image.title="agri-platform" \
      org.opencontainers.image.description="Smart Crop AI - agricultural decision-support platform (FastAPI + ML)" \
      org.opencontainers.image.source="https://github.com/bhadra2299/Crop-Yield-Prediction-System" \
      org.opencontainers.image.licenses="Proprietary"

# The runtime account is fixed at uid/gid 10001 so volume ownership, Kubernetes
# runAsUser and bind-mount permissions are predictable across environments.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    AGRI_ENV=production \
    AGRI_LOG_JSON=true \
    AGRI_PROJECT_ROOT=/app \
    AGRI_CONFIG_DIR=/app/configs \
    AGRI_MODEL_REGISTRY=/app/models/registry \
    AGRI_API_HOST=0.0.0.0 \
    AGRI_API_PORT=8000

# tini  -> PID 1 that reaps zombies and forwards SIGTERM to uvicorn (graceful shutdown)
# libgomp1 -> OpenMP runtime used by scikit-learn / xgboost wheels
# ca-certificates -> outbound HTTPS to OpenWeather / NASA POWER
# No curl/wget: the healthcheck below is pure Python because slim has neither.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tini \
        libgomp1 \
        ca-certificates \
 && groupadd --gid 10001 agri \
 && useradd --uid 10001 --gid 10001 \
            --create-home --home-dir /app --shell /usr/sbin/nologin agri \
 && rm -rf /var/lib/apt/lists/*

# Application venv from the builder. The runtime PATH already points at it.
COPY --from=builder --chown=root:root /opt/venv /opt/venv

WORKDIR /app

# Config files are the single source of truth for training AND serving
# (agri.common.paths resolves CONFIG_DIR from AGRI_CONFIG_DIR). Data and models
# are NOT copied: data/raw comes from a read-only mount, models from a volume.
COPY --chown=agri:agri configs ./configs
COPY --chown=agri:agri pyproject.toml ./

# Writable scaffolding owned by the runtime user. The registry is deliberately
# EMPTY: the service must boot and report degraded health with no trained model.
RUN install -d -o agri -g agri -m 0755 \
        /app/models/registry \
        /app/data/raw \
        /app/data/interim \
        /app/reports/metrics \
        /app/reports/figures

USER agri

EXPOSE 8000

# Health = "the server answers on /v1/health".
#   200 -> healthy
#   503 -> DEGRADED (e.g. empty model registry / strict_models=false). Still alive.
#   anything else / connection refused -> unhealthy.
# Pure stdlib: python is guaranteed present, curl is not.
# http.client does not raise on 503, so the status is inspected directly.
HEALTHCHECK --interval=30s --timeout=6s --start-period=25s --retries=3 \
    CMD ["python", "-c", "import sys, http.client; c = http.client.HTTPConnection('127.0.0.1', 8000, timeout=4); c.request('GET', '/v1/health'); s = c.getresponse().status; print('health', s); sys.exit(0 if s in (200, 503) else 1)"]

ENTRYPOINT ["/usr/bin/tini", "--"]

# Serving entry point. AGRI_API_HOST/PORT in the environment are the settings
# layer's business; uvicorn is bound explicitly so the container is self-describing.
CMD ["uvicorn", "agri.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

# -----------------------------------------------------------------------------
# Stage 3 - vision: runtime + the optional torch/torchvision extra.
#   Only for the image-classification modules (soil_image, disease).
#   Build: docker build --file docker/Dockerfile --target vision -t agri-platform:vision .
#   The default target does NOT install this extra.
# -----------------------------------------------------------------------------
FROM builder AS vision-builder

# CPU-only wheels keep the image ~2 GB instead of ~7 GB with the CUDA runtime.
# pip compares versions across both indexes; the "+cpu" local version sorts above
# the plain release, so the CPU build wins. Override the ARG to change that.
ARG TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url "${TORCH_WHEEL_INDEX}" ".[vision]"

FROM runtime AS vision

COPY --from=vision-builder --chown=root:root /opt/venv /opt/venv

# -----------------------------------------------------------------------------
# Stage 4 - production: DEFAULT target (must stay last).
#   A bare "docker build --file docker/Dockerfile ." builds this alias of the
#   slim runtime, so the vision extra can never leak into the default image.
# -----------------------------------------------------------------------------
FROM runtime AS production