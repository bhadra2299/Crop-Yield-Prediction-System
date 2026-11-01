"""The FastAPI application.

Construction is explicit and side-effect free: importing this module does not open
a registry, load a model or touch the network. That matters because tests import
it, the OpenAPI schema generator imports it, and a crash during import would take
all of that down.

The lifespan hook performs the two things that must happen once per worker:
configure logging, and warm the predictors whose config asks for it. A warm-up
failure is logged and tolerated - a module that cannot load must not prevent the
process from starting, because /v1/health exists precisely to report that state.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from agri import __version__
from agri.api.errors import install_error_handlers
from agri.api.registry_cache import get_cache
from agri.api.routers import ingest, predict, system
from agri.common.config import available_module_configs, get_settings
from agri.common.logging import get_logger, new_request_id, setup_logging
from agri.common.settings_helpers import describe_environment

_log = get_logger("api.app")

DESCRIPTION = """
Smart Crop AI serves ten agricultural modules behind one versioned API.

Each module answers one question - which crop suits this soil, how much water this
field needs, what a mandi price is likely to do - and every response carries the
reasoning, the model version and an honest confidence, not just a number.

Several modules are advisory rather than authoritative: soil images are trained on
synthetic data, disease weights are third-party, and the irrigation engine is a
published physics calculation with no field-truth accuracy figure. Where that is
true, the response says so in its reasons.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and warm the modules that opt into it."""
    settings = get_settings()
    setup_logging(settings.log_level, json_output=settings.log_json)

    _log.info(
        "service starting",
        extra={"version": __version__, "environment": describe_environment()},
    )
    if not settings.api_key_set:
        _log.warning(
            "authentication is DISABLED because AGRI_API_KEYS is empty",
            extra={"impact": "any caller can reach the prediction endpoints"},
        )

    cache = get_cache()
    warmed, failed = [], []
    for name in available_module_configs():
        try:
            from agri.common.config import load_module_config

            if not load_module_config(name).serving.warmup:
                continue
        except Exception:
            continue
        start = time.perf_counter()
        predictor = cache.get(name)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if predictor is not None and predictor.ready:
            warmed.append({"module": name, "ms": round(elapsed_ms, 1)})
        else:
            failed.append({"module": name, "reason": cache.load_failure(name) or "not ready"})

    _log.info(
        "warm-up complete",
        extra={"warmed": len(warmed), "failed": len(failed), "failures": failed[:5]},
    )
    yield
    _log.info("service stopping")


def create_app() -> FastAPI:
    """Build the application. Called by uvicorn and by the tests."""
    settings = get_settings()

    app = FastAPI(
        title="Smart Crop AI",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        root_path=settings.api_root_path or "",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*", "X-API-Key"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    @app.middleware("http")
    async def correlate_request(request: Request, call_next):
        """Attach a request id to every response and every log line."""
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        # Access log: method, path, status, duration. No bodies, no headers.
        _log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response

    install_error_handlers(app)

    # The frontend calls /predict/* directly, so the versioned router is mounted
    # at /v1 and the unversioned prefix is aliased for backwards compatibility.
    app.include_router(system.router, prefix="/v1")
    app.include_router(predict.router, prefix="/v1")
    app.include_router(ingest.router, prefix="/v1")
    app.include_router(predict.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": "Smart Crop AI",
            "version": __version__,
            "docs": "/docs",
            "health": "/v1/health",
            "modules": available_module_configs(),
        }

    return app


app = create_app()
