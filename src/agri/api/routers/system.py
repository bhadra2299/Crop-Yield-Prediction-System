"""Health, readiness, metrics and model-inventory endpoints.

Health is the endpoint a load balancer polls, so its contract is explicit:

  200  every required module is ready to serve
  503  at least one required module is NOT ready

A 503 from /v1/health is a legitimate, informative answer - it means the process
is alive and is telling you it cannot serve usefully yet. The container
HEALTHCHECK in docker/Dockerfile therefore accepts both, because a container that
reports degraded is not the same as a container that has crashed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status

from agri.api.registry_cache import get_cache
from agri.api.security import enforce_rate_limit, require_api_key
from agri.common.config import available_module_configs, get_settings
from agri.common.logging import get_logger
from agri.common.paths import METRICS_DIR, read_json
from agri.common.registry import ModelRegistry
from agri.common.schemas import HealthStatus, ModuleHealth
from agri.common.settings_helpers import describe_environment

_log = get_logger("api.system")

router = APIRouter(tags=["system"])

#: Modules whose absence makes the service unable to answer its core question
#: (what should I do with this field). Weather and the image modules are
#: genuinely optional: their absence degrades an answer, it does not prevent one.
REQUIRED_MODULES = ("crop_recommendation", "irrigation")


@router.get("/health", summary="Liveness and readiness for every module")
async def health(response: Response) -> dict:
    """Report per-module readiness.

    200 means every REQUIRED module can serve. 503 means at least one cannot, which
    is an informative answer rather than a crash: the process is alive and is telling
    you what it is missing.

    Optional modules (weather, the image models) never degrade the status. Their
    absence makes an advisory less complete, not wrong.
    """
    registry = ModelRegistry()
    modules: list[dict] = []
    degraded: list[str] = []

    for name in available_module_configs():
        predictor = get_cache().get(name)

        if predictor is None:
            # No predictor object at all: report it, do not pretend it is fine.
            modules.append(
                ModuleHealth(
                    name=name,
                    status=HealthStatus.DOWN,
                    ready=False,
                    version=None,
                    artifact_present=registry.latest_version(name) is not None,
                    detail=get_cache().load_failure(name) or "predictor unavailable",
                ).model_dump(mode="json")
            )
            if name in REQUIRED_MODULES:
                degraded.append(name)
            continue

        report = predictor.health()
        modules.append(report.model_dump(mode="json"))
        if not report.ready and name in REQUIRED_MODULES:
            degraded.append(name)

    ready = not degraded
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    _log.info(
        "health checked",
        extra={"ready": ready, "degraded": degraded, "modules": len(modules)},
    )
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "degraded_modules": degraded,
        "required_modules": list(REQUIRED_MODULES),
        "checked_at": datetime.now(UTC).isoformat(),
        "environment": describe_environment(),
        "modules": modules,
        "auth_enabled": bool(get_settings().api_key_set),
        "note": (
            "A degraded status means at least one required module has no usable artefact. "
            "Train the listed modules, or check /v1/models for what is registered."
        )
        if not ready
        else "All required modules are ready.",
    }


@router.get("/ready/{module}", summary="Readiness for one module")
async def ready(module: str, response: Response) -> dict:
    if module not in available_module_configs():
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            "module": module,
            "status": HealthStatus.DOWN.value,
            "ready": False,
            "detail": "No such module.",
        }

    predictor = get_cache().get(module)
    if predictor is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "module": module,
            "status": HealthStatus.DOWN.value,
            "ready": False,
            "detail": get_cache().load_failure(module) or "predictor unavailable",
        }

    report = predictor.health()
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"module": module, **report.model_dump(mode="json")}


@router.get(
    "/models",
    summary="Registered artefact inventory",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def models() -> dict:
    """What is registered, with the metrics and provenance recorded at train time."""
    registry = ModelRegistry()
    entries = []
    for name in available_module_configs():
        registered = registry.list_entries(name)
        latest = registry.latest_version(name)
        detail = None
        if latest:
            try:
                manifest = registry.get(name, latest).manifest
                detail = {
                    "trained_at": manifest.get("trained_at"),
                    "dataset": manifest.get("dataset"),
                    "data_digest": manifest.get("data_digest"),
                    "git_sha": manifest.get("git_sha"),
                    "metrics": manifest.get("metrics"),
                    "artifact_bytes": manifest.get("artifact_bytes"),
                }
            except Exception as exc:
                detail = {"error": f"manifest unreadable: {exc}"}
        entries.append(
            {
                "module": name,
                "registered_versions": registered,
                "latest": latest,
                "manifest": detail,
            }
        )
    return {"count": len(entries), "models": entries}


@router.get(
    "/metrics/{module}",
    summary="The evaluation report for a module",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def metrics(module: str, response: Response) -> dict:
    """Serve the metrics report written at train or evaluate time, plus the gate verdict."""
    if module not in available_module_configs():
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"module": module, "detail": "No such module."}

    candidates = sorted(METRICS_DIR.glob(f"{module}_*.json")) if METRICS_DIR.is_dir() else []
    if not candidates:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            "module": module,
            "detail": "No metrics report has been written yet. Train or evaluate the module first.",
        }

    # The evaluated report is preferred when present: it reflects the current
    # artefact rather than whatever was recorded when the model was fitted.
    preferred = [path for path in candidates if "evaluated" in path.name] or candidates
    latest = max(preferred, key=lambda path: path.stat().st_mtime)
    return {"module": module, "report_file": latest.name, "report": read_json(latest)}


@router.get("/version", summary="Build and library versions")
async def version() -> dict:
    import platform
    import sys

    from agri import __version__

    libraries: dict[str, str] = {}
    for name in ("fastapi", "pydantic", "numpy", "pandas", "sklearn", "xgboost"):
        try:
            module = __import__(name)
            libraries[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue

    return {
        "name": "smart-crop-ai",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": libraries,
    }
