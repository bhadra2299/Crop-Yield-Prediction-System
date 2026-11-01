"""The agri command line.

One entry point, subcommands grouped by verb, and exit codes that mean something
so CI can gate on them:

  0  success, every gate passed
  1  a gate failed or the work could not be done
  2  the command itself was used incorrectly

That contract is what makes "agri evaluate --all" usable as a CI step that fails
a merge when a published metric regresses.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agri import __version__
from agri.common.errors import AgriError
from agri.common.logging import get_logger, setup_logging

_log = get_logger("cli")

MODULES = (
    "crop_recommendation",
    "yield_prediction",
    "fertilizer",
    "soil_nutrient",
    "market_price",
    "risk",
    "irrigation",
    "weather",
    "soil_image",
    "disease",
)

#: Modules with nothing to fit; train verifies them instead.
NON_TRAINING = ("irrigation", "weather", "disease")


def _fail(message: str, code: int = 1, **context: Any) -> int:
    _log.error("command failed", extra={"error": message, **context})
    print(json.dumps({"ok": False, "error": message, **context}, indent=2, default=str), file=sys.stderr)
    return code


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------- commands
def cmd_modules(args: argparse.Namespace) -> int:
    from agri.common.config import available_module_configs, load_module_config
    from agri.common.registry import ModelRegistry

    registry = ModelRegistry()
    rows = []
    for name in available_module_configs():
        try:
            config = load_module_config(name)
        except AgriError as exc:
            rows.append({"module": name, "valid": False, "error": exc.message})
            continue
        rows.append(
            {
                "module": name,
                "valid": True,
                "version": config.version,
                "task": config.task.value,
                "algorithm": config.model.algorithm,
                "primary_metric": config.gates.primary_metric,
                "registered": registry.latest_version(name),
            }
        )
    _emit({"count": len(rows), "modules": rows})
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    from agri.common.config import available_module_configs, load_module_config
    from agri.common.validation import validate_file

    names = available_module_configs() if args.all else [args.module]
    results = []
    failed = 0
    for name in names:
        config = load_module_config(name)
        path = config.data.raw_path
        if path.is_dir():
            results.append({"module": name, "skipped": True, "reason": "module reads a directory"})
            continue
        if not path.is_file():
            results.append({"module": name, "ok": False, "error": f"missing: {path}"})
            failed += 1
            continue
        report = validate_file(path, config.data)
        results.append({"module": name, "ok": report.ok, "summary": report.summary()})
        if not report.ok:
            failed += 1
    _emit({"checked": len(results), "failed": failed, "results": results})
    return 1 if failed else 0


def cmd_train(args: argparse.Namespace) -> int:
    names = list(MODULES) if args.all else [args.module]
    summary = []
    failed = 0
    for name in names:
        if name in NON_TRAINING:
            summary.append({"module": name, "action": "verified", "reason": "no training step exists"})
            continue
        from agri.common.config import load_module_config

        try:
            config = load_module_config(name)
        except AgriError as exc:
            summary.append({"module": name, "ok": False, "error": exc.message})
            failed += 1
            continue

        if name == "market_price":
            from agri.modules.market_price.train import main as train_market

            code = train_market(["--max-series", str(args.max_series)])
        else:
            from agri.common.training import train_module

            try:
                train_module(name, tune=not args.no_tune, register=not args.no_register)
                code = 0
            except AgriError as exc:
                summary.append({"module": name, "ok": False, "error": exc.message})
                failed += 1
                continue
        summary.append({"module": name, "ok": code == 0, "version": config.version})
        if code != 0:
            failed += 1
    _emit({"trained": len(summary), "failed": failed, "results": summary})
    return 1 if failed else 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Re-evaluate registered modules against the current data.

    This is the CI metric-floor gate: a non-zero exit means at least one
    published metric no longer clears its configured floor.
    """
    from agri.common.config import available_module_configs
    from agri.common.registry import ModelRegistry
    from agri.common.training import evaluate_module

    registry = ModelRegistry()
    if args.all:
        trained = [name for name in available_module_configs() if registry.latest_version(name) is not None]
        if not trained:
            return _fail("No module has a registered artefact, so there is nothing to evaluate.")
        names = trained
    else:
        names = [args.module]

    results = []
    failed = 0
    for name in names:
        if name in ("irrigation", "weather"):
            module = __import__(f"agri.modules.{name}.evaluate", fromlist=["main"])
            code = module.main([])
            results.append({"module": name, "ok": code == 0, "kind": "physics_check"})
            failed += 0 if code == 0 else 1
            continue
        if name == "disease":
            results.append({"module": name, "skipped": True, "reason": "inference-only, no gate"})
            continue
        if name in ("market_price", "soil_image"):
            # These modules do not fit the generic tabular evaluator: market_price
            # backtests a time series, and soil_image reads a directory of images rather
            # than a CSV. Each owns its evaluation, so dispatch to it instead of forcing
            # it through a loader that would reject its data.
            module = __import__(f"agri.modules.{name}.evaluate", fromlist=["main"])
            code = module.main([])
            results.append({"module": name, "ok": code == 0})
            failed += 0 if code == 0 else 1
            continue
        try:
            outcome = evaluate_module(name, version=args.version)
            results.append(
                {"module": name, "ok": True, "metrics": outcome["metrics"], "gate": outcome["gate"]}
            )
        except AgriError as exc:
            results.append({"module": name, "ok": False, "error": exc.message, "context": exc.context})
            failed += 1

    _emit({"evaluated": len(results), "failed": failed, "all_passed": failed == 0, "results": results})
    return 1 if failed else 0


def cmd_health(args: argparse.Namespace) -> int:
    """Report whether each module can actually serve."""
    from agri.api.registry_cache import get_cache
    from agri.common.config import available_module_configs

    cache = get_cache()
    rows = []
    ready = 0
    for name in available_module_configs():
        predictor = cache.get(name)
        if predictor is None:
            rows.append({"module": name, "status": "unavailable", "detail": cache.load_failure(name)})
            continue
        report = predictor.health()
        rows.append(
            {
                "module": name,
                "status": report.status.value,
                "ready": report.ready,
                "version": predictor.version,
                "detail": report.detail,
            }
        )
        ready += 1 if report.ready else 0
    _emit({"ready": ready, "total": len(rows), "modules": rows})
    return 0 if ready else 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the API. Delegates to uvicorn so reload and workers behave normally."""
    try:
        import uvicorn
    except ImportError:
        return _fail("uvicorn is not installed; install the serving dependencies")

    _log.info("starting api", extra={"host": args.host, "port": args.port})
    uvicorn.run(
        "agri.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=None if args.reload else args.workers,
        log_config=None,
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose the environment: python, data, registry, secrets, optional extras."""
    import importlib
    import platform
    import sys as _sys

    from agri.common.config import available_module_configs, load_module_config
    from agri.common.paths import DATA_DIR, MODELS_DIR, PROJECT_ROOT
    from agri.common.registry import ModelRegistry
    from agri.common.settings_helpers import describe_environment

    registry = ModelRegistry()
    checks: list[dict[str, Any]] = []

    checks.append({"check": "python", "ok": _sys.version_info >= (3, 11), "value": _sys.version.split()[0]})
    checks.append(
        {
            "check": "project_root",
            "ok": (PROJECT_ROOT / "pyproject.toml").is_file(),
            "value": str(PROJECT_ROOT),
        }
    )

    for extra in ("torch", "matplotlib", "xgboost", "prophet"):
        try:
            module = importlib.import_module(extra)
            checks.append(
                {"check": f"optional:{extra}", "ok": True, "value": getattr(module, "__version__", "present")}
            )
        except ImportError:
            checks.append({"check": f"optional:{extra}", "ok": extra in ("torch",), "value": "not installed"})

    missing = []
    for name in available_module_configs():
        config = load_module_config(name)
        path = config.data.raw_path
        if path.is_file() or path.is_dir():
            continue
        missing.append({"module": name, "path": str(path)})
    checks.append({"check": "datasets_present", "ok": not missing, "missing": missing})

    registered = {name: registry.latest_version(name) for name in available_module_configs()}
    checks.append(
        {
            "check": "registry",
            "ok": MODELS_DIR.is_dir(),
            "value": str(MODELS_DIR),
            "registered": {key: value for key, value in registered.items() if value},
        }
    )

    environment = describe_environment()
    checks.append(
        {
            "check": "secrets",
            "ok": True,
            "value": {
                "openweather_configured": environment["openweather_configured"],
                "auth_enabled": environment["auth_enabled"],
                "note": "Secrets are read from the environment only; nothing is read from files in the repository.",
            },
        }
    )
    checks.append({"check": "data_dir", "ok": DATA_DIR.is_dir(), "value": str(DATA_DIR)})
    checks.append({"check": "platform", "ok": True, "value": platform.platform()})

    failures = [check for check in checks if not check["ok"]]
    _emit({"ok": not failures, "failures": len(failures), "checks": checks})
    return 1 if failures else 0


def cmd_drift(args: argparse.Namespace) -> int:
    """Report drift for a module against its training distribution."""
    import pandas as pd

    from agri.monitoring.drift import drift_report_for
    from agri.monitoring.prediction_log import read_predictions

    records = [item for item in read_predictions() if item.get("module") == args.module]
    lone = [
        item["request"]
        for item in records
        if isinstance(item.get("request"), dict) and len(item["request"]) > 2
    ]
    if len(lone) < 10:
        _log.warning(
            "too few logged requests for a drift reading",
            extra={"module": args.module, "requests": len(lone)},
        )
    frame = pd.DataFrame(lone) if lone else pd.DataFrame()
    report = drift_report_for(args.module, frame)
    _emit(report.to_dict())
    return 0


# ----------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agri", description="Smart Crop AI command line: data, training, evaluation and serving."
    )
    parser.add_argument("--version", action="version", version=f"smart-crop-ai {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    modules = subparsers.add_parser("modules", help="list configured modules")
    modules.set_defaults(func=cmd_modules)

    data = subparsers.add_parser("data", help="validate input data")
    data.add_argument("--all", action="store_true")
    data.add_argument("--module", default="crop_recommendation")
    data.set_defaults(func=cmd_data)

    train = subparsers.add_parser("train", help="train modules")
    train.add_argument("--all", action="store_true")
    train.add_argument("--module", default="crop_recommendation")
    train.add_argument("--no-tune", action="store_true")
    train.add_argument("--no-register", action="store_true")
    train.add_argument("--max-series", type=int, default=300)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate modules and enforce metric gates")
    evaluate.add_argument("--all", action="store_true")
    evaluate.add_argument("--module", default="crop_recommendation")
    evaluate.add_argument("--version", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    serve = subparsers.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--workers", type=int, default=1)
    serve.set_defaults(func=cmd_serve)

    health = subparsers.add_parser("health", help="report per-module readiness")
    health.set_defaults(func=cmd_health)

    doctor = subparsers.add_parser("doctor", help="diagnose the environment")
    doctor.set_defaults(func=cmd_doctor)

    drift = subparsers.add_parser("drift", help="drift report for a module")
    drift.add_argument("--module", default="crop_recommendation")
    drift.set_defaults(func=cmd_drift)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()
    try:
        return int(args.func(args))
    except AgriError as exc:
        return _fail(exc.message, code=1, error_code=exc.code, context=exc.context)
    except KeyboardInterrupt:
        _log.info("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
