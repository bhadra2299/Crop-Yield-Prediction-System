"""Runnable entry points shared by every tabular module.

Each module's train.py / evaluate.py / predict.py is a two-line wrapper around
these, so the CLI surface is identical across modules and cannot drift.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from agri.common.errors import AgriError
from agri.common.logging import get_logger, setup_logging
from agri.common.registry import ModelRegistry
from agri.common.training import add_training_arguments, evaluate_module, train_module

_log = get_logger("modules.cli")


def run_train(module_name: str, argv: list[str] | None = None) -> int:
    """Train a module. Exit code 1 when the metric gate fails, which CI relies on."""
    parser = add_training_arguments(argparse.ArgumentParser(description=f"Train {module_name}"))
    args = parser.parse_args(argv)
    setup_logging()
    try:
        result = train_module(
            module_name, tune=not args.no_tune, n_iter=args.n_iter, register=not args.no_register
        )
    except AgriError as exc:
        _log.error("training failed", extra={"module": module_name, "code": exc.code, "error": exc.message})
        print(json.dumps({"ok": False, "module": module_name, "error": exc.to_dict()}, indent=2))
        return 1
    print(json.dumps({"ok": True, **result}, indent=2, default=str))
    return 0


def run_evaluate(module_name: str, argv: list[str] | None = None) -> int:
    """Evaluate a registered module. Exit code 1 on any gate failure."""
    parser = argparse.ArgumentParser(description=f"Evaluate {module_name}")
    parser.add_argument("--version", default=None, help="registry version; defaults to LATEST")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        result = evaluate_module(module_name, version=args.version)
    except AgriError as exc:
        _log.error("evaluation failed", extra={"module": module_name, "code": exc.code, "error": exc.message})
        print(json.dumps({"ok": False, "module": module_name, "error": exc.to_dict()}, indent=2))
        return 1
    print(json.dumps({"ok": True, **result}, indent=2, default=str))
    return 0


def run_predict(module_name: str, predictor_class: type, sample: dict[str, Any]) -> int:
    """Load the registered artefact and print one Prediction envelope as JSON.

    Used by the API warm-up path and by integration tests; it never starts a server.
    """
    setup_logging()
    predictor = predictor_class()
    predictor.load()
    if not predictor.ready:
        print(json.dumps({"ok": False, "module": module_name, "error": predictor.health().detail}, indent=2))
        return 1
    if predictor.request_model is None:
        print(
            json.dumps(
                {"ok": False, "module": module_name, "error": "module does not declare a request_model"},
                indent=2,
            )
        )
        return 1
    request = predictor.request_model(**sample)
    prediction = predictor.predict(request)
    print(
        json.dumps(
            {"ok": True, "module": module_name, "prediction": prediction.model_dump()}, indent=2, default=str
        )
    )
    return 0


def run_health(module_names: list[str]) -> int:
    """Print a health table for the given modules. Exit 1 if any is not ready."""
    registry = ModelRegistry()
    rows = []
    for name in module_names:
        entry = registry.latest_version(name)
        rows.append({"module": name, "registered_version": entry, "trained": entry is not None})
    print(json.dumps(rows, indent=2))
    return 0 if all(row["trained"] for row in rows) else 1
