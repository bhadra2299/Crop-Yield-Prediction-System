"""A process-wide cache of loaded module predictors.

Loading an artefact is expensive (joblib deserialise, model warm-up) and must not
happen per request. This cache loads each module lazily on first use and keeps it
for the life of the worker.

It also owns the definition of "ready": a module whose artefact failed to load is
cached too, so a broken module is not retried on every single request, and its
health() reports the stored failure reason instead.
"""

from __future__ import annotations

import importlib
import threading

from agri.common.base import BasePredictor
from agri.common.logging import get_logger

_log = get_logger("api.registry")


class PredictorCache:
    """Thread-safe lazy cache of module predictors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._predictors: dict[str, BasePredictor] = {}
        self._load_failures: dict[str, str] = {}

    @staticmethod
    def module_names() -> list[str]:
        """Every module package that exposes a Predictor class."""
        from agri.common.config import available_module_configs

        names = []
        for name in available_module_configs():
            try:
                package = importlib.import_module(f"agri.modules.{name}")
            except ImportError as exc:
                _log.warning("module package failed to import", extra={"module": name, "error": str(exc)})
                continue
            if callable(getattr(package, "Predictor", None)):
                names.append(name)
        return names

    def get(self, name: str) -> BasePredictor | None:
        """Return a loaded predictor, or None when the module has no Predictor."""
        with self._lock:
            if name in self._predictors:
                return self._predictors[name]
            if name in self._load_failures:
                return None
        return self._build(name)

    def _build(self, name: str) -> BasePredictor | None:
        try:
            package = importlib.import_module(f"agri.modules.{name}")
            predictor_class = getattr(package, "Predictor", None)
            if predictor_class is None:
                with self._lock:
                    self._load_failures[name] = "module does not expose a Predictor class"
                return None
        except ImportError as exc:
            with self._lock:
                self._load_failures[name] = f"import failed: {exc}"
            _log.warning("predictor import failed", extra={"module": name, "error": str(exc)})
            return None

        try:
            predictor = predictor_class()
            predictor.load()
        except Exception as exc:
            # A module that cannot be constructed must not take down the worker.
            with self._lock:
                self._load_failures[name] = f"{type(exc).__name__}: {exc}"
            _log.warning("predictor construction failed", extra={"module": name, "error": str(exc)})
            return None

        if not predictor.ready:
            health = predictor.health()
            with self._lock:
                self._load_failures[name] = health.detail or "predictor not ready"
            _log.info("predictor not ready", extra={"module": name, "detail": health.detail})
        else:
            _log.info("predictor loaded", extra={"module": name, "version": predictor.version})

        with self._lock:
            self._predictors[name] = predictor
        return predictor

    def load_failure(self, name: str) -> str | None:
        with self._lock:
            return self._load_failures.get(name)

    def reload(self, name: str | None = None) -> None:
        """Drop cached predictors so the next request reloads them from the registry.

        Used after retraining: without this a worker would keep serving the old
        artefact until it restarted.
        """
        with self._lock:
            if name is None:
                self._predictors.clear()
                self._load_failures.clear()
            else:
                self._predictors.pop(name, None)
                self._load_failures.pop(name, None)
        _log.info("predictor cache cleared", extra={"module": name or "all"})


_cache = PredictorCache()


def get_cache() -> PredictorCache:
    return _cache
