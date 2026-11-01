"""The module contract.

Every one of the ten modules exposes exactly one class deriving from
BasePredictor, plus train/evaluate entry points. The base class owns the parts
that must not vary: config loading, artefact resolution, readiness state,
timing, error translation and logging. A module only implements _load and
_predict.

Why a template method rather than free functions:
  * the API can treat all modules identically (same health, same error envelope)
  * a module that forgets to load its artefact fails loudly at first predict
  * latency and version stamping happen in exactly one place
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from agri.common.config import ModuleConfig, load_module_config
from agri.common.errors import (
    AgriError,
    DataValidationError,
    ModelNotTrainedError,
    ModuleExecutionError,
)
from agri.common.logging import get_logger
from agri.common.registry import ModelRegistry, RegistryEntry
from agri.common.schemas import Confidence, HealthStatus, ModuleHealth, Prediction

ReqT = TypeVar("ReqT", bound=BaseModel)
ResT = TypeVar("ResT", bound=Prediction)


class PredictorState(StrEnum):
    """Lifecycle of a predictor instance. Only READY may serve traffic."""

    NOT_LOADED = "not_loaded"
    READY = "ready"
    FAILED = "failed"


class BasePredictor(ABC, Generic[ReqT, ResT]):
    """Base class for all module predictors.

    Subclasses set module_name and implement _load / _predict. They must return
    a Prediction subclass so the API can serialise every module uniformly.
    """

    #: Must match the configs/<module_name>.yaml filename and the registry folder.
    module_name: ClassVar[str] = ""
    #: False for physics/inference modules that carry their own logic, not a joblib artefact.
    #: Pydantic model for the request body. Declared here so tooling can find it
    #: on every module, tabular or not, without importing module internals.
    request_model: ClassVar[Any] = None
    requires_artifact: ClassVar[bool] = True
    artifact_filename: ClassVar[str] = "model.joblib"

    def __init__(
        self,
        config: ModuleConfig | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        if not self.module_name:
            raise ValueError(f"{type(self).__name__} must define module_name")
        self.config: ModuleConfig = config or load_module_config(self.module_name)
        self.registry = registry or ModelRegistry()
        self.entry: RegistryEntry | None = None
        self.artifact: Any = None
        self._state: PredictorState = PredictorState.NOT_LOADED
        self._failure: str | None = None
        self.log = get_logger(f"modules.{self.module_name}")

    # ------------------------------------------------------------- lifecycle
    @property
    def state(self) -> PredictorState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state is PredictorState.READY

    @property
    def version(self) -> str:
        return self.entry.version if self.entry else self.config.version

    def load(self) -> BasePredictor[ReqT, ResT]:
        """Load the artefact and any heavy resources. Idempotent.

        A failure is recorded rather than raised, so /ready reports the specific
        module that is down while every other module keeps serving.
        """
        if self._state is PredictorState.READY:
            return self
        try:
            if self.requires_artifact:
                self.entry = self.registry.get(self.module_name, self.config.version)
                self.artifact = self.entry.load()
            self._load()
            self._state = PredictorState.READY
            self._failure = None
            self.log.info(
                "module ready",
                extra={"module": self.module_name, "version": self.version, "artifact": bool(self.entry)},
            )
        except AgriError as exc:
            self._state = PredictorState.FAILED
            self._failure = exc.message
            self.log.warning(
                "module failed to load", extra={"module": self.module_name, "error": exc.message}
            )
        except Exception as exc:
            self._state = PredictorState.FAILED
            self._failure = f"{type(exc).__name__}: {exc}"
            self.log.exception("unexpected error loading module", extra={"module": self.module_name})
        return self

    def _load(self) -> None:
        """Hook for extra resources (tokenizers, physics tables, calibration curves)."""
        return None

    def _fallback(self, request: ReqT) -> ResT:
        """Called instead of raising when serving.allow_fallback is set.

        The default refuses to invent an answer. A module may override this to
        return a documented degraded response - it must set Confidence.LOW or
        UNKNOWN and explain itself in 'reasons'.
        """
        raise ModelNotTrainedError(
            f"Module '{self.module_name}' is not ready and defines no fallback",
            context={"failure": self._failure, "version": self.config.version},
        )

    # ------------------------------------------------------------- prediction
    def predict(self, request: ReqT | dict[str, Any]) -> ResT:
        """Validate readiness, run _predict, stamp metadata and translate errors.

        Accepts either the module's request model or a plain dict. Dicts are coerced
        through request_model so that a caller cannot bypass validation - every module
        ships a SAMPLE dict, the CLI passes dicts, and _predict is entitled to assume a
        validated model. Without this, predict(SAMPLE) failed with AttributeError
        because the raw dict reached _predict.
        """
        if self._state is PredictorState.NOT_LOADED:
            self.load()

        request = self._coerce_request(request)

        if self._state is not PredictorState.READY:
            if self.config.serving.allow_fallback:
                result = self._fallback(request)
                return self._finalize(result, latency_ms=None)
            raise ModelNotTrainedError(
                f"Module '{self.module_name}' is not ready",
                context={"failure": self._failure, "version": self.config.version},
            )

        started = time.perf_counter()
        try:
            result = self._predict(request)
        except AgriError:
            raise
        except Exception as exc:
            self.log.exception("prediction failed", extra={"module": self.module_name})
            raise ModuleExecutionError(
                f"Module '{self.module_name}' failed to produce a prediction",
                context={"error": f"{type(exc).__name__}: {exc}"},
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        if latency_ms > self.config.serving.latency_budget_ms:
            self.log.warning(
                "latency budget exceeded",
                extra={
                    "module": self.module_name,
                    "latency_ms": round(latency_ms, 1),
                    "budget_ms": self.config.serving.latency_budget_ms,
                },
            )
        return self._finalize(result, latency_ms=latency_ms)

    def _coerce_request(self, request: Any) -> Any:
        """Turn a dict into the declared request model, translating validation errors."""
        model = self.request_model
        if model is None or isinstance(request, model):
            return request
        if isinstance(request, dict):
            try:
                return model.model_validate(request)
            except ValidationError as exc:
                raise DataValidationError(
                    f"Request does not match the {self.module_name} contract",
                    context={"errors": exc.errors(include_url=False)},
                ) from exc
        raise DataValidationError(
            f"Expected {model.__name__} or a dict, got {type(request).__name__}",
            context={"module": self.module_name},
        )

    @abstractmethod
    def _predict(self, request: ReqT) -> ResT:
        """Produce a response. Runs only when the module is READY."""
        raise NotImplementedError

    def _finalize(self, result: ResT, *, latency_ms: float | None) -> ResT:
        """Stamp version and latency. Subclasses should not override this."""
        if not isinstance(result, Prediction):
            raise ModuleExecutionError(
                f"{type(self).__name__}._predict returned {type(result).__name__}, "
                f"which is not a Prediction envelope"
            )
        result.model_name = self.module_name
        result.model_version = self.version
        result.latency_ms = round(latency_ms, 2) if latency_ms is not None else None
        if result.confidence is Confidence.UNKNOWN and result.confidence_score is not None:
            result.confidence = Confidence.from_score(
                result.confidence_score,
                high=self.config.serving.confidence_high,
                medium=self.config.serving.confidence_medium,
            )
        return result

    # ----------------------------------------------------------------- health
    def health(self) -> ModuleHealth:
        """Readiness snapshot for /v1/health. Never raises."""
        return ModuleHealth(
            name=self.module_name,
            status=HealthStatus.OK if self.ready else HealthStatus.DOWN,
            ready=self.ready,
            version=self.version,
            artifact_present=bool(self.entry and self.entry.exists()),
            detail=self._failure,
            latency_budget_ms=self.config.serving.latency_budget_ms,
        )

    # ---------------------------------------------------------------- helpers
    def classify_confidence(self, score: float | None) -> Confidence:
        return Confidence.from_score(
            score,
            high=self.config.serving.confidence_high,
            medium=self.config.serving.confidence_medium,
        )


# Backwards-friendly alias: the contract reads better as 'Module' when listed.
Module = BasePredictor
