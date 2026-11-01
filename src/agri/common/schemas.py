"""Shared response primitives. Module-specific request/response models live in the module."""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class TaskType(StrEnum):
    """What kind of model backs a module - drives metrics, gates and UI copy."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    PHYSICS = "physics"
    INFERENCE = "inference"


class Confidence(StrEnum):
    """Coarse, human-facing confidence band. Never claim more than the data supports."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

    @classmethod
    def from_score(cls, score: float | None, high: float = 0.80, medium: float = 0.60) -> Confidence:
        if score is None:
            return cls.UNKNOWN
        if score >= high:
            return cls.HIGH
        if score >= medium:
            return cls.MEDIUM
        return cls.LOW


class Interval(BaseModel):
    """Prediction interval or uncertainty range, in the target's own units."""

    model_config = ConfigDict(frozen=True)

    low: float = Field(..., description="Lower bound of the interval")
    high: float = Field(..., description="Upper bound of the interval")
    level: float = Field(0.80, ge=0.0, le=1.0, description="Nominal coverage, e.g. 0.80 for an 80% interval")

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


class Prediction(BaseModel, Generic[T]):
    """Uniform envelope every module returns.

    A module fills what it honestly knows and leaves the rest null - the API
    never fabricates a confidence or an interval it did not compute.
    """

    model_config = ConfigDict(protected_namespaces=())

    value: T = Field(..., description="The prediction itself")
    confidence_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Model probability or normalised score where available"
    )
    confidence: Confidence = Confidence.UNKNOWN
    interval: Interval | None = Field(None, description="Uncertainty range when the model provides one")
    alternatives: list[dict] | None = Field(
        None, description="Runner-up classes with scores, low confidence only"
    )
    reasons: list[str] = Field(default_factory=list, description="Plain-language drivers of this prediction")
    model_name: str = Field(..., description="Module that produced this")
    model_version: str = Field(..., description="Registered artefact version")
    latency_ms: float | None = Field(None, description="Server-side inference time")


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class ModuleHealth(BaseModel):
    """Per-module readiness, surfaced by /v1/health and /v1/ready."""

    name: str
    status: HealthStatus
    ready: bool
    version: str | None = None
    artifact_present: bool = False
    detail: str | None = None
    latency_budget_ms: int | None = None


class ErrorBody(BaseModel):
    """Machine-readable error envelope. 'code' is stable across releases."""

    code: str
    message: str
    context: dict = Field(default_factory=dict)
    request_id: str | None = None


class ProblemDetail(BaseModel):
    """RFC 9457-style problem document returned by the API."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    code: str
    instance: str | None = None
    request_id: str | None = None


class Split(StrEnum):
    """Dataset partition. 'train' and 'test' are never mixed."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class BaselineKind(StrEnum):
    """Naive comparator every model must beat for its gate to be meaningful."""

    MAJORITY_CLASS = "majority_class"
    MEAN = "mean"
    MEDIAN = "median"
    LAST_VALUE = "last_value"
    PERSISTENCE = "persistence"


Op = Literal["lte", "gte", "eq", "ne", "in", "between"]
