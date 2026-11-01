"""Exception hierarchy. Every failure a caller can see has a stable code."""

from __future__ import annotations

from typing import Any


class AgriError(Exception):
    """Base class for every error raised by this package.

    The 'code' attribute is a stable machine-readable string that the API
    surfaces in the error envelope, so clients can branch on it without
    parsing prose.
    """

    code: str = "agri_error"
    http_status: int = 500

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


class ConfigError(AgriError):
    """Malformed or missing configuration."""

    code = "config_error"
    # 503, not 500. A configuration failure means this deployment cannot serve the
    # request as configured - typically a missing API key - which is an availability
    # problem, not a bug. A 500 sends an operator looking for a crash that is not there
    # and makes a missing key indistinguishable from a broken endpoint.
    http_status = 503


class DataValidationError(AgriError):
    """A dataset or request payload violated its declared schema."""

    code = "data_validation_error"
    http_status = 422


class ModelNotTrainedError(AgriError):
    """No artefact is registered for the requested module or version."""

    code = "model_not_trained"
    http_status = 503


class ModuleExecutionError(AgriError):
    """A module loaded but failed while producing a prediction."""

    code = "module_execution_error"
    http_status = 500


class ExternalServiceError(AgriError):
    """An upstream API failed or returned something unusable."""

    code = "external_service_error"
    http_status = 502


class RegistryError(AgriError):
    """The model registry is inconsistent or a write failed."""

    code = "registry_error"
    http_status = 500
