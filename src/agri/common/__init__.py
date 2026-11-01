"""Shared contract and utilities. Every module imports from here, never sideways."""

from agri.common.base import BasePredictor, Module, PredictorState
from agri.common.config import ModuleConfig, Settings, get_settings, load_module_config
from agri.common.errors import (
    AgriError,
    ConfigError,
    DataValidationError,
    ExternalServiceError,
    ModelNotTrainedError,
    ModuleExecutionError,
    RegistryError,
)
from agri.common.logging import get_logger, setup_logging
from agri.common.paths import PROJECT_ROOT, ensure_dir, read_json, write_json
from agri.common.registry import ModelRegistry, RegistryEntry
from agri.common.schemas import (
    Confidence,
    HealthStatus,
    Interval,
    ModuleHealth,
    Prediction,
    TaskType,
)
from agri.common.seed import set_global_seed

__all__ = [
    "PROJECT_ROOT",
    "AgriError",
    "BasePredictor",
    "Confidence",
    "ConfigError",
    "DataValidationError",
    "ExternalServiceError",
    "HealthStatus",
    "Interval",
    "ModelNotTrainedError",
    "ModelRegistry",
    "Module",
    "ModuleConfig",
    "ModuleExecutionError",
    "ModuleHealth",
    "Prediction",
    "PredictorState",
    "RegistryEntry",
    "RegistryError",
    "Settings",
    "TaskType",
    "ensure_dir",
    "get_logger",
    "get_settings",
    "load_module_config",
    "read_json",
    "set_global_seed",
    "setup_logging",
    "write_json",
]
