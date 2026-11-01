"""Configuration: environment settings plus one YAML file per module.

Design rules
------------
1. A module never reads a path, a hyperparameter, or a threshold from code.
2. Config files are the single source of truth for training AND serving, so
   offline evaluation and online inference cannot drift.
3. Values may reference environment variables as ${VAR} or ${VAR:-default};
   secrets therefore never appear in the YAML.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agri.common.errors import ConfigError
from agri.common.paths import CONFIG_DIR, PROJECT_ROOT, sha256_text
from agri.common.schemas import TaskType

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def interpolate_env(raw: str) -> str:
    """Replace ${VAR} and ${VAR:-default} with environment values.

    An unset variable with no default is an error rather than an empty string -
    silently substituting "" is how misconfigured API keys reach production.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        value = os.getenv(name)
        if value is not None:
            return value
        if default is not None:
            return default
        raise ConfigError(f"Configuration references unset environment variable '{name}'")

    return _ENV_PATTERN.sub(_replace, raw)


class Settings(BaseSettings):
    """Process-wide runtime settings, read from the environment (and .env)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    env: Literal["development", "staging", "production"] = Field("development", alias="AGRI_ENV")
    log_level: str = Field("INFO", alias="AGRI_LOG_LEVEL")
    log_json: bool = Field(False, alias="AGRI_LOG_JSON")

    api_host: str = Field("0.0.0.0", alias="AGRI_API_HOST")
    api_port: int = Field(8000, alias="AGRI_API_PORT")
    api_root_path: str = Field("", alias="AGRI_API_ROOT_PATH")
    cors_origins: str = Field("http://localhost:8080,http://localhost:5173", alias="AGRI_CORS_ORIGINS")

    api_keys: str = Field("", alias="AGRI_API_KEYS")
    rate_limit_per_minute: int = Field(60, alias="AGRI_RATE_LIMIT_PER_MINUTE")
    max_upload_bytes: int = Field(5 * 1024 * 1024, alias="AGRI_MAX_UPLOAD_BYTES")

    openweather_api_key: str = Field("", alias="OPENWEATHER_API_KEY")
    nasa_power_base_url: str = Field(
        "https://power.larc.nasa.gov/api/temporal/daily/point", alias="NASA_POWER_BASE_URL"
    )

    model_registry: str = Field("models/registry", alias="AGRI_MODEL_REGISTRY")
    config_dir: str = Field("configs", alias="AGRI_CONFIG_DIR")
    strict_models: bool = Field(False, alias="AGRI_STRICT_MODELS")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def api_key_set(self) -> set[str]:
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def config_path(self) -> Path:
        candidate = Path(self.config_dir)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    @property
    def registry_path(self) -> Path:
        candidate = Path(self.model_registry)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call get_settings.cache_clear() in tests."""
    return Settings()


class DataSpec(BaseModel):
    """Where the data comes from and which columns matter."""

    model_config = ConfigDict(extra="forbid")

    raw: str = Field(..., description="CSV path relative to the project root, or a directory of images")
    target: str | None = Field(None, description="Target column; None for unsupervised/inference modules")
    features: list[str] = Field(default_factory=list, description="Ordered model input columns")
    drop: list[str] = Field(default_factory=list, description="Columns excluded to prevent leakage")
    id_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    group_column: str | None = Field(None, description="Entity column for grouped/time-aware evaluation")
    dtype_map: dict[str, str] = Field(default_factory=dict)
    required_columns: list[str] = Field(default_factory=list)
    allow_missing: list[str] = Field(default_factory=list)
    max_missing_fraction: float = Field(0.0, ge=0.0, le=1.0)
    min_rows: int = Field(1, ge=1)
    image_size: tuple[int, int] | None = None
    classes: list[str] = Field(default_factory=list)

    @property
    def raw_path(self) -> Path:
        candidate = Path(self.raw)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


class SplitSpec(BaseModel):
    """How the dataset is partitioned. Time series must never be randomly split."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["stratified", "random", "time", "repeated_kfold", "group", "none"] = "stratified"
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    val_size: float = Field(0.0, ge=0.0, lt=1.0)
    n_splits: int = Field(5, ge=2)
    n_repeats: int = Field(1, ge=1)
    time_column: str | None = None
    train_end: str | int | float | None = Field(
        None, description="Inclusive end of the training window, e.g. 2014"
    )
    test_start: str | int | float | None = Field(
        None, description="Inclusive start of the test window, e.g. 2015"
    )
    holdout_years: int | None = Field(None, description="Hold out the most recent N periods")
    shuffle: bool = True

    @model_validator(mode="after")
    def _time_strategy_needs_column(self) -> SplitSpec:
        if self.strategy == "time" and not self.time_column:
            raise ValueError("split.strategy='time' requires split.time_column")
        return self


class ModelSpec(BaseModel):
    """Algorithm choice and hyperparameters. 'params' is passed to the estimator as-is."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["sklearn", "xgboost", "torch", "statsmodels", "physics", "external"] = "sklearn"
    algorithm: str
    params: dict[str, Any] = Field(default_factory=dict)
    calibrate: bool = Field(False, description="Wrap classifiers in probability calibration")
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    feature_importance: bool = True
    uncertainty: Literal["none", "quantile", "conformal", "ensemble_std"] = "none"
    target_transform: Literal["none", "log", "log1p", "sqrt"] = "none"
    class_weight: str | None = None

    @field_validator("params")
    @classmethod
    def _no_nested_estimators(cls, value: dict[str, Any]) -> dict[str, Any]:
        # A nested estimator in YAML would be un-serialisable and unreviewable.
        for key, item in value.items():
            if item.__class__.__name__.endswith(("Classifier", "Regressor", "Forest")):
                raise ValueError(f"model.params.{key} must be a literal, not an estimator instance")
        return value


class GateSpec(BaseModel):
    """The metric floor CI enforces. A build fails if a model drops below it."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str
    minimum: float | None = None
    maximum: float | None = Field(None, description="For error metrics such as RMSE or MAPE")
    secondary_metrics: list[str] = Field(default_factory=list)
    #: Floors applied to named secondary metrics, e.g. {"directional_accuracy": 0.5}.
    #: A module whose headline metric is an error score often also needs a second,
    #: independent bar - beating the baseline on mean error while having no
    #: directional skill is not a useful forecaster.
    secondary_minimums: dict[str, float] = Field(default_factory=dict)
    must_beat_baseline: bool = True
    baseline: str | None = Field(None, description="BaselineKind value; inferred when null")

    def check(self, value: float) -> tuple[bool, str]:
        """Return (passed, human explanation)."""
        if self.minimum is not None and value < self.minimum:
            return False, f"{self.primary_metric}={value:.4f} is below the floor {self.minimum}"
        if self.maximum is not None and value > self.maximum:
            return False, f"{self.primary_metric}={value:.4f} exceeds the ceiling {self.maximum}"
        bound = self.minimum if self.minimum is not None else self.maximum
        return True, f"{self.primary_metric}={value:.4f} satisfies the bound {bound}"


class ServingSpec(BaseModel):
    """How the module behaves behind the API."""

    model_config = ConfigDict(extra="forbid")

    latency_budget_ms: int = Field(500, ge=1)
    confidence_high: float = Field(0.80, ge=0.0, le=1.0)
    confidence_medium: float = Field(0.60, ge=0.0, le=1.0)
    cacheable: bool = False
    warmup: bool = True
    allow_fallback: bool = Field(
        True, description="When the artefact is missing, return a documented degraded answer instead of 503"
    )

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> ServingSpec:
        if self.confidence_medium > self.confidence_high:
            raise ValueError("serving.confidence_medium must not exceed serving.confidence_high")
        return self


class ModuleConfig(BaseModel):
    """Complete, validated configuration for one module."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0.0"
    task: TaskType
    description: str = ""
    owner: str = "data-science"
    seed: int = 42
    data: DataSpec
    split: SplitSpec = Field(default_factory=SplitSpec)
    model: ModelSpec
    gates: GateSpec
    serving: ServingSpec = Field(default_factory=ServingSpec)

    # Path of the YAML this came from, for provenance in the registry manifest.
    source_path: str | None = None

    @field_validator("name")
    @classmethod
    def _snake_case(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
            raise ValueError(f"module name '{value}' must be lower_snake_case")
        return value

    @property
    def config_digest(self) -> str:
        """Stable digest of the model-relevant config, recorded in the manifest."""
        payload = self.model_dump_json(exclude={"description", "owner", "source_path"})
        return sha256_text(payload)

    def artifact_dir(self, registry_root: Path | None = None) -> Path:
        root = registry_root or get_settings().registry_path
        return root / self.name / self.version


def load_module_config(name: str, config_dir: Path | None = None) -> ModuleConfig:
    """Load, interpolate and validate configs/<name>.yaml.

    Raises ConfigError with the file path so a bad YAML is obvious in CI.
    """
    directory = config_dir or get_settings().config_path or CONFIG_DIR
    path = Path(directory) / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"No configuration file for module '{name}'", context={"expected": str(path)})

    try:
        raw_text = interpolate_env(path.read_text(encoding="utf-8"))
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(payload).__name__}")

    payload["source_path"] = str(path)
    try:
        config = ModuleConfig.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError carries the useful detail
        raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc

    if config.name != name:
        raise ConfigError(
            f"{path} declares name '{config.name}' but was loaded as '{name}'",
            context={"expected": name, "actual": config.name},
        )
    return config


def available_module_configs(config_dir: Path | None = None) -> list[str]:
    """Names of every module that has a YAML config, sorted."""
    directory = Path(config_dir or get_settings().config_path or CONFIG_DIR)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))
