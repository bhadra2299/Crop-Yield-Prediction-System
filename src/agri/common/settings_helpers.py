"""Small helpers over Settings that avoid duplicating environment logic.

These exist so that modules can ask a question about configuration - "is a
provider key present?" - without each re-implementing the environment lookup, and
without ever echoing the secret itself.
"""

from __future__ import annotations

import os

from agri.common.config import get_settings


def openweather_key_status() -> bool:
    """Whether an OpenWeather key is configured. Never returns the key itself."""
    try:
        return bool(get_settings().openweather_api_key)
    except Exception:
        return bool(os.getenv("OPENWEATHER_API_KEY"))


def describe_environment() -> dict[str, str | bool]:
    """A safe, secret-free summary of the active environment, for /v1/health."""
    settings = get_settings()
    return {
        "env": settings.env,
        "log_level": settings.log_level,
        "log_json": settings.log_json,
        "strict_models": settings.strict_models,
        "auth_enabled": bool(settings.api_key_set),
        "openweather_configured": bool(settings.openweather_api_key),
        "model_registry": str(settings.registry_path),
        "config_dir": str(settings.config_dir),
    }
