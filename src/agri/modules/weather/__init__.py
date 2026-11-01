"""Weather module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .predict import Predictor, Request, Response

__all__ = ["Predictor", "Request", "Response"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import predict

        return getattr(predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
