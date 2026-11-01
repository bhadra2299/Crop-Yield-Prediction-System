"""Irrigation scheduling module.

Predictor, Request and Response are resolved lazily (PEP 562) so that running
python -m agri.modules.irrigation.predict does not import predict.py twice.
"""

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
