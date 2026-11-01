"""soil_image module: classify a soil photograph into one of ten soil types.

Predictor, Request and Response are resolved lazily (PEP 562) rather than imported
eagerly. An eager import here would make python -m agri.modules.soil_image.predict
execute predict.py twice - once as a package attribute and once as __main__ - which
Python warns about and which produces two distinct Predictor classes.

Data honesty: the training corpus in data/raw/soil_images is PROCEDURALLY GENERATED
(synthetic texture synthesis, see that directory's README.md and
generate_soil_images.py). It is not field photography. Every artefact, metric
report and response from this module records that fact, and the reported macro_f1
must be read as separability of synthetic textures, never as field accuracy.
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
