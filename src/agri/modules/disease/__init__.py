"""disease module: detect plant leaf disease from a photograph using third-party weights.

Predictor, Request and Response are resolved lazily (PEP 562) rather than imported
eagerly. An eager import here would make python -m agri.modules.disease.predict
execute predict.py twice - once as a package attribute and once as __main__ - which
Python produces two distinct Predictor classes for.

This module performs inference only. It does not train, and it registers no
artefact: the weights are fetched once from the Hugging Face Hub and cached under
the directory named in configs/disease.yaml. They were trained on PlantVillage and
have NOT been validated on regional Indian varieties, so every response carries
that limitation.
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
