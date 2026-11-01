"""Prediction endpoints.

Every endpoint follows the same three steps: resolve the module, call predict,
record the call for monitoring. The router deliberately contains no business
logic - the orchestration lives in agri.orchestrator and the modules own their
own domain rules.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import ValidationError

from agri.api.registry_cache import get_cache
from agri.api.security import enforce_rate_limit, require_api_key
from agri.common.errors import DataValidationError, ModelNotTrainedError
from agri.common.logging import get_logger
from agri.common.schemas import Prediction
from agri.monitoring.prediction_log import record_prediction

_log = get_logger("api.predict")

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)


def _resolve(module: str) -> Any:
    """Return a ready predictor or raise the right typed error."""
    predictor = get_cache().get(module)
    if predictor is None:
        reason = get_cache().load_failure(module) or "module is not available"
        _log.warning("predict requested for unavailable module", extra={"module": module})
        raise ModelNotTrainedError(
            f"The {module} module is not available: {reason}",
            context={"module": module, "hint": "Check /v1/health for per-module status"},
        )
    return predictor


def _serve(module: str, request_model: Any, payload: Any, endpoint: str) -> Prediction:
    predictor = _resolve(module)
    # A malformed body is the caller's fault and must be a 4xx. Building the request
    # model directly let a pydantic ValidationError (or a TypeError for an unexpected
    # field) escape as an unhandled 500, which both hides the real problem and inflates
    # the server error rate for what is a client mistake.
    try:
        request = request_model(**payload)
    except ValidationError as exc:
        raise DataValidationError(
            f"Request body does not match the {module} contract",
            context={"errors": exc.errors(include_url=False)},
        ) from exc
    except TypeError as exc:
        raise DataValidationError(
            f"Request body does not match the {module} contract",
            context={"error": str(exc)},
        ) from exc
    prediction = predictor.predict(request)
    record_prediction(module=module, endpoint=endpoint, request=payload, prediction=prediction, ok=True)
    return prediction


# --------------------------------------------------------------------- tabular
@router.post(
    "/crop-recommendation",
    response_model=None,
    summary="Recommend a crop from soil chemistry and climate",
)
async def crop_recommendation(payload: dict) -> dict:
    from agri.modules.crop_recommendation.predict import Predictor, Request

    del Predictor
    return _serve("crop_recommendation", Request, payload, "crop-recommendation").model_dump()


@router.post("/yield-prediction", summary="Predict crop yield in tonnes per hectare")
async def yield_prediction(payload: dict) -> dict:
    from agri.modules.yield_prediction.predict import Request

    return _serve("yield_prediction", Request, payload, "yield-prediction").model_dump()


@router.post("/fertilizer", summary="Recommend a fertilizer blend")
async def fertilizer(payload: dict) -> dict:
    from agri.modules.fertilizer.predict import Request

    return _serve("fertilizer", Request, payload, "fertilizer").model_dump()


@router.post("/soil-nutrient", summary="Estimate soil nitrogen, phosphorus and potassium")
async def soil_nutrient(payload: dict) -> dict:
    from agri.modules.soil_nutrient.predict import Request

    return _serve("soil_nutrient", Request, payload, "soil-nutrient").model_dump()


@router.post("/risk", summary="Composite field risk band")
async def risk(payload: dict) -> dict:
    from agri.modules.risk.predict import Request

    return _serve("risk", Request, payload, "risk").model_dump()


@router.post("/market-price", summary="Forecast a mandi price as a relative change")
async def market_price(payload: dict) -> dict:
    from agri.modules.market_price.predict import Request

    return _serve("market_price", Request, payload, "market-price").model_dump()


# ---------------------------------------------------------- physics / external
@router.post("/irrigation", summary="FAO-56 irrigation scheduling")
async def irrigation(payload: dict) -> dict:
    from agri.modules.irrigation.predict import Request

    return _serve("irrigation", Request, payload, "irrigation").model_dump()


@router.post("/weather", summary="Live weather with FAO-56 reference ET")
async def weather(payload: dict) -> dict:
    from agri.modules.weather.predict import Request

    return _serve("weather", Request, payload, "weather").model_dump()


# ----------------------------------------------------------------------- image
@router.post("/soil-image", summary="Classify soil type from a photograph")
async def soil_image(payload: dict) -> dict:
    from agri.modules.soil_image.predict import Request

    return _serve("soil_image", Request, payload, "soil-image").model_dump()


@router.post("/disease", summary="Detect plant leaf disease from a photograph")
async def disease(payload: dict) -> dict:
    from agri.modules.disease.predict import Request

    return _serve("disease", Request, payload, "disease").model_dump()


# ------------------------------------------------------------------- composite
@router.post("/full-recommendation", summary="Orchestrated multi-module advisory")
async def full_recommendation(payload: dict) -> dict:
    """Run the whole advisory chain, degrading per module rather than failing whole."""
    from agri.orchestrator.chain import full_recommendation as run_chain

    result = run_chain(payload)
    record_prediction(
        module="orchestrator",
        endpoint="full-recommendation",
        request={"keys": sorted(payload.keys())},
        prediction=None,
        ok=result.get("complete", False),
    )
    return result
