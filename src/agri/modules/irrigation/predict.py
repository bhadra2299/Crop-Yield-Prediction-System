"""irrigation: FAO-56 soil water balance irrigation scheduling.

This module has no trained artefact, and that is the point. Irrigation depth at a
given field on a given day is a deterministic physical calculation, not a
statistical inference. Fitting a model to 200 rows of synthetic pump on/off data
would produce a number that looks learned and is less defensible than the
published equation it replaced.

So the module serves FAO-56 (see agri.features.physics) and reports:
  * whether irrigation is needed, by comparing root-zone depletion against RAW
  * how much to apply, capped by the application limit
  * the full arithmetic, so an agronomist can check it by hand

What it does NOT do is claim an accuracy percentage. There is no ground truth
for "correct irrigation depth" on a public dataset, so the module is validated
against FAO-56 reference values and physical closure, not against field truth.
A sensor-derived error can only be computed once a real field series exists.
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agri.common.base import BasePredictor
from agri.common.errors import DataValidationError
from agri.common.schemas import Confidence, Prediction
from agri.common.seed import set_global_seed
from agri.features.physics import (
    DailyWeather,
    IrrigationRecommendation,
    SoilProfile,
    depletion_from_moisture,
    recommend_irrigation,
)

DEFAULT_LATITUDE = 17.385  # Hyderabad; overridden per request


class SoilInput(BaseModel):
    """Soil water properties. Defaults are a medium-textured loam."""

    field_capacity: float = Field(
        0.32, gt=0, le=1.0, description="volumetric water content at field capacity"
    )
    wilting_point: float = Field(0.14, ge=0, lt=1.0, description="volumetric water content at wilting point")
    root_depth_m: float = Field(0.6, gt=0, le=3.0, description="effective rooting depth, metres")
    depletion_fraction: float = Field(
        0.55, gt=0, le=1.0, description="readily available water fraction p (FAO-56 Table 22)"
    )


class Request(BaseModel):
    """One irrigation decision.

    Supply either current moisture as a sensor reading (preferred, because it is
    measured) or an explicit depletion. When moisture is given it is converted to
    depletion through the same soil parameters, so the two paths agree.
    """

    crop: str = Field("generic", description="crop name, used for labelling only")
    latitude: float = Field(DEFAULT_LATITUDE, ge=-90, le=90, description="field latitude, degrees")
    elevation_m: float = Field(0.0, ge=-430, le=6000, description="field elevation, metres")

    soil: SoilInput = Field(default_factory=SoilInput)

    current_moisture: float | None = Field(
        None, ge=0, le=1.0, description="sensor volumetric moisture, m^3/m^3"
    )
    depletion_mm: float | None = Field(None, ge=0, le=500, description="current root-zone depletion, mm")

    forecast_rain_mm: float = Field(0.0, ge=0, le=500, description="rain expected before the crop suffers")

    #: Optional forecast series. When supplied it is summarised into an ETc-driven
    #: depletion projection, which is a better predictor of tomorrow's deficit
    #: than today's rain forecast alone.
    forecast: list[dict[str, float]] = Field(
        default_factory=list,
        description="daily forecast entries with t_max, t_min, rainfall and optionally humidity_mean",
    )


class Response(Prediction[float]):
    """Irrigation depth in mm plus the reasoning and the water-balance figures."""

    value: float = Field(
        ..., description="recommended application depth, mm (0 when no irrigation is needed)"
    )
    should_irrigate: bool = Field(..., description="whether irrigation is recommended at all")
    urgency: str = Field(..., description="none, monitor, soon or now")
    taw_mm: float = Field(..., description="total available water in the root zone, mm")
    raw_mm: float = Field(..., description="readily available water, mm")
    depletion_mm: float = Field(..., description="current root-zone depletion, mm")
    projected_depletion_mm: float | None = Field(
        None, description="depletion projected forward over the supplied forecast, mm"
    )
    steps: list[str] = Field(default_factory=list, description="the arithmetic behind the recommendation")


class Predictor(BasePredictor[Request, Response]):
    """Serves FAO-56 irrigation scheduling. No trained artefact is involved."""

    module_name: ClassVar[str] = "irrigation"
    requires_artifact: ClassVar[bool] = False
    request_model = Request
    response_model = Response

    def _load(self) -> None:
        set_global_seed(self.config.seed)
        params = self.config.model.params
        # Config values act as the defaults the service layer supplies when a
        # request omits soil details; per-request values always win.
        self.defaults: dict[str, float] = {
            "field_capacity": float(params.get("field_capacity", 0.32)),
            "wilting_point": float(params.get("wilting_point", 0.14)),
            "root_depth_m": float(params.get("root_depth_m", 0.6)),
            "depletion_fraction": float(params.get("depletion_fraction_p", 0.55)),
        }
        self.max_application_mm = float(params.get("max_application_mm", 40.0))
        self.effective_rain_fraction = float(params.get("effective_rain_fraction", 0.8))

    # ------------------------------------------------------------------ helpers
    def _soil(self, request: Request) -> SoilProfile:
        """Merge the request's soil block over the configured defaults."""
        soil = request.soil
        return SoilProfile(
            field_capacity=soil.field_capacity,
            wilting_point=soil.wilting_point,
            root_depth=soil.root_depth_m,
            depletion_fraction=soil.depletion_fraction,
        )

    def _current_depletion(self, request: Request, soil: SoilProfile) -> float:
        """Resolve depletion from whichever input the caller supplied."""
        if request.current_moisture is not None:
            return depletion_from_moisture(request.current_moisture, soil)
        if request.depletion_mm is not None:
            return float(request.depletion_mm)
        raise DataValidationError(
            "Supply either current_moisture or depletion_mm; without one there is "
            "no root-zone state to schedule against",
            context={"module": self.module_name},
        )

    def _project(self, request: Request, soil: SoilProfile, depletion_mm: float) -> float | None:
        """Project depletion forward across the forecast, ignoring new irrigation.

        The projection answers the question that actually matters for scheduling:
        will the crop cross the stress threshold before the next opportunity to
        irrigate? It is a projection, not a prediction of the future.
        """
        if not request.forecast:
            return None

        from agri.features.physics import compute_et0, compute_ks, compute_raw, compute_taw

        taw = compute_taw(soil.field_capacity, soil.wilting_point, soil.root_depth)
        raw = compute_raw(taw, soil.depletion_fraction)
        depletion = depletion_mm
        start = date.today()
        for index, entry in enumerate(request.forecast):
            day = start.fromordinal(start.toordinal() + index)
            weather = DailyWeather(
                t_max=float(entry.get("t_max", 30.0)),
                t_min=float(entry.get("t_min", 20.0)),
                humidity_mean=entry.get("humidity_mean"),
                wind_speed_2m=float(entry.get("wind_speed_2m", 2.0)),
                solar_radiation=entry.get("solar_radiation"),
                rainfall=float(entry.get("rainfall", 0.0)),
                elevation=request.elevation_m,
            )
            et0 = compute_et0(weather, request.latitude, day.timetuple().tm_yday)
            kc = float(entry.get("kc", 1.0))
            ks = compute_ks(depletion, raw, taw)
            depletion = depletion - max(weather.rainfall, 0.0) * self.effective_rain_fraction + kc * ks * et0
        return max(depletion, 0.0)

    def _confidence(self, request: Request, recommendation: IrrigationRecommendation) -> Confidence:
        """Confidence reflects input quality, not model certainty.

        A physics calculation is exact given its inputs; what varies is how much
        the inputs are trusted. A measured moisture reading with a forecast is
        high confidence, a bare depletion value with no forecast is medium, and a
        single reading with no forecast and no soil detail is low.
        """
        has_measurement = request.current_moisture is not None
        has_forecast = bool(request.forecast)
        if has_measurement and has_forecast:
            return Confidence.HIGH
        if has_measurement or has_forecast:
            return Confidence.MEDIUM
        return Confidence.LOW

    # ------------------------------------------------------------------ inference
    def _predict(self, request: Request) -> Response:
        soil = self._soil(request)
        depletion = self._current_depletion(request, soil)

        recommendation = recommend_irrigation(
            depletion,
            soil,
            forecast_rain_mm=request.forecast_rain_mm,
            max_application_mm=self.max_application_mm,
            effective_rain_fraction=self.effective_rain_fraction,
        )

        projected = self._project(request, soil, depletion)
        steps = list(recommendation.steps)
        if projected is not None:
            steps.append(
                f"projected depletion after {len(request.forecast)} forecast day(s) = {projected:.1f} mm"
            )
            # Bring tomorrow's problem forward: if the projection crosses RAW but
            # today's state has not, the honest answer is to schedule soon rather
            # than to declare the field fine.
            if not recommendation.should_irrigate and projected > recommendation.raw_mm:
                follow_up = recommend_irrigation(
                    projected,
                    soil,
                    forecast_rain_mm=0.0,
                    max_application_mm=self.max_application_mm,
                    effective_rain_fraction=self.effective_rain_fraction,
                )
                recommendation = IrrigationRecommendation(
                    should_irrigate=follow_up.should_irrigate,
                    depth_mm=follow_up.depth_mm,
                    reason=(
                        "The crop is not stressed today, but the forecast drives the root zone past "
                        "its readily available water before the next practical opportunity to irrigate. "
                        + follow_up.reason
                    ),
                    urgency="soon",
                    depletion_mm=recommendation.depletion_mm,
                    raw_mm=recommendation.raw_mm,
                    taw_mm=recommendation.taw_mm,
                    forecast_rain_mm=recommendation.forecast_rain_mm,
                    steps=steps,
                )

        confidence = self._confidence(request, recommendation)
        reasons = [
            recommendation.reason,
            f"Soil holds {recommendation.taw_mm:.0f} mm of available water, of which "
            f"{recommendation.raw_mm:.0f} mm is readily available before the crop works harder to extract it.",
        ]
        if request.current_moisture is not None:
            reasons.append(
                f"Depletion was derived from a measured moisture reading of "
                f"{request.current_moisture:.3f} m3/m3 against field capacity {soil.field_capacity:.2f}."
            )
        else:
            reasons.append(
                "Depletion was supplied directly rather than measured, so it carries whatever error "
                "the estimate that produced it had."
            )
        if not request.forecast:
            reasons.append("No forecast was supplied, so this decision reflects only the current soil state.")

        return Response(
            value=recommendation.depth_mm,
            should_irrigate=recommendation.should_irrigate,
            urgency=recommendation.urgency,
            taw_mm=recommendation.taw_mm,
            raw_mm=recommendation.raw_mm,
            depletion_mm=recommendation.depletion_mm,
            projected_depletion_mm=projected,
            steps=steps,
            confidence_score=None,
            confidence=confidence,
            reasons=reasons,
            model_name=self.module_name,
            model_version=self.version,
        )


SAMPLE: dict[str, Any] = {
    "crop": "Rice",
    "latitude": 17.385,
    "elevation_m": 542,
    "soil": {
        "field_capacity": 0.32,
        "wilting_point": 0.14,
        "root_depth_m": 0.6,
        "depletion_fraction": 0.55,
    },
    "current_moisture": 0.18,
    "forecast_rain_mm": 0.0,
    "forecast": [
        {"t_max": 34.0, "t_min": 24.0, "humidity_mean": 45.0, "rainfall": 0.0, "kc": 1.05},
        {"t_max": 35.0, "t_min": 25.0, "humidity_mean": 40.0, "rainfall": 0.0, "kc": 1.05},
    ],
}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("irrigation", Predictor, SAMPLE))
