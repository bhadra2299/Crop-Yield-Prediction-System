"""risk: composite risk band from yield, rainfall, area and input intensity.

This module is different from the other tabular modules in one important way:
the dataset has no pre-labelled risk column, so the risk band is derived from a
declared, auditable rule set rather than from a human annotator.

The rule is deliberately simple and stated in full: a field is High risk when
its yield sits in the bottom quartile of comparable fields (same crop and season),
Medium in the next quartile, and Low otherwise. A crop-relative yardstick is used
because absolute yield is meaningless across 54 crops - 0.8 t/ha is excellent for
pulses and disastrous for sugarcane.

Because the label is derived from Yield, Yield itself must not be a feature, or
the model would simply relearn the rule it was taught. It learns instead from
the conditions that precede harvest: rainfall, area, inputs and year.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agri.common.schemas import Prediction
from agri.common.tabular import TabularPredictor
from agri.features import CategoryIndexEncoder, ColumnSelector, Pipeline

#: Boundaries in the crop-and-season-relative yield distribution.
HIGH_QUANTILE = 0.25
MEDIUM_QUANTILE = 0.50

RISK_COLUMN = "risk_band"
FERT_PER_HA = "fertilizer_per_hectare"
PEST_PER_HA = "pesticide_per_hectare"


def derive_risk_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the derived risk band. Deterministic and identical in train and serve.

    Grouping by crop and season means a pulse and a cane field are judged against
    their own peers rather than against each other.
    """
    frame = frame.copy()
    if "Yield" not in frame.columns:
        return frame

    group_keys = [column for column in ("Crop", "Season") if column in frame.columns]
    if group_keys:
        grouped = frame.groupby(group_keys)["Yield"]
        high_cut = grouped.transform(lambda values: values.quantile(HIGH_QUANTILE))
        medium_cut = grouped.transform(lambda values: values.quantile(MEDIUM_QUANTILE))
    else:
        high_cut = pd.Series(frame["Yield"].quantile(HIGH_QUANTILE), index=frame.index)
        medium_cut = pd.Series(frame["Yield"].quantile(MEDIUM_QUANTILE), index=frame.index)

    bands = np.where(
        frame["Yield"] <= high_cut,
        "High",
        np.where(frame["Yield"] <= medium_cut, "Medium", "Low"),
    )
    frame[RISK_COLUMN] = bands
    return frame


class Request(BaseModel):
    """Pre-harvest conditions for one field."""

    Crop: str = Field(..., description="crop name")
    Season: str = Field(..., description="Kharif, Rabi, Summer, Whole Year, Autumn or Winter")
    State: str = Field(..., description="Indian state")
    Crop_Year: int = Field(..., ge=1960, le=2100, description="cropping year")
    Area: float = Field(..., gt=0, description="sown area, hectares")
    Annual_Rainfall: float = Field(..., ge=0, le=12000, description="annual rainfall, mm")
    Fertilizer: float = Field(..., ge=0, description="fertilizer applied, kg")
    Pesticide: float = Field(..., ge=0, description="pesticide applied, kg")


class Response(Prediction[str]):
    """Risk band plus the factors that drove it."""

    contributing_factors: dict[str, float] = Field(
        default_factory=dict, description="Feature importances for the factors that moved this prediction"
    )


class Predictor(TabularPredictor):
    """Serves the risk model."""

    module_name: ClassVar[str] = "risk"
    request_model = Request
    response_model = Response
    field_map: ClassVar[dict[str, str]] = {}
    categorical_columns: ClassVar[list[str]] = ["Crop", "Season", "State"]

    @classmethod
    def prepare_features(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Derive the label and the input-intensity ratios, in that order.

        The ratios matter: total Fertilizer and Pesticide are state-level totals that
        mostly encode how big the state is, while the amount applied per hectare is
        the actual agronomic decision and is available before harvest. Without them
        the model has very little to work with, because the label is yield-derived
        and yield itself is deliberately excluded.
        """
        frame = frame.copy()
        area = frame["Area"].replace(0, np.nan) if "Area" in frame.columns else None
        if area is not None:
            frame[FERT_PER_HA] = (frame["Fertilizer"] / area).fillna(0.0)
            frame[PEST_PER_HA] = (frame["Pesticide"] / area).fillna(0.0)
        if RISK_COLUMN not in frame.columns:
            frame = derive_risk_band(frame)
        return frame

    @classmethod
    def build_feature_pipeline(cls, config: Any, frame: Any) -> Pipeline:
        steps: list[Any] = [ColumnSelector(config.data.features, drop=config.data.drop)]
        if cls.categorical_columns:
            steps.append(CategoryIndexEncoder.fit(frame, cls.categorical_columns))
        return Pipeline(steps=steps)


SAMPLE: dict[str, Any] = {
    "Crop": "Rice",
    "Season": "Kharif",
    "State": "Karnataka",
    "Crop_Year": 2015,
    "Area": 73814,
    "Annual_Rainfall": 2051.4,
    "Fertilizer": 7024878.38,
    "Pesticide": 22882.34,
}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("risk", Predictor, SAMPLE))
