"""fertilizer: recommend a fertilizer blend.

The module declares only what makes it different: the request shape, the response
type, the categorical columns, and the sample payload used by the CLI. Training,
artefact resolution, envelope construction and gate enforcement all live in
agri.common.tabular and agri.common.training, so every module behaves uniformly.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agri.common.schemas import Prediction
from agri.common.tabular import TabularPredictor
from agri.features import CategoryIndexEncoder, ColumnSelector, Pipeline


class Request(BaseModel):
    """One prediction request. Every field maps to a declared feature column."""

    Temparature: float = Field(..., ge=-30, le=60, description="temperature, degrees C")
    Humidity: float = Field(..., ge=0, le=100, description="relative humidity, percent")
    Moisture: float = Field(..., ge=0, le=100, description="soil moisture, percent")
    Soil_Type: str = Field(..., description="Sandy, Loamy, Black, Red or Clayey")
    Crop_Type: str = Field(..., description="crop being grown")
    Nitrogen: float = Field(..., ge=0, description="soil nitrogen")
    Potassium: float = Field(..., ge=0, description="soil potassium")
    Phosphorous: float = Field(..., ge=0, description="soil phosphorus")


class Response(Prediction[str]):
    """Response envelope for fertilizer."""


class Predictor(TabularPredictor):
    """Serves the fertilizer artefact."""

    module_name: ClassVar[str] = "fertilizer"
    request_model = Request
    response_model = Response
    field_map: ClassVar[dict[str, str]] = {"Soil_Type": "Soil Type", "Crop_Type": "Crop Type"}
    #: Columns encoded to integer codes with a vocabulary fitted at train time.
    categorical_columns: ClassVar[list[str]] = ["Soil Type", "Crop Type"]

    @classmethod
    def build_feature_pipeline(cls, config: Any, frame: Any) -> Pipeline:
        """Select the declared features and encode categoricals.

        The encoder is fitted on the training partition only, then travels inside
        the serialised bundle, so serving reuses the identical vocabulary.
        """
        steps: list[Any] = [ColumnSelector(config.data.features, drop=config.data.drop)]
        if cls.categorical_columns:
            steps.append(CategoryIndexEncoder.fit(frame, cls.categorical_columns))
        return Pipeline(steps=steps)


SAMPLE: dict[str, Any] = {
    "Temparature": 26,
    "Humidity": 52,
    "Moisture": 38,
    "Soil_Type": "Sandy",
    "Crop_Type": "Maize",
    "Nitrogen": 37,
    "Potassium": 0,
    "Phosphorous": 0,
}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("fertilizer", Predictor, SAMPLE))
