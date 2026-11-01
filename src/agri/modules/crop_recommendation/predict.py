"""crop_recommendation: recommend a crop from soil chemistry and climate.

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

    N: float = Field(..., ge=0, le=200, description="soil nitrogen, kg/ha")
    P: float = Field(..., ge=0, le=200, description="soil phosphorus, kg/ha")
    K: float = Field(..., ge=0, le=250, description="soil potassium, kg/ha")
    temperature: float = Field(..., ge=-30, le=60, description="mean temperature, degrees C")
    humidity: float = Field(..., ge=0, le=100, description="relative humidity, percent")
    ph: float = Field(..., ge=0, le=14, description="soil pH")
    rainfall: float = Field(..., ge=0, le=6000, description="seasonal rainfall, mm")


class Response(Prediction[str]):
    """Response envelope for crop_recommendation."""


class Predictor(TabularPredictor):
    """Serves the crop_recommendation artefact."""

    module_name: ClassVar[str] = "crop_recommendation"
    request_model = Request
    response_model = Response
    field_map: ClassVar[dict[str, str]] = {}
    #: Columns encoded to integer codes with a vocabulary fitted at train time.
    categorical_columns: ClassVar[list[str]] = []

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
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.88,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202.9,
}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("crop_recommendation", Predictor, SAMPLE))
