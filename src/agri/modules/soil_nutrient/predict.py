"""soil_nutrient: estimate soil nitrogen, phosphorus and potassium.

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

    temperature: float = Field(..., ge=-30, le=60, description="mean temperature, degrees C")
    humidity: float = Field(..., ge=0, le=100, description="relative humidity, percent")
    ph: float = Field(..., ge=0, le=14, description="soil pH")
    rainfall: float = Field(..., ge=0, le=6000, description="seasonal rainfall, mm")


class Response(Prediction[float]):
    """Response envelope for soil_nutrient.

    value is soil nitrogen (kg/ha). This module predicts ONE nutrient, not three: the
    config targets N and drops P and K, and the response used to declare a dict which
    the regression head never produced. Phosphorus and potassium would each need their
    own model, and on this dataset they are entangled with the crop label rather than
    with climate and pH, so predicting them from these four inputs would be fitting the
    label rather than the soil.
    """


class Predictor(TabularPredictor):
    """Serves the soil_nutrient artefact."""

    module_name: ClassVar[str] = "soil_nutrient"
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


SAMPLE: dict[str, Any] = {"temperature": 20.88, "humidity": 82, "ph": 6.5, "rainfall": 202.9}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("soil_nutrient", Predictor, SAMPLE))
