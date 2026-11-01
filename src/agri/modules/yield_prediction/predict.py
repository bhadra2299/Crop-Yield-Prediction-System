"""yield_prediction: predict crop yield in tonnes per hectare.

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

    Crop: str = Field(..., description="crop name as it appears in the dataset")
    Crop_Year: int = Field(..., ge=1960, le=2100, description="cropping year")
    Season: str = Field(..., description="Kharif, Rabi, Summer, Whole Year, Autumn or Winter")
    State: str = Field(..., description="Indian state")
    Area: float = Field(..., gt=0, description="sown area, hectares")
    Annual_Rainfall: float = Field(..., ge=0, le=12000, description="annual rainfall, mm")
    Fertilizer: float = Field(..., ge=0, description="fertilizer applied, kg")
    Pesticide: float = Field(..., ge=0, description="pesticide applied, kg")


class Response(Prediction[float]):
    """Response envelope for yield_prediction."""


class Predictor(TabularPredictor):
    """Serves the yield_prediction artefact."""

    module_name: ClassVar[str] = "yield_prediction"
    request_model = Request
    response_model = Response
    field_map: ClassVar[dict[str, str]] = {}
    #: Columns encoded to integer codes with a vocabulary fitted at train time.
    categorical_columns: ClassVar[list[str]] = ["Crop", "Season", "State"]

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
    "Crop": "Rice",
    "Crop_Year": 2015,
    "Season": "Kharif",
    "State": "Karnataka",
    "Area": 73814,
    "Annual_Rainfall": 2051.4,
    "Fertilizer": 7024878.38,
    "Pesticide": 22882.34,
}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("yield_prediction", Predictor, SAMPLE))
