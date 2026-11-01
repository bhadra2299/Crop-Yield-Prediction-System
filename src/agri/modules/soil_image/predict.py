"""soil_image: classify a photograph into one of ten soil types.

What this module is
-------------------
A mobilenet_v2 classifier fine-tuned on the corpus in data/raw/soil_images. The
artefact is a torch state dict plus class order, input geometry and ImageNet
normalisation constants, registered through agri.common.registry as model.pt with a
JSON manifest beside it. Serving rebuilds the architecture and loads the weights; it
never fits anything.

What this module is NOT
-----------------------
It is not a field-validated soil classifier. The corpus is PROCEDURALLY GENERATED
(fractal noise and simulated texture, see data/raw/soil_images/generate_soil_images.py),
so the held-out macro_f1 measures how separable those synthetic textures are. It is
not an accuracy estimate for photographs of real soil, and the manifest, metrics
report and every response say so.

Torch is an optional extra. Importing this package without it succeeds; loading the
predictor then fails with a clear ConfigError instead of an ImportError traceback.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field, model_validator

from agri.common.base import BasePredictor
from agri.common.errors import ModuleExecutionError
from agri.common.schemas import Confidence, Prediction

from . import provenance
from .imaging import decode_request_image
from .model import build_model, build_transforms, ensure_torch, load_state_dict
from .training import SoilImageArtifact

#: Probability margin below which the top two classes are described as ambiguous.
#: This only shapes the human-readable reason; it never changes the prediction.
AMBIGUOUS_MARGIN = 0.15


class Request(BaseModel):
    """One soil classification request.

    The image arrives either as base64 (what the HTTP layer forwards after reading
    an upload) or as a server-side path (CLI and offline scripts). Exactly one must
    be supplied; the validator rejects both or neither before any decoding work.
    """

    image_base64: str | None = Field(
        None, description="base64-encoded JPEG/PNG bytes; a data: URL prefix is accepted"
    )
    image_path: str | None = Field(
        None, description="server-side image path, relative to the project root; CLI use only"
    )
    top_k: int = Field(3, ge=1, le=10, description="how many classes to return in class_probabilities")

    @model_validator(mode="after")
    def _exactly_one_image(self) -> Request:
        supplied = [
            name
            for name, value in (("image_base64", self.image_base64), ("image_path", self.image_path))
            if value
        ]
        if len(supplied) != 1:
            raise ValueError("supply exactly one of image_base64 or image_path")
        return self


class Response(Prediction[str]):
    """Predicted soil class plus the honest caveats that apply to it."""

    value: str = Field(..., description="predicted soil class name, one of the configured ten")
    class_probabilities: dict[str, float] = Field(
        default_factory=dict, description="softmax probability for the top-k classes"
    )
    top_k: int = Field(3, description="number of classes included in class_probabilities")
    synthetic_training_data: bool = Field(
        True, description="true when the model was trained on procedurally generated images"
    )
    limitations: list[str] = Field(
        default_factory=list, description="caveats the caller must apply to this prediction"
    )


class Predictor(BasePredictor[Request, Response]):
    """Serves the registered soil_image artefact. Loads weights; never trains."""

    module_name: ClassVar[str] = "soil_image"
    #: A registered artefact is required: there is no meaningful untrained fallback.
    requires_artifact: ClassVar[bool] = True
    request_model = Request
    response_model = Response
    artifact_filename: ClassVar[str] = "model.pt"

    def _load(self) -> None:
        """Rebuild mobilenet_v2 and load the registered state dict.

        The BasePredictor contract has already resolved the registry entry and
        deserialised the artefact into self.artifact; this hook only turns that into
        a ready model. Anything undefined here must fail loudly through AgriError so
        /v1/health can name the reason.
        """
        ensure_torch()

        artifact = self.artifact
        if not isinstance(artifact, SoilImageArtifact):
            raise ModuleExecutionError(
                f"{self.module_name} artefact is {type(artifact).__name__}, expected SoilImageArtifact"
            )

        self.classes: list[str] = [str(name) for name in artifact.classes]
        if not self.classes:
            raise ModuleExecutionError(f"{self.module_name} artefact contains no class names")
        self.input_size: tuple[int, int] = (int(artifact.image_size[0]), int(artifact.image_size[1]))
        self.mean = tuple(float(value) for value in artifact.mean)
        self.std = tuple(float(value) for value in artifact.std)
        self.synthetic_training_data = bool(artifact.synthetic_data)
        self.device = str(self.config.model.params.get("device", "cpu"))

        self.model = build_model(
            len(self.classes), pretrained=False, dropout=float(artifact.dropout), device=self.device
        )
        load_state_dict(self.model, artifact.state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.transform = build_transforms(self.input_size, self.mean, self.std, train=False)

    def _predict(self, request: Request) -> Response:
        """Run one forward pass and shape the response envelope."""
        import torch

        image = decode_request_image(request.image_base64, request.image_path)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()[0]

        order = np.argsort(probabilities)[::-1]
        top_k = min(int(request.top_k), len(self.classes))
        ranked = [(self.classes[int(index)], float(probabilities[int(index)])) for index in order[:top_k]]
        value, top_probability = ranked[0]
        alternatives = (
            [{"class": name, "score": round(score, 6)} for name, score in ranked[1:]]
            if top_probability < float(self.config.serving.confidence_high)
            else None
        )
        reasons = [
            f"The model assigned class '{value}' the highest probability ({top_probability:.1%}).",
        ]
        if len(ranked) > 1:
            runner_up, runner_up_score = ranked[1]
            margin = top_probability - runner_up_score
            if margin < AMBIGUOUS_MARGIN:
                reasons.append(
                    f"The next most likely class is '{runner_up}' at {runner_up_score:.1%}; "
                    f"a margin of {margin:.1%} means the texture is ambiguous to this model."
                )
            else:
                reasons.append(
                    f"The next most likely class is '{runner_up}' at {runner_up_score:.1%}, "
                    f"{margin:.1%} behind the top class."
                )
        reasons.append(
            "Confidence reflects the softmax probability only. It is not a measure of how well "
            "the model would do on a photograph from a real field."
        )

        return Response(
            value=value,
            class_probabilities={name: round(score, 6) for name, score in ranked},
            top_k=top_k,
            confidence_score=top_probability,
            confidence=Confidence.UNKNOWN,  # finalised from the configured thresholds
            alternatives=alternatives,
            reasons=reasons,
            synthetic_training_data=self.synthetic_training_data,
            limitations=[provenance.SERVING_CAVEAT, provenance.LIMITATION],
            model_name=self.module_name,
            model_version=self.version,
        )


#: CLI sample. The path is a fixture pointing at a corpus image, not a default
#: input: the module never picks an image on its own.
SAMPLE: dict[str, Any] = {
    "image_path": "data/raw/soil_images/loamy_soil/loamy_soil_0001.jpg",
    "top_k": 3,
}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("soil_image", Predictor, SAMPLE))
