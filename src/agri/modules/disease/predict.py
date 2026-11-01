"""disease: leaf disease detection from a photograph using third-party weights.

What this module is
-------------------
A thin, honest serving wrapper around a public image-classification checkpoint
(model id from configs/disease.yaml, currently
linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification). It performs
inference only: no training, no fine-tuning, no registry artefact. Weights are
downloaded once into the configured cache directory and reused.

The limitation you must not drop
--------------------------------
These weights are THIRD-PARTY. They were trained on the PlantVillage dataset,
which is laboratory-style leaf photography dominated by a handful of crops and
diseases. They have NOT been validated on regional Indian varieties, local disease
races, field lighting, soil background or phone cameras. Every response therefore
carries an explicit limitations list, and this module must never be presented as a
diagnosis. It is a screening hint that a human should confirm.

Failure behaviour
-----------------
If the weights cannot be obtained (offline first run, gated repo, no vision
extra), _load raises an AgriError - ExternalServiceError for an upstream/dependency
failure - so the orchestrator can degrade this one signal and still answer. It
never fabricates a prediction to look healthy.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field, model_validator

from agri.common.base import BasePredictor
from agri.common.errors import DataValidationError, ExternalServiceError
from agri.common.schemas import Confidence, Prediction

from .imaging import decode_request_image
from .weights import load_pretrained, require_vision_stack, resolve_cache_dir

#: Attached to every response. Wording is deliberately blunt.
THIRD_PARTY_LIMITATION = (
    "Predictions come from third-party weights trained on PlantVillage, a "
    "laboratory-style leaf dataset. They are NOT validated on regional crop "
    "varieties, local disease races, field lighting or phone cameras, and this "
    "module is a screening hint rather than a diagnosis."
)
NO_ACCURACY_CLAIM = (
    "No accuracy figure is reported for this module: no labelled regional dataset "
    "exists here, and PlantVillage performance does not transfer automatically."
)


class Request(BaseModel):
    """One leaf disease inference request."""

    image_base64: str | None = Field(
        None, description="base64-encoded JPEG/PNG leaf image; a data: URL prefix is accepted"
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
    """Predicted label plus the caveats that make it interpretable."""

    value: str = Field(..., description="predicted class label as published by the model")
    display_label: str = Field(
        "", description="the same label with the PlantVillage separators removed, for display"
    )
    class_probabilities: dict[str, float] = Field(
        default_factory=dict, description="averaged softmax probability for the top-k classes"
    )
    is_healthy: bool = Field(False, description="true when the winning label is the healthy class")
    crop: str | None = Field(None, description="crop parsed from the label, when the label names one")
    tta_views: int = Field(1, description="number of augmented views averaged into this prediction")
    weights_source: str = Field("", description="where the third-party weights were loaded from")
    limitations: list[str] = Field(
        default_factory=list, description="caveats the caller must apply to this prediction"
    )


class Predictor(BasePredictor[Request, Response]):
    """Serves the cached PlantVillage checkpoint. Inference only."""

    module_name: ClassVar[str] = "disease"
    #: No registry artefact: the weights are third-party and fetched into a cache.
    requires_artifact: ClassVar[bool] = False
    request_model = Request
    response_model = Response

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        """Fetch or reuse cached weights. Raises AgriError, never fabricates output."""
        require_vision_stack()
        params = self.config.model.params
        model_id = str(params.get("model_id") or "").strip()
        if not model_id:
            raise ExternalServiceError(
                "configs/disease.yaml does not name a model_id, so no weights can be resolved",
                context={"module": "disease"},
            )
        cache_dir = resolve_cache_dir(params.get("cache_dir"))
        self.device = str(params.get("device", "cpu"))
        self.model, self.preprocessor, self.id2label, self.weights_source = load_pretrained(
            model_id,
            cache_dir,
            allow_download=bool(params.get("allow_download", True)),
            device=self.device,
        )
        self.tta_crops = max(1, int(params.get("tta_crops", 3)))
        self.tta_flips = bool(params.get("tta_flips", True))
        self.low_confidence_threshold = float(params.get("low_confidence_threshold", 0.5))
        self.healthy_label_count = sum(1 for label in self.id2label.values() if "healthy" in label.lower())

    # ------------------------------------------------------------------ helpers
    def _torch(self) -> Any:
        """Import torch lazily so package import stays cheap and torch-optional."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by require_vision_stack
            raise ExternalServiceError(
                "disease inference needs torch, which is not installed",
                context={"module": "disease"},
            ) from exc
        return torch

    def build_views(self, image: Any) -> list[Any]:
        """Deterministic test-time augmentation: centre plus corner crops, optional flips.

        Crops are taken at 80 percent of the frame, which is the scale at which a
        leaf usually fills the photo. Averaging them reduces the sensitivity to how
        the leaf happened to be framed, which is the least controlled part of a
        farmer's upload.
        """
        width, height = image.size
        crop_w = int(width * 0.8)
        crop_h = int(height * 0.8)
        positions = [
            ((width - crop_w) // 2, (height - crop_h) // 2),
            (0, 0),
            (width - crop_w, 0),
            (0, height - crop_h),
            (width - crop_w, height - crop_h),
        ]
        views: list[Any] = []
        for left, top in positions:
            views.append(image.crop((left, top, left + crop_w, top + crop_h)))
            if len(views) >= self.tta_crops:
                break
        if len(views) < self.tta_crops:
            while len(views) < self.tta_crops:
                views.append(image)
        if self.tta_flips:
            from PIL import ImageOps

            views.extend(ImageOps.mirror(view) for view in list(views))
        return views

    @staticmethod
    def _display_label(label: str) -> str:
        return label.replace("___", " - ").replace("_", " ").strip()

    # ---------------------------------------------------------------- inference
    def _predict(self, request: Request) -> Response:
        torch = self._torch()
        image = decode_request_image(request.image_base64, request.image_path)
        views = self.build_views(image)
        if not views:  # pragma: no cover - build_views always returns at least one
            raise DataValidationError("No augmented views could be built from the image")

        tensor_inputs = {key: value.to(self.device) for key, value in self.preprocessor.batch(views).items()}
        with torch.inference_mode():
            logits = self.model(**tensor_inputs).logits
            probabilities = torch.softmax(logits, dim=1).float().mean(dim=0).cpu().numpy()

        order = np.argsort(probabilities)[::-1]
        top_k = min(int(request.top_k), len(self.id2label))
        ranked = [(self.id2label[int(index)], float(probabilities[int(index)])) for index in order[:top_k]]
        value, top_probability = ranked[0]
        is_healthy = "healthy" in value.lower()
        crop = None
        if "___" in value:
            crop = value.split("___", 1)[0].replace("_", " ").strip() or None

        alternatives = (
            [{"class": name, "score": round(score, 6)} for name, score in ranked[1:]]
            if top_probability < float(self.config.serving.confidence_high)
            else None
        )
        reasons = [
            f"The pretrained model's highest averaged probability is '{self._display_label(value)}' "
            f"at {top_probability:.1%} over {len(views)} augmented views.",
        ]
        if len(ranked) > 1:
            runner_up, runner_up_score = ranked[1]
            reasons.append(
                f"Second place is '{self._display_label(runner_up)}' at {runner_up_score:.1%}; "
                "when two classes are close the image does not separate them."
            )
        if top_probability < self.low_confidence_threshold:
            reasons.append(
                f"Averaged confidence is below the configured {self.low_confidence_threshold:.0%} "
                "threshold, so treat this as a weak signal and prefer a human inspection."
            )
        reasons.append(NO_ACCURACY_CLAIM)

        return Response(
            value=value,
            display_label=self._display_label(value),
            class_probabilities={name: round(score, 6) for name, score in ranked},
            is_healthy=is_healthy,
            crop=crop,
            tta_views=len(views),
            weights_source=str(self.weights_source.get("model_id", "")),
            confidence_score=top_probability,
            confidence=Confidence.UNKNOWN,  # finalised from the configured thresholds
            alternatives=alternatives,
            reasons=reasons,
            limitations=[THIRD_PARTY_LIMITATION, NO_ACCURACY_CLAIM],
            model_name=self.module_name,
            model_version=self.version,
        )

    # ------------------------------------------------------------------- health
    def health(self) -> Any:
        """Readiness plus the reason, and the limitation once ready."""
        health = super().health()
        if health.ready:
            health.detail = THIRD_PARTY_LIMITATION
        return health


#: CLI sample. A soil texture is used deliberately: it is the only image bundled in
#: this checkout. The prediction is meaningless as plant disease evidence and is
#: printed only to prove the inference path runs end to end.
SAMPLE: dict[str, Any] = {
    "image_path": "data/raw/soil_images/loamy_soil/loamy_soil_0001.jpg",
    "top_k": 3,
}


if __name__ == "__main__":
    from agri.common.runner import run_predict

    sys.exit(run_predict("disease", Predictor, SAMPLE))
