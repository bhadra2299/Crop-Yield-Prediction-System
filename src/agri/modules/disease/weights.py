"""Resolve, download, cache and apply the third-party PlantVillage preprocessing.

The model id comes from configs/disease.yaml (model.params.model_id); the cache
directory comes from the same file (model.params.cache_dir). Nothing here hardcodes
either.

Why the preprocessing is implemented here
-----------------------------------------
The upstream checkpoint ships a preprocessor_config.json whose type is
MobileNetV2FeatureExtractor. That class was removed from recent transformers
releases, so AutoImageProcessor no longer resolves it and inference would fail on a
perfectly valid model. Rather than pin an old transformers, the spec is read from
the same JSON (shortest edge, crop size, rescale, mean, std) and applied with PIL
and torch here. The numbers are the model author's; only the plumbing is ours.

Offline behaviour is deliberate and layered:

  1. try the cache only (local_files_only=True). If the weights were downloaded
     once, the service starts with no network at all.
  2. unless allow_download is false or HF_HUB_OFFLINE is set, retry with the Hub.
  3. if every attempt fails, raise ExternalServiceError. That class - not a bare
     Exception and not ImportError - is what lets the orchestrator degrade this one
     signal and still answer the farmer.

The weights are third-party. They were trained on PlantVillage and are NOT
validated on regional varieties; see predict.py for the limitation attached to
every response.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agri.common.errors import ConfigError, ExternalServiceError
from agri.common.logging import get_logger
from agri.common.paths import PROJECT_ROOT, ensure_dir, read_json

_log = get_logger("modules.disease.weights")

#: Packages needed for inference. transformers is used only for the model class.
REQUIRED_PACKAGES: tuple[str, ...] = ("torch", "transformers")

#: Preprocessing defaults from MobileNetV2's published ImageNet configuration. Used
#: only when the checkpoint ships no preprocessor_config.json; every value is
#: recorded in the weights source so a mismatch is visible.
DEFAULT_PREPROCESSING: dict[str, Any] = {
    "shortest_edge": 256,
    "crop_size": [224, 224],
    "rescale_factor": 1.0 / 255.0,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "resample": "bilinear",
}


def _module_available(name: str) -> bool:
    """Cheap availability probe that does not import the package."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def missing_packages() -> list[str]:
    """Names of required inference packages that are not installed."""
    return [name for name in REQUIRED_PACKAGES if not _module_available(name)]


def require_vision_stack() -> None:
    """Raise ConfigError when the optional vision extra is absent.

    ConfigError, not ExternalServiceError: nothing upstream failed, the deployment
    is missing a declared optional dependency.
    """
    missing = missing_packages()
    if missing:
        raise ConfigError(
            "disease requires the vision extra (torch and transformers); missing: "
            + ", ".join(missing)
            + ". Install with pip install -e '.[vision]' plus transformers, or use the "
            "vision-enabled image container.",
            context={"module": "disease", "missing": missing},
        )


def resolve_cache_dir(configured: str | os.PathLike[str] | None) -> Path:
    """Resolve the weight cache directory from config, relative to the project root."""
    if not configured:
        raise ConfigError(
            "disease config must set model.params.cache_dir so weights are cached somewhere explicit",
            context={"module": "disease"},
        )
    candidate = Path(configured).expanduser()
    resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return ensure_dir(resolved)


def _offline_requested() -> bool:
    """Honour the Hugging Face offline convention."""
    return os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def _download_metadata_file(filename: str, model_id: str, cache_dir: Path, *, local_files_only: bool) -> Path:
    """Fetch one small metadata file (not the weights) through the Hub cache."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
    )


class ImagePreprocessor:
    """Resize, centre-crop, rescale and normalise exactly as the checkpoint expects.

    Deliberately small and explicit: a processor object that hides its tensor
    contract is how train/serve preprocessing drift starts.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.shortest_edge = int(spec.get("shortest_edge", 256))
        crop = spec.get("crop_size") or [224, 224]
        self.crop_height = int(crop[0])
        self.crop_width = int(crop[1])
        self.rescale_factor = float(spec.get("rescale_factor", 1.0 / 255.0))
        self.mean = tuple(float(value) for value in spec.get("image_mean", (0.5, 0.5, 0.5)))
        self.std = tuple(float(value) for value in spec.get("image_std", (0.5, 0.5, 0.5)))
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ExternalServiceError(
                "preprocessor_config.json must provide three mean and three std values",
                context={"module": "disease", "mean": list(self.mean), "std": list(self.std)},
            )
        if any(value == 0.0 for value in self.std):
            raise ExternalServiceError(
                "preprocessor_config.json declares a zero normalisation std",
                context={"module": "disease", "std": list(self.std)},
            )
        self.spec = {
            "shortest_edge": self.shortest_edge,
            "crop_size": [self.crop_height, self.crop_width],
            "rescale_factor": self.rescale_factor,
            "image_mean": list(self.mean),
            "image_std": list(self.std),
        }

    def _resize_shortest(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= height:
            new_width = self.shortest_edge
            new_height = round(height * self.shortest_edge / width)
        else:
            new_height = self.shortest_edge
            new_width = round(width * self.shortest_edge / height)
        return image.resize((max(1, new_width), max(1, new_height)), Image.BILINEAR)

    def _centre_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        left = max(0, (width - self.crop_width) // 2)
        top = max(0, (height - self.crop_height) // 2)
        cropped = image.crop(
            (left, top, left + min(width, self.crop_width), top + min(height, self.crop_height))
        )
        if cropped.size != (self.crop_width, self.crop_height):
            # Upstream pads a too-small image with zeros rather than upscaling.
            canvas = Image.new("RGB", (self.crop_width, self.crop_height), (0, 0, 0))
            canvas.paste(
                cropped, ((self.crop_width - cropped.size[0]) // 2, (self.crop_height - cropped.size[1]) // 2)
            )
            return canvas
        return cropped

    def preprocess(self, image: Image.Image) -> Any:
        """Return one CHW float tensor for a single RGB PIL image."""
        import torch

        rgb = image.convert("RGB")
        resized = self._resize_shortest(rgb)
        cropped = self._centre_crop(resized)
        array = np.asarray(cropped, dtype=np.float32) * self.rescale_factor
        array = (array - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def batch(self, images: list[Image.Image]) -> dict[str, Any]:
        """Stack several images into the model input mapping."""
        import torch

        return {"pixel_values": torch.stack([self.preprocess(image) for image in images])}


def _preprocessing_spec(
    model_id: str, cache_dir: Path, *, local_files_only: bool
) -> tuple[dict[str, Any], str]:
    """Read preprocessor_config.json from the cache or Hub, mapping it to our spec."""
    try:
        path = _download_metadata_file(
            "preprocessor_config.json", model_id, cache_dir, local_files_only=local_files_only
        )
    except Exception as exc:
        _log.warning(
            "preprocessor_config.json unavailable; using documented MobileNetV2 defaults",
            extra={"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"},
        )
        return dict(DEFAULT_PREPROCESSING), "defaults"

    try:
        raw = read_json(path)
    except Exception as exc:
        raise ExternalServiceError(
            "preprocessor_config.json is present but unreadable; refusing to guess preprocessing",
            context={"module": "disease", "path": str(path), "error": str(exc)},
        ) from exc

    crop = raw.get("crop_size") or {}
    size = raw.get("size") or {}
    spec = {
        "shortest_edge": int(size.get("shortest_edge", DEFAULT_PREPROCESSING["shortest_edge"])),
        "crop_size": [
            int(crop.get("height", DEFAULT_PREPROCESSING["crop_size"][0])),
            int(crop.get("width", DEFAULT_PREPROCESSING["crop_size"][1])),
        ],
        "rescale_factor": float(raw.get("rescale_factor", DEFAULT_PREPROCESSING["rescale_factor"])),
        "image_mean": list(raw.get("image_mean", DEFAULT_PREPROCESSING["image_mean"])),
        "image_std": list(raw.get("image_std", DEFAULT_PREPROCESSING["image_std"])),
    }
    return spec, str(path)


def id2label_from(model: Any) -> dict[int, str]:
    """Read the label map from the model config, refusing a model without one."""
    raw = getattr(getattr(model, "config", None), "id2label", None) or {}
    try:
        mapping = {int(key): str(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ExternalServiceError(
            "The PlantVillage model exposes an unusable id2label map",
            context={"module": "disease", "error": str(exc)},
        ) from exc
    if len(mapping) < 2:
        raise ExternalServiceError(
            "The PlantVillage model exposes no class labels; predictions would be unnameable",
            context={"module": "disease", "labels": len(mapping)},
        )
    return dict(sorted(mapping.items()))


def load_pretrained(
    model_id: str,
    cache_dir: Path,
    *,
    allow_download: bool = True,
    device: str = "cpu",
) -> tuple[Any, ImagePreprocessor, dict[int, str], dict[str, Any]]:
    """Load the cached (or downloaded) model plus its preprocessing and labels.

    Returns (model, preprocessor, id2label, source) where source records how the
    weights were obtained, so /v1/health can say whether this process runs from
    cache or from a fresh download.
    """
    require_vision_stack()
    from transformers import AutoModelForImageClassification

    offline = _offline_requested() or not allow_download
    modes = [True] if offline else [True, False]
    failures: list[str] = []
    for local_files_only in modes:
        label = "cache-only" if local_files_only else "hub-download"
        try:
            _log.info(
                "loading third-party disease weights",
                extra={"model_id": model_id, "mode": label, "cache_dir": str(cache_dir)},
            )
            model = AutoModelForImageClassification.from_pretrained(
                model_id,
                cache_dir=str(cache_dir),
                local_files_only=local_files_only,
            )
            spec, spec_source = _preprocessing_spec(model_id, cache_dir, local_files_only=local_files_only)
            preprocessor = ImagePreprocessor(spec)
            id2label = id2label_from(model)
            model = model.to(device)
            model.eval()
            source = {
                "model_id": model_id,
                "cache_dir": str(cache_dir),
                "loaded_from_cache_only": local_files_only,
                "attempt": label,
                "num_labels": len(id2label),
                "preprocessing_source": spec_source,
                "preprocessing": preprocessor.spec,
            }
            _log.info("disease weights ready", extra=source)
            return model, preprocessor, id2label, source
        except ExternalServiceError:
            raise
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            _log.warning(
                "disease weight load attempt failed",
                extra={"model_id": model_id, "mode": label, "error": f"{type(exc).__name__}: {exc}"},
            )

    raise ExternalServiceError(
        "Third-party disease weights are unavailable: not in the local cache and the "
        "Hugging Face Hub could not be reached (offline, gated or rate-limited)",
        context={
            "module": "disease",
            "model_id": model_id,
            "cache_dir": str(cache_dir),
            "allow_download": bool(allow_download),
            "offline_env": _offline_requested(),
            "attempts": failures,
            "hint": (
                "Pre-populate the cache directory on a connected host, or unset HF_HUB_OFFLINE / "
                "set model.params.allow_download true. The rest of the platform keeps working; "
                "this single signal is degraded."
            ),
        },
    )
