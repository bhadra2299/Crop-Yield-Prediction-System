"""MobileNetV2 architecture, transforms and checkpoint plumbing.

torch and torchvision are an optional extra (the 'vision' extra in pyproject). The
default Docker image does not install them, so this package must import cleanly
without them: every torch import is guarded, and the functions that genuinely need
torch raise an AgriError with an actionable message instead of ImportError.

The architecture is fixed to torchvision's mobilenet_v2 with a replaced classifier
head. That keeps the artefact small (about 9 MB of state dict), runs in tens of
milliseconds on a CPU, and needs no custom layer code at serve time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agri.common.errors import ConfigError, ModuleExecutionError
from agri.common.logging import get_logger
from agri.common.paths import EXTERNAL_DIR, ensure_dir

_log = get_logger("modules.soil_image.model")

try:  # optional vision extra
    import torch
    from torch import nn
    from torchvision import transforms
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: str | None = None
except Exception as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
    MobileNet_V2_Weights = None  # type: ignore[assignment]
    mobilenet_v2 = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

#: ImageNet statistics. Frozen into the artefact manifest so serving cannot drift.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

ARCHITECTURE = "mobilenet_v2"

#: Recorded verbatim in the registry manifest. Training and serving share it.
#
#: Photometric jitter is deliberately small. In this corpus the class signal is
#: largely colour and grain, so a brightness or saturation jitter of 0.25 (a first
#: attempt) removes the very feature being learned: that run underfitted at
#: macro_f1 0.46 with train accuracy 0.66 while ROC-AUC already reached 0.93.
#: Geometry is jittered more freely because it does not change soil colour.
AUGMENTATION_POLICY: tuple[str, ...] = (
    "RandomResizedCrop(scale=(0.80, 1.00), ratio=(0.90, 1.10))",
    "RandomHorizontalFlip(p=0.5)",
    "RandomRotation(degrees=8)",
    "ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02)",
    "ToTensor",
    "Normalize(ImageNet mean/std)",
)


def ensure_torch() -> None:
    """Raise a clear AgriError when the vision extra is not installed.

    ConfigError is the right class: the deployment is missing a declared optional
    dependency, which is a configuration problem, not a bad request.
    """
    if not TORCH_AVAILABLE:
        raise ConfigError(
            "soil_image requires the vision extra (torch and torchvision); install it with "
            "pip install -e '.[vision]' or the vision-enabled Docker image",
            context={"module": "soil_image", "import_error": TORCH_IMPORT_ERROR},
        )


def build_transforms(
    input_size: tuple[int, int],
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    *,
    train: bool,
) -> Any:
    """Build the train or eval transform pipeline for the configured input size.

    Training augmentation is deliberately mild. The corpus is synthetic, so heavy
    augmentation would mostly teach the network to undo our own generator rather
    than to recognise texture; the policy is recorded in the manifest so a metric
    can always be read against the augmentation that produced it.

    The photometric terms are the small ones on purpose: colour is the dominant
    class signal here, so large brightness or saturation jitter removes the feature
    instead of regularising it.
    """
    ensure_torch()
    height, width = int(input_size[0]), int(input_size[1])
    resize_side = round(256 * max(height, width) / 224)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop((height, width), scale=(0.80, 1.00), ratio=(0.90, 1.10)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(resize_side),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def torch_home() -> Path:
    """Resolve a writable torch hub cache directory.

    torch defaults to ~/.cache/torch. That path is not writable on several locked
    down Windows and container setups, and the resulting PermissionError is easily
    misread as "the download host is unreachable". An explicit TORCH_HOME always
    wins; otherwise the project cache under data/external/cache is used, which is
    the directory the repository already ignores, so a checkout behaves the same on
    any host and the weights land somewhere reproducible and out of git.
    """
    configured = os.getenv("TORCH_HOME")
    if configured:
        return ensure_dir(Path(configured).expanduser())
    return ensure_dir(EXTERNAL_DIR / "cache" / "torch")


def build_model(
    num_classes: int,
    *,
    pretrained: bool = False,
    dropout: float = 0.2,
    device: str = "cpu",
) -> Any:
    """Build mobilenet_v2 with a fresh classifier head.

    When pretrained=True the ImageNet weights are fetched from the torchvision
    download host. Offline, that raises; the fallback is random initialisation with
    a warning, never a crash, so the training pipeline still runs (slower to
    converge and with a lower score, which the manifest records via 'pretrained').
    """
    ensure_torch()
    if num_classes < 2:
        raise ConfigError("A classifier needs at least two classes", context={"num_classes": num_classes})

    model = None
    if pretrained:
        try:
            os.environ["TORCH_HOME"] = str(torch_home())
            model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            _log.info("loaded ImageNet pretrained mobilenet_v2 weights")
        except Exception as exc:
            _log.warning(
                "pretrained weights unavailable; falling back to random initialisation",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            model = None
    if model is None:
        model = mobilenet_v2(weights=None)

    in_features = int(model.classifier[1].in_features)
    model.classifier = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_features, num_classes))
    return model.to(device)


def set_backbone_trainable(model: Any, trainable: bool) -> None:
    """Freeze or unfreeze the convolutional trunk, leaving the head trainable.

    Freezing for the first epochs is what makes CPU training of a ten-class
    problem finish in minutes instead of hours.
    """
    ensure_torch()
    for parameter in model.features.parameters():
        parameter.requires_grad = bool(trainable)


def load_state_dict(model: Any, state_dict: dict[str, Any]) -> Any:
    """Load a saved state dict strictly and report a mismatch as a clear error."""
    ensure_torch()
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ModuleExecutionError(
            "Saved soil_image weights do not match the mobilenet_v2 architecture",
            context={"module": "soil_image", "error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    return model
