"""Training and offline evaluation for the soil image classifier.

Both entry points live here rather than in train.py / evaluate.py so the CLI files
stay thin, and so training and evaluation cannot drift apart.

Why this does not go through agri.common.training.train_module
--------------------------------------------------------------
train_module is the tabular pipeline: it calls read_frame, which raises
ModuleExecutionError for a directory ("reads a directory, not a CSV; use its own
loader"). That error is the framework telling a non-tabular module to bring its own
loader, so this file is that loader plus the same discipline the tabular path
enforces: seed everything, split deterministically, never touch the test partition
while fitting, compare against a naive baseline, enforce the configured gate, write
the metrics report, and register the artefact with a reproducible manifest.

Honesty contract
----------------
The corpus is PROCEDURALLY GENERATED. Every manifest and report written here names
that, carries synthetic_data=true, and repeats the limitation string from
provenance.py. The gate (macro_f1 >= 0.85) therefore measures separability of
synthetic textures; it is not, and must never be presented as, field accuracy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agri.common.config import ModuleConfig, load_module_config
from agri.common.errors import AgriError, ConfigError, ModuleExecutionError
from agri.common.logging import get_logger
from agri.common.metrics import (
    baseline_predictions,
    compute_metrics,
    confusion,
    default_baseline,
    evaluate_gate,
    per_class_report,
)
from agri.common.paths import FIGURES_DIR, git_sha
from agri.common.registry import ModelRegistry, RegistryEntry
from agri.common.schemas import BaselineKind
from agri.common.seed import set_global_seed
from agri.common.tabular import plot_confusion, write_metrics_report

from . import provenance
from .dataset import (
    DatasetIndex,
    SoilImageDataset,
    SplitIndex,
    dataset_digest,
    index_dataset,
    stratified_split,
)
from .model import (
    ARCHITECTURE,
    AUGMENTATION_POLICY,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    build_transforms,
    ensure_torch,
    load_state_dict,
    set_backbone_trainable,
)

MODULE_NAME = "soil_image"
ARTIFACT_FILENAME = "model.pt"

_log = get_logger("modules.soil_image.training")


@dataclass
class SoilImageArtifact:
    """Everything needed to reproduce a prediction, in one serialisable unit.

    The state dict alone is not enough: the class order, input geometry and
    normalisation constants must travel with it, or serving can silently disagree
    with training. This object is what the registry serialises into model.pt; the
    JSON manifest beside it carries the human-readable sidecar fields.

    Note on the file extension: the registry serialises every artefact with joblib
    (see agri.common.registry.ModelRegistry.save), so model.pt is a joblib file
    whose payload is this dataclass holding the torch state dict. Load it with
    entry.load(), not torch.load.
    """

    state_dict: dict[str, Any]
    classes: list[str]
    image_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    architecture: str = ARCHITECTURE
    dropout: float = 0.2
    synthetic_data: bool = provenance.SYNTHETIC_DATA
    data_provenance: str = provenance.DATA_PROVENANCE
    limitation: str = provenance.LIMITATION
    augmentation: tuple[str, ...] = AUGMENTATION_POLICY
    trained_metrics: dict[str, float] = field(default_factory=dict)
    per_class_f1: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    epochs: int = 0
    pretrained: bool = False


# --------------------------------------------------------------------- helpers
def _resolve_params(
    config: ModuleConfig,
    *,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    pretrained: bool | None,
    freeze_epochs: int | None = None,
    num_workers: int | None = None,
    image_size: tuple[int, int] | None = None,
    linear_probe: bool = False,
) -> dict[str, Any]:
    """Merge CLI overrides over configs/soil_image.yaml.

    Every default comes from the YAML; nothing here invents a hyperparameter. The
    resolved values are recorded in the manifest, so a hand-run override can never
    masquerade as the configured baseline.
    """
    params = dict(config.model.params)
    resolved = {
        "epochs": int(epochs if epochs is not None else params.get("epochs", 6)),
        "batch_size": int(batch_size if batch_size is not None else params.get("batch_size", 32)),
        "learning_rate": float(
            learning_rate if learning_rate is not None else params.get("learning_rate", 0.001)
        ),
        "weight_decay": float(params.get("weight_decay", 0.0001)),
        "freeze_backbone_epochs": int(
            freeze_epochs if freeze_epochs is not None else params.get("freeze_backbone_epochs", 0)
        ),
        "dropout": float(params.get("dropout", 0.2)),
        "num_workers": int(num_workers if num_workers is not None else params.get("num_workers", 0)),
        "device": str(params.get("device", "cpu")),
        "pretrained": bool(pretrained if pretrained is not None else params.get("pretrained", True)),
        "num_threads": params.get("num_threads"),
        "linear_probe": bool(linear_probe),
    }
    if resolved["epochs"] < 1:
        raise ConfigError("epochs must be at least 1", context={"epochs": resolved["epochs"]})
    if resolved["batch_size"] < 1:
        raise ConfigError("batch_size must be at least 1", context={"batch_size": resolved["batch_size"]})
    if resolved["learning_rate"] <= 0:
        raise ConfigError(
            "learning_rate must be positive", context={"learning_rate": resolved["learning_rate"]}
        )
    configured_size = image_size or config.data.image_size or (224, 224)
    resolved["input_size"] = (int(configured_size[0]), int(configured_size[1]))
    if resolved["num_workers"] < 0:
        raise ConfigError("num_workers cannot be negative", context={"num_workers": resolved["num_workers"]})
    return resolved


def _data_loader(
    records: list[Any], transform: Any, *, batch_size: int, shuffle: bool, num_workers: int, device: str
) -> Any:
    from torch.utils.data import DataLoader

    dataset = SoilImageDataset(records, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )


def _frozen_features(model: Any, loader: Any, device: str) -> tuple[Any, Any]:
    """One inference-only backbone pass per image; returns CPU features and labels.

    Features are pooled exactly the way torchvision's classifier sees them
    (global average pool over the last feature map), so a head trained on these
    tensors is identical to the head the assembled model would train.
    """
    import torch

    model.eval()
    features: list[Any] = []
    targets: list[Any] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            pooled = torch.nn.functional.adaptive_avg_pool2d(model.features(inputs), 1).flatten(1)
            features.append(pooled.detach().cpu())
            targets.append(labels.detach().cpu())
    return torch.cat(features), torch.cat(targets)


def _fit_linear_head(
    features: Any,
    targets: Any,
    *,
    num_classes: int,
    params: dict[str, Any],
    device: str,
    seed: int,
) -> tuple[Any, list[dict[str, float]]]:
    """Train the classifier head on frozen features.

    The backbone is never updated here; every epoch is a full pass over cached
    features, which is why this converges in seconds rather than hours on CPU.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    head = nn.Sequential(nn.Dropout(params["dropout"]), nn.Linear(int(features.shape[1]), num_classes)).to(
        device
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    features = features.to(device)
    targets = targets.to(device)
    history: list[dict[str, float]] = []
    for epoch in range(1, params["epochs"] + 1):
        head.train()
        started = time.perf_counter()
        order = torch.randperm(features.shape[0], device=device)
        running_loss = 0.0
        correct = 0
        seen = 0
        for start in range(0, features.shape[0], params["batch_size"]):
            index = order[start : start + params["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = head(features[index])
            loss = criterion(logits, targets[index])
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * int(index.numel())
            correct += int((logits.argmax(dim=1) == targets[index]).sum())
            seen += int(index.numel())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(1, seen),
                "train_accuracy": correct / max(1, seen),
                "seconds": round(time.perf_counter() - started, 1),
                "backbone_frozen": 1.0,
            }
        )
    head.eval()
    return head, history


def _as_probabilities(head: Any, features: Any, device: str) -> Any:
    """Softmax the head over cached features, returning a numpy array."""
    import torch

    head.eval()
    with torch.inference_mode():
        return torch.softmax(head(features.to(device)), dim=1).detach().cpu().numpy()


def _train_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any, device: str) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * int(targets.size(0))
        correct += int((logits.argmax(dim=1) == targets).sum())
        seen += int(targets.size(0))
    return total_loss / max(seen, 1), correct / max(seen, 1)


def _score_loader(model: Any, loader: Any, criterion: Any, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, true indices) over a loader, without gradients."""
    import torch

    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            probabilities.append(torch.softmax(logits, dim=1).float().cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(probabilities, axis=0), np.concatenate(labels, axis=0)


def _clean_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Drop NaN and infinite values before they reach a JSON manifest."""
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            clean[key] = numeric
    return clean


def _evaluate_predictions(
    config: ModuleConfig,
    classes: list[str],
    probabilities: np.ndarray,
    y_true_index: np.ndarray,
    y_train_index: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any], dict[str, float]]:
    """Compute test metrics, report detail and per-class F1."""
    predicted_index = probabilities.argmax(axis=1)
    y_true = [classes[int(index)] for index in y_true_index]
    y_pred = [classes[int(index)] for index in predicted_index]
    metrics = _clean_metrics(compute_metrics(y_true, y_pred, task=config.task, y_score=probabilities))

    primary = config.gates.primary_metric
    kind = BaselineKind(config.gates.baseline) if config.gates.baseline else default_baseline(config.task)
    train_labels = np.asarray([classes[int(index)] for index in y_train_index])
    baseline_pred = baseline_predictions(kind, train_labels, n_test=len(y_true))
    baseline_metrics = _clean_metrics(compute_metrics(y_true, baseline_pred, task=config.task))
    if primary in baseline_metrics:
        metrics["baseline_score"] = float(baseline_metrics[primary])
        denominator = max(abs(baseline_metrics[primary]), 1e-12)
        metrics["relative_lift_over_baseline"] = float(
            (metrics[primary] - baseline_metrics[primary]) / denominator
        )
        metrics["lift_over_baseline"] = float(metrics[primary] - baseline_metrics[primary])

    detail: dict[str, Any] = {
        "confusion_matrix": confusion(y_true, y_pred, labels=classes),
        "per_class": per_class_report(y_true, y_pred),
        "n_test": len(y_true),
    }
    per_class_f1 = {
        str(label): float(stats.get("f1-score", 0.0))
        for label, stats in detail["per_class"].items()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    }
    detail["per_class_f1"] = per_class_f1
    detail["baseline_metrics"] = baseline_metrics
    return metrics, detail, per_class_f1


def _write_report(
    config: ModuleConfig,
    metrics: dict[str, float],
    detail: dict[str, Any],
    gate: Any,
    *,
    upstream_manifest: dict[str, Any] | None,
    training: dict[str, Any],
    split: SplitIndex,
    digest: str,
) -> None:
    """Persist the metrics report. Provenance is written, not implied."""
    write_metrics_report(
        config,
        metrics,
        detail,
        gate.to_dict(),
        manifest={
            "git_sha": git_sha(),
            "data_digest": digest,
            **provenance.limitation_bundle(),
            **(upstream_manifest or {}),
        },
        extra={
            **provenance.limitation_bundle(),
            "per_class_f1": detail.get("per_class_f1", {}),
            "training": training,
            "split_counts": {"total": split.counts(), "per_class": split.per_class_counts()},
            "class_names": config.data.classes,
        },
    )


def _load_index(config: ModuleConfig) -> tuple[DatasetIndex, str]:
    index = index_dataset(config.data.raw_path, config.data.classes)
    digest = dataset_digest(index)
    _log.info(
        "corpus indexed",
        extra={"classes": len(index.classes), "images": len(index.all_records()), "digest": digest[:16]},
    )
    return index, digest


# -------------------------------------------------------------------- training
def train_soil_image(
    *,
    epochs: int | None = None,
    max_per_class: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    pretrained: bool | None = None,
    freeze_epochs: int | None = None,
    num_workers: int | None = None,
    image_size: tuple[int, int] | None = None,
    linear_probe: bool = False,
    register: bool = True,
    version: str | None = None,
) -> dict[str, Any]:
    """Fine-tune mobilenet_v2 on the synthetic soil corpus and register the result.

    Parameters are optional overrides. Passing None means "use the config", which is
    the behaviour the CLI relies on for its defaults.

    Raises AgriError when the configured metric gate fails, which is what gives the
    CLI a usable non-zero exit code in CI.
    """
    ensure_torch()
    import torch
    from torch import nn

    config = load_module_config(MODULE_NAME)
    classes = list(config.data.classes)
    if not classes:
        raise ConfigError("soil_image declares no classes", context={"config": config.source_path})
    params = _resolve_params(
        config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pretrained=pretrained,
        freeze_epochs=freeze_epochs,
        num_workers=num_workers,
        image_size=image_size,
        linear_probe=linear_probe,
    )
    device = params["device"]
    if params["num_threads"]:
        torch.set_num_threads(int(params["num_threads"]))

    set_global_seed(config.seed)
    index, digest = _load_index(config)
    split = stratified_split(
        index.records,
        seed=config.seed,
        val_size=float(config.split.val_size),
        test_size=float(config.split.test_size),
        class_order=classes,
        max_train_per_class=max_per_class,
    )
    if max_per_class is not None:
        _log.warning(
            "training on a capped subset; the test partition is unchanged",
            extra={"max_per_class": max_per_class, "n_train": len(split.train)},
        )

    train_transform = build_transforms(params["input_size"], train=True)
    eval_transform = build_transforms(params["input_size"], train=False)
    if linear_probe:
        # Features are extracted once and reused for every epoch, so a random per-image
        # augmentation would be baked in as a single fixed view. Use the deterministic
        # eval transform for the probe and say so in the manifest.
        train_transform = eval_transform
    train_loader = _data_loader(
        split.train,
        train_transform,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        device=device,
    )
    val_loader = _data_loader(
        split.validation,
        eval_transform,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        device=device,
    )
    test_loader = _data_loader(
        split.test,
        eval_transform,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        device=device,
    )

    model = build_model(
        len(classes),
        pretrained=params["pretrained"],
        dropout=params["dropout"],
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    freeze_epochs = params["freeze_backbone_epochs"] if params["pretrained"] else 0
    history: list[dict[str, float]] = []

    if linear_probe:
        # Frozen backbone plus a trained linear head. Same transfer-learning recipe as a
        # fully frozen epoch, except the backbone runs once per image instead of once per
        # epoch, so a CPU-only run converges in minutes rather than hours. The artefact is
        # unchanged in kind: the state dict still holds the ImageNet backbone and the head
        # that was actually trained.
        started = time.perf_counter()
        train_features, train_targets = _frozen_features(model, train_loader, device)
        val_features, val_targets = _frozen_features(model, val_loader, device)
        test_features, test_targets = _frozen_features(model, test_loader, device)
        _log.info(
            "backbone features extracted",
            extra={
                "module": config.name,
                "seconds": round(time.perf_counter() - started, 1),
                "n_train": int(train_features.shape[0]),
            },
        )
        head, history = _fit_linear_head(
            train_features,
            train_targets,
            num_classes=len(classes),
            params=params,
            device=device,
            seed=config.seed,
        )
        model.classifier = head
        probabilities = _as_probabilities(head, test_features, device)
        y_true_index = test_targets.numpy()
        val_probabilities = _as_probabilities(head, val_features, device)
        val_true = val_targets.numpy()
        val_accuracy = float((val_probabilities.argmax(axis=1) == val_true).mean())
        val_f1 = float(
            compute_metrics(
                [classes[int(i)] for i in val_true],
                [classes[int(i)] for i in val_probabilities.argmax(axis=1)],
                task=config.task,
            )["macro_f1"]
        )
        history[-1].update({"val_accuracy": val_accuracy, "val_macro_f1": val_f1})
        _log.info("linear probe complete", extra={"module": config.name, **history[-1]})
    else:
        for epoch in range(1, params["epochs"] + 1):
            frozen = epoch <= freeze_epochs
            set_backbone_trainable(model, not frozen)
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )
            started = time.perf_counter()
            train_loss, train_accuracy = _train_epoch(model, train_loader, criterion, optimizer, device)
            elapsed = time.perf_counter() - started
            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "seconds": round(elapsed, 1),
                "backbone_frozen": float(frozen),
            }
            # Validation is monitoring only; scoring it every epoch would add minutes of
            # CPU time for a number nothing is tuned on. It is measured once, at the end.
            if val_loader is not None and len(split.validation) and epoch == params["epochs"]:
                val_probs, val_true = _score_loader(model, val_loader, criterion, device)
                val_accuracy = float((val_probs.argmax(axis=1) == val_true).mean())
                val_f1 = float(
                    compute_metrics(
                        [classes[int(i)] for i in val_true],
                        [classes[int(i)] for i in val_probs.argmax(axis=1)],
                        task=config.task,
                    )["macro_f1"]
                )
                row.update({"val_accuracy": val_accuracy, "val_macro_f1": val_f1})
            history.append(row)
            _log.info("epoch complete", extra={"module": config.name, **row})

        probabilities, y_true_index = _score_loader(model, test_loader, criterion, device)
    y_train_index = np.asarray([record.label_index for record in split.train], dtype=np.int64)
    metrics, detail, per_class_f1 = _evaluate_predictions(
        config, classes, probabilities, y_true_index, y_train_index
    )
    gate = evaluate_gate(metrics, config.gates)
    _log.info("gate evaluated", extra={"module": config.name, **gate.to_dict()})

    resolved_version = str(version or config.version)
    artifact = SoilImageArtifact(
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        classes=classes,
        image_size=params["input_size"],
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        architecture=ARCHITECTURE,
        dropout=params["dropout"],
        trained_metrics=metrics,
        per_class_f1=per_class_f1,
        seed=config.seed,
        epochs=params["epochs"],
        pretrained=bool(params["pretrained"]),
    )
    training_block = {
        **{key: value for key, value in params.items() if key != "num_threads"},
        "max_train_per_class": max_per_class,
        "linear_probe": bool(linear_probe),
        "backbone_frozen_all_epochs": bool(linear_probe),
        "augmentation": (
            ["none: frozen features are extracted once with the deterministic eval transform"]
            if linear_probe
            else list(AUGMENTATION_POLICY)
        ),
        "history": history,
        "split": {
            "seed": config.seed,
            "val_size": float(config.split.val_size),
            "test_size": float(config.split.test_size),
            "max_train_per_class": max_per_class,
        },
        "n_train": len(split.train),
        "n_validation": len(split.validation),
        "n_test": len(split.test),
        "device": device,
    }

    entry: RegistryEntry | None = None
    if register and not gate.passed:
        # Never publish an artefact that misses its own configured floor. The report is
        # written first so the failing number stays auditable, but the registry must not
        # end up holding a model its own gate rejects - that is exactly the state a
        # regression run exists to catch, and LATEST would otherwise serve it.
        _write_report(
            config,
            metrics,
            detail,
            gate,
            upstream_manifest={"version": resolved_version, "registered": False},
            training=training_block,
            split=split,
            digest=digest,
        )
        plot_confusion(detail, config.name, FIGURES_DIR / f"{config.name}_confusion.png")
        raise AgriError(
            f"Metric gate failed for {config.name}: {', '.join(gate.failed_checks)}; refusing to register",
            context={
                "primary_metric": gate.primary_metric,
                "value": gate.primary_value,
                "messages": gate.messages,
                "registered": False,
                "note": "The gate measures separability of synthetic textures only.",
            },
        )

    if register:
        registry = ModelRegistry()
        entry = registry.save(
            config.name,
            resolved_version,
            artifact,
            metrics={key: float(value) for key, value in metrics.items()},
            config_digest=config.config_digest,
            data_digest=digest,
            dataset=f"{config.data.raw_path.name} (procedurally generated synthetic textures)",
            params={key: value for key, value in params.items() if key != "num_threads"},
            artifact_file=ARTIFACT_FILENAME,
            extra={
                **provenance.limitation_bundle(),
                "gate": gate.to_dict(),
                "classes": classes,
                "class_names": classes,
                "image_size": list(params["input_size"]),
                "normalisation": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
                "architecture": ARCHITECTURE,
                "pretrained": bool(params["pretrained"]),
                "linear_probe": bool(linear_probe),
                "backbone_frozen_all_epochs": bool(linear_probe),
                "augmentation": (
                    ["none: frozen features are extracted once with the deterministic eval transform"]
                    if linear_probe
                    else list(AUGMENTATION_POLICY)
                ),
                "seed": config.seed,
                "epochs": params["epochs"],
                "batch_size": params["batch_size"],
                "learning_rate": params["learning_rate"],
                "weight_decay": params["weight_decay"],
                "freeze_backbone_epochs": freeze_epochs,
                "max_train_per_class": max_per_class,
                "per_class_f1": per_class_f1,
                "split": training_block["split"],
                "n_train": len(split.train),
                "n_validation": len(split.validation),
                "n_test": len(split.test),
                "device": device,
                "history": history,
            },
        )

    _write_report(
        config,
        metrics,
        detail,
        gate,
        upstream_manifest={
            "version": resolved_version,
            "artifact_file": ARTIFACT_FILENAME,
            "artifact_sha256": entry.manifest.get("artifact_sha256") if entry else None,
        },
        training=training_block,
        split=split,
        digest=digest,
    )
    plot_confusion(detail, config.name, FIGURES_DIR / f"{config.name}_confusion.png")

    if not gate.passed:
        raise AgriError(
            f"Metric gate failed for {config.name}: {', '.join(gate.failed_checks)}",
            context={
                "primary_metric": gate.primary_metric,
                "value": gate.primary_value,
                "messages": gate.messages,
                "registered": False,
                "note": "The gate measures separability of synthetic textures only.",
            },
        )

    return {
        "module": config.name,
        "version": resolved_version,
        "registered": bool(register),
        "synthetic_data": provenance.SYNTHETIC_DATA,
        "limitation": provenance.LIMITATION,
        "metrics": metrics,
        "gate": gate.to_dict(),
        "training": {
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
            "pretrained": bool(params["pretrained"]),
            "n_train": len(split.train),
            "n_validation": len(split.validation),
            "n_test": len(split.test),
            "max_train_per_class": max_per_class,
            "device": device,
            "history": history,
        },
    }


# ------------------------------------------------------------------ evaluation
def evaluate_soil_image(*, version: str | None = None) -> dict[str, Any]:
    """Re-evaluate a registered artefact against a freshly rebuilt test partition.

    Nothing is fitted here. The split is rebuilt from the split parameters recorded
    in the manifest, so the test partition is the same one the training run held
    out. If the corpus on disk no longer matches the recorded digest, evaluation
    refuses to run: a metric attached to a stale model is worse than no metric.
    """
    ensure_torch()
    config = load_module_config(MODULE_NAME)
    registry = ModelRegistry()
    entry = registry.get(MODULE_NAME, version)
    artifact = entry.load()
    if not isinstance(artifact, SoilImageArtifact):
        raise ModuleExecutionError(
            f"{MODULE_NAME} artefact is {type(artifact).__name__}, expected SoilImageArtifact",
            context={"version": entry.version},
        )
    classes = list(artifact.classes)
    if list(config.data.classes) and classes != list(config.data.classes):
        raise ConfigError(
            "Class list in the artefact does not match the current config; the artefact is stale",
            context={"artifact_classes": classes, "config_classes": list(config.data.classes)},
        )

    index, digest = _load_index(config)
    recorded_digest = entry.manifest.get("data_digest")
    if recorded_digest and recorded_digest != digest:
        raise AgriError(
            f"Corpus digest changed since {MODULE_NAME} v{entry.version} was trained; refusing to evaluate",
            context={"recorded": recorded_digest, "current": digest},
        )

    split_params = entry.manifest.get("split") or {}
    seed = int(split_params.get("seed", config.seed))
    val_size = float(split_params.get("val_size", config.split.val_size))
    test_size = float(split_params.get("test_size", config.split.test_size))
    max_train_per_class = split_params.get("max_train_per_class", entry.manifest.get("max_train_per_class"))
    split = stratified_split(
        index.records,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        class_order=classes,
        max_train_per_class=None if max_train_per_class is None else int(max_train_per_class),
    )

    device = str(entry.manifest.get("device", "cpu"))
    model = build_model(len(classes), pretrained=False, dropout=artifact.dropout, device=device)
    load_state_dict(model, artifact.state_dict)
    model.eval()

    eval_transform = build_transforms(
        tuple(artifact.image_size), tuple(artifact.mean), tuple(artifact.std), train=False
    )
    test_loader = _data_loader(
        split.test,
        eval_transform,
        batch_size=int(entry.manifest.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
        device=device,
    )
    import torch

    criterion = torch.nn.CrossEntropyLoss()
    probabilities, y_true_index = _score_loader(model, test_loader, criterion, device)
    y_train_index = np.asarray([record.label_index for record in split.train], dtype=np.int64)
    metrics, detail, per_class_f1 = _evaluate_predictions(
        config, classes, probabilities, y_true_index, y_train_index
    )
    gate = evaluate_gate(metrics, config.gates)
    _write_report(
        config,
        metrics,
        detail,
        gate,
        upstream_manifest={
            "evaluated_from_registry": True,
            "version": entry.version,
            "artifact_file": entry.manifest.get("artifact_file"),
            "artifact_sha256": entry.manifest.get("artifact_sha256"),
            "trained_at": entry.manifest.get("trained_at"),
        },
        training={"evaluated": True, "epochs": entry.manifest.get("epochs"), "device": device},
        split=split,
        digest=digest,
    )
    plot_confusion(detail, config.name, FIGURES_DIR / f"{config.name}_confusion.png")
    _log.info(
        "evaluation complete",
        extra={"module": MODULE_NAME, "version": entry.version, "passed": gate.passed},
    )
    if not gate.passed:
        raise AgriError(
            f"Metric gate failed for {MODULE_NAME} v{entry.version}: {', '.join(gate.failed_checks)}",
            context={"metrics": metrics, "gate": gate.to_dict()},
        )
    return {
        "module": MODULE_NAME,
        "version": entry.version,
        "evaluated_from_registry": True,
        "synthetic_data": provenance.SYNTHETIC_DATA,
        "limitation": provenance.LIMITATION,
        "metrics": metrics,
        "per_class_f1": per_class_f1,
        "gate": gate.to_dict(),
    }
