"""Corpus indexing, deterministic splits and the map-style image dataset.

Real layout on disk (verified, not assumed)::

    data/raw/soil_images/<class_name>/*.jpg      # 420 images per class

There is NO train/val/test directory level. The split is therefore derived here,
stratified per class, from the configured seed. The same function is used by
training and by evaluation, so the test partition is reconstructed identically
without ever serialising it.

The dataset class deliberately does not subclass torch.utils.data.Dataset: it only
implements __len__ and __getitem__, which is all DataLoader requires. That keeps
this module importable when torch is not installed, which the default Docker image
depends on.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agri.common.errors import ConfigError, DataValidationError
from agri.common.logging import get_logger
from agri.common.paths import sha256_file

_log = get_logger("modules.soil_image.dataset")

#: Extensions the indexer accepts. Case-insensitive.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class ImageRecord:
    """One indexed image: where it lives and which class it belongs to."""

    path: Path
    label: str
    label_index: int


@dataclass
class DatasetIndex:
    """Every image in the corpus, grouped by class, plus the declared class order."""

    root: Path
    classes: list[str]
    records: dict[str, list[ImageRecord]] = field(default_factory=dict)

    def all_records(self) -> list[ImageRecord]:
        return [record for label in self.classes for record in self.records[label]]

    def counts(self) -> dict[str, int]:
        return {label: len(self.records[label]) for label in self.classes}


@dataclass
class SplitIndex:
    """The three partitions, as image records."""

    train: list[ImageRecord] = field(default_factory=list)
    validation: list[ImageRecord] = field(default_factory=list)
    test: list[ImageRecord] = field(default_factory=list)
    max_train_per_class: int | None = None

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def per_class_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for name in SPLIT_NAMES:
            partition = getattr(self, name)
            counts: dict[str, int] = {}
            for record in partition:
                counts[record.label] = counts.get(record.label, 0) + 1
            out[name] = counts
        return out


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def index_dataset(raw_dir: Path, classes: Iterable[str]) -> DatasetIndex:
    """Walk the corpus and return a validated index.

    Fails loudly on the two mistakes that silently corrupt a training run:

      * a declared class directory is missing or empty (the model would be trained
        without that class and every metric for it would be zero),
      * an image directory exists that the config does not declare (the run would
        silently drop data, and the class list and corpus have drifted apart).
    """
    declared = [str(name) for name in classes]
    if not declared:
        raise ConfigError("soil_image config declares no classes", context={"raw_dir": str(raw_dir)})
    if not raw_dir.is_dir():
        raise ConfigError(
            "Soil image corpus directory is missing",
            context={"raw_dir": str(raw_dir), "hint": "run data/raw/soil_images/generate_soil_images.py"},
        )

    index = DatasetIndex(root=raw_dir, classes=declared)
    for label_index, label in enumerate(declared):
        directory = raw_dir / label
        if not directory.is_dir():
            raise ConfigError(
                "A declared soil class has no directory in the corpus",
                context={"class": label, "expected": str(directory), "declared": declared},
            )
        images = _list_images(directory)
        if not images:
            raise ConfigError(
                "A declared soil class directory contains no images",
                context={"class": label, "directory": str(directory)},
            )
        index.records[label] = [
            ImageRecord(path=path, label=label, label_index=label_index) for path in images
        ]

    undeclared = sorted(
        child.name
        for child in raw_dir.iterdir()
        if child.is_dir() and child.name not in declared and _list_images(child)
    )
    if undeclared:
        raise ConfigError(
            "Corpus contains class directories the config does not declare",
            context={"undeclared": undeclared, "declared": declared, "raw_dir": str(raw_dir)},
        )
    return index


def stratified_split(
    records: dict[str, list[ImageRecord]],
    *,
    seed: int,
    val_size: float,
    test_size: float,
    class_order: Iterable[str],
    max_train_per_class: int | None = None,
) -> SplitIndex:
    """Deterministically split each class into train/validation/test.

    The split is per class, so every partition is stratified by construction. It
    is derived from (seed, class order, filename order) alone, which is why
    evaluation can rebuild the exact test partition months later without a
    serialised split file.

    max_train_per_class caps only the training partition. Validation and test keep
    their full size, so a subset run yields a smaller model but the reported metric
    is still measured on the whole test partition - a subset run must not quietly
    shrink the evidence.
    """
    if val_size < 0.0 or test_size < 0.0 or val_size + test_size >= 1.0:
        raise ConfigError(
            "split.val_size plus split.test_size must leave a non-empty training partition",
            context={"val_size": val_size, "test_size": test_size},
        )

    rng = np.random.default_rng(seed)
    split = SplitIndex(max_train_per_class=max_train_per_class)
    for label in class_order:
        items = sorted(records[label], key=lambda record: record.path.name)
        count = len(items)
        if count < 3:
            raise ConfigError(
                f"Class '{label}' has only {count} images; at least 3 are needed for a three-way split",
                context={"class": label, "count": count},
            )
        order = rng.permutation(count)
        shuffled = [items[int(position)] for position in order]
        n_test = max(1, round(count * test_size))
        n_val = max(1, round(count * val_size))
        n_train = count - n_test - n_val
        if n_train < 1:
            raise ConfigError(
                f"Class '{label}' leaves no training images after the split",
                context={"class": label, "count": count, "n_test": n_test, "n_val": n_val},
            )
        split.test.extend(shuffled[:n_test])
        split.validation.extend(shuffled[n_test : n_test + n_val])
        train = shuffled[n_test + n_val :]
        if max_train_per_class is not None:
            if max_train_per_class < 1:
                raise ConfigError(
                    "max_train_per_class must be at least 1",
                    context={"max_train_per_class": max_train_per_class},
                )
            train = train[:max_train_per_class]
        split.train.extend(train)

    if not split.train:
        raise ConfigError("The training partition is empty", context=split.counts())
    _log.info("split built", extra=split.counts())
    return split


def dataset_digest(index: DatasetIndex, *, workers: int = 8) -> str:
    """Content digest of every indexed image, used to pin the dataset in a manifest.

    Names alone are not enough: the corpus is regenerated by a script, so a rerun
    can produce different pixels under identical filenames. Hashing content is what
    lets evaluation refuse to attach a fresh metric to a stale model.

    Hashing is I/O bound, so the per-file digests run on a small thread pool. On
    Windows this is the difference between about four seconds and about forty for
    the 4200-file corpus; the resulting digest is order-independent because the file
    list is sorted before hashing.
    """
    order = sorted(index.all_records(), key=lambda item: item.path.as_posix())
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        file_digests = list(pool.map(lambda record: sha256_file(record.path), order))
    digest = hashlib.sha256()
    for record, file_digest in zip(order, file_digests, strict=True):
        relative = record.path.relative_to(index.root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class SoilImageDataset:
    """Map-style dataset returning (normalised tensor, integer label).

    transform is a torchvision pipeline supplied by the caller, so this class needs
    no torch import of its own. Images are converted to RGB on read; anything
    undecodable raises DataValidationError rather than crashing inside a worker.
    """

    def __init__(self, records: list[ImageRecord], transform: Any) -> None:
        if not records:
            raise DataValidationError("Cannot build a dataset from an empty record list")
        self._records = list(records)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, position: int) -> tuple[Any, int]:
        record = self._records[position]
        try:
            with Image.open(record.path) as handle:
                image = handle.convert("RGB")
        except OSError as exc:
            raise DataValidationError(
                "An image in the training corpus is not decodable",
                context={"path": str(record.path), "error": f"{type(exc).__name__}: {exc}"},
            ) from exc
        return self._transform(image), int(record.label_index)

    @property
    def records(self) -> list[ImageRecord]:
        return self._records
