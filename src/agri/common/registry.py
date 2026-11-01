"""Versioned model registry.

Layout::

    models/registry/<module>/<version>/model.joblib
    models/registry/<module>/<version>/manifest.json
    models/registry/<module>/LATEST            # one line: the promoted version

The manifest is the audit trail: which data digest, which config digest, which
commit, which metrics, which library versions produced this artefact. Anything
that cannot be reproduced from the manifest is a bug.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from agri.common.errors import ModelNotTrainedError, RegistryError
from agri.common.logging import get_logger
from agri.common.paths import MODELS_DIR, ensure_dir, git_sha, read_json, sha256_file, write_json

_log = get_logger("registry")

LATEST_POINTER = "LATEST"
MANIFEST_NAME = "manifest.json"
DEFAULT_ARTIFACT = "model.joblib"

_CORE_LIBS = ("numpy", "pandas", "scikit-learn", "xgboost", "scipy", "statsmodels", "torch")


def _lib_versions() -> dict[str, str]:
    """Versions of the libraries that affect numerical output."""
    versions: dict[str, str] = {}
    from importlib.metadata import PackageNotFoundError, version

    for lib in _CORE_LIBS:
        try:
            versions[lib] = version(lib)
        except PackageNotFoundError:
            continue
    return versions


@dataclass
class RegistryEntry:
    """A single trained artefact plus its manifest."""

    name: str
    version: str
    path: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_path(self) -> Path:
        return self.path / self.manifest.get("artifact_file", DEFAULT_ARTIFACT)

    @property
    def metrics(self) -> dict[str, float]:
        return self.manifest.get("metrics", {}) or {}

    @property
    def trained_at(self) -> str:
        return self.manifest.get("trained_at", "")

    def exists(self) -> bool:
        return self.artifact_path.is_file()

    def load(self) -> Any:
        """Deserialise the artefact. Joblib is used for all sklearn-family models."""
        if not self.exists():
            raise ModelNotTrainedError(
                f"Artefact file missing for {self.name} v{self.version}",
                context={"expected": str(self.artifact_path)},
            )
        return joblib.load(self.artifact_path)


class ModelRegistry:
    """Reads and writes versioned artefacts under models/registry."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else MODELS_DIR

    # ------------------------------------------------------------------ paths
    def module_dir(self, name: str) -> Path:
        return self.root / name

    def version_dir(self, name: str, version: str) -> Path:
        return self.module_dir(name) / version

    # ------------------------------------------------------------- discovery
    def versions(self, name: str) -> list[str]:
        directory = self.module_dir(name)
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_dir())

    def latest_version(self, name: str) -> str | None:
        """Promoted version if the pointer exists, else the newest by trained_at."""
        pointer = self.module_dir(name) / LATEST_POINTER
        if pointer.is_file():
            promoted = pointer.read_text(encoding="utf-8").strip()
            if promoted and self.version_dir(name, promoted).is_dir():
                return promoted
        versions = self.versions(name)
        if not versions:
            return None
        return max(versions, key=lambda v: self._trained_at(name, v) or "")

    def _trained_at(self, name: str, version: str) -> str:
        manifest = self.version_dir(name, version) / MANIFEST_NAME
        if not manifest.is_file():
            return ""
        try:
            return str(read_json(manifest).get("trained_at", ""))
        except Exception:  # a corrupt manifest must not break discovery
            return ""

    def get(self, name: str, version: str | None = None) -> RegistryEntry:
        """Resolve an entry, raising ModelNotTrainedError when absent."""
        resolved = version or self.latest_version(name)
        if resolved is None:
            raise ModelNotTrainedError(
                f"No trained artefact registered for module '{name}'",
                context={"registry_root": str(self.root), "hint": "run the module's train step"},
            )
        directory = self.version_dir(name, resolved)
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise RegistryError(f"Missing manifest for {name} v{resolved}", context={"path": str(directory)})
        entry = RegistryEntry(name=name, version=resolved, path=directory, manifest=read_json(manifest_path))
        if not entry.exists():
            raise ModelNotTrainedError(
                f"Manifest present but artefact missing for {name} v{resolved}",
                context={"expected": str(entry.artifact_path)},
            )
        return entry

    def exists(self, name: str, version: str | None = None) -> bool:
        try:
            self.get(name, version)
        except (ModelNotTrainedError, RegistryError):
            return False
        return True

    def list_entries(self, name: str | None = None) -> list[RegistryEntry]:
        """Every registered entry, newest first. Used by the model-info endpoint."""
        names = (
            [name]
            if name
            else sorted(p.name for p in self.root.iterdir() if p.is_dir())
            if self.root.is_dir()
            else []
        )
        entries: list[RegistryEntry] = []
        for module in names:
            for version in self.versions(module):
                manifest_path = self.version_dir(module, version) / MANIFEST_NAME
                if manifest_path.is_file():
                    try:
                        manifest = read_json(manifest_path)
                    except Exception:
                        continue
                    entries.append(
                        RegistryEntry(module, version, self.version_dir(module, version), manifest)
                    )
        if name:
            entries.sort(key=lambda e: e.trained_at, reverse=True)
        return entries

    # ----------------------------------------------------------------- writes
    def save(
        self,
        name: str,
        version: str,
        artifact: Any,
        *,
        metrics: dict[str, float] | None = None,
        config_digest: str | None = None,
        data_digest: str | None = None,
        dataset: str | None = None,
        params: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        promote: bool = True,
        artifact_file: str = DEFAULT_ARTIFACT,
    ) -> RegistryEntry:
        """Persist an artefact and its manifest atomically, optionally promoting it."""
        directory = ensure_dir(self.version_dir(name, version))
        artifact_path = directory / artifact_file
        tmp_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")

        try:
            joblib.dump(artifact, tmp_path)
            tmp_path.replace(artifact_path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RegistryError(f"Failed to serialise artefact for {name} v{version}: {exc}") from exc

        manifest: dict[str, Any] = {
            "name": name,
            "version": version,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "artifact_file": artifact_file,
            "artifact_sha256": sha256_file(artifact_path),
            "artifact_bytes": artifact_path.stat().st_size,
            "data_digest": data_digest,
            "dataset": dataset,
            "config_digest": config_digest,
            "git_sha": git_sha(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "libraries": _lib_versions(),
            "metrics": metrics or {},
            "params": params or {},
        }
        if extra:
            manifest.update(extra)
        write_json(directory / MANIFEST_NAME, manifest)
        _log.info(
            "registered artefact",
            extra={"module": name, "version": version, "metrics": metrics or {}, "promote": promote},
        )
        if promote:
            self.promote(name, version)
        return RegistryEntry(name=name, version=version, path=directory, manifest=manifest)

    def promote(self, name: str, version: str) -> None:
        """Point LATEST at a version. Rollback is one call away."""
        if not self.version_dir(name, version).is_dir():
            raise RegistryError(f"Cannot promote unknown version {version} of {name}")
        ensure_dir(self.module_dir(name))
        (self.module_dir(name) / LATEST_POINTER).write_text(version + "\n", encoding="utf-8")
        _log.info("promoted version", extra={"module": name, "version": version})

    def delete(self, name: str, version: str) -> None:
        """Remove a version. Refuses to delete the promoted one."""
        if self.latest_version(name) == version:
            raise RegistryError(
                f"Refusing to delete the promoted version {version} of {name}; promote another first"
            )
        directory = self.version_dir(name, version)
        if directory.is_dir():
            for child in sorted(directory.rglob("*"), reverse=True):
                child.unlink() if child.is_file() else child.rmdir()
            directory.rmdir()
