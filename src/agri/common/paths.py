"""Canonical filesystem layout. Nothing else in the codebase hardcodes a path."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _discover_root() -> Path:
    """Find the project root: AGRI_PROJECT_ROOT, else the nearest pyproject.toml."""
    override = os.getenv("AGRI_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # src/agri/common/paths.py -> project root is three levels up from src/
    return here.parents[3]


PROJECT_ROOT: Path = _discover_root()
SRC_DIR: Path = PROJECT_ROOT / "src"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

CONFIG_DIR: Path = Path(os.getenv("AGRI_CONFIG_DIR") or PROJECT_ROOT / "configs")

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = Path(os.getenv("AGRI_MODEL_REGISTRY") or PROJECT_ROOT / "models" / "registry")
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
CACHE_DIR: Path = DATA_DIR / "external" / "cache"


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def atomic_write(path: str | Path, mode: str = "w", encoding: str = "utf-8") -> Iterator[Any]:
    """Write a file atomically: temp file in the same directory, then os.replace.

    A crash mid-write leaves the previous file intact, which matters for model
    artefacts and metrics reports that CI reads.
    """
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, mode, encoding=None if "b" in mode else encoding) as handle:
            yield handle
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Serialise *payload* to JSON atomically. Returns the written path."""
    with atomic_write(path) as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True, default=str)
        handle.write("\n")
    return Path(path)


def read_json(path: str | Path) -> Any:
    """Read JSON, raising a clear error if the file is missing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file - used to pin datasets in the registry manifest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a string, for config digests."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_paths(paths: list[str | Path]) -> str:
    """Order-independent aggregate digest of several files (dataset + config)."""
    digest = hashlib.sha256()
    for item in sorted(str(p) for p in paths):
        p = Path(item)
        digest.update(str(p.name).encode())
        if p.is_file():
            digest.update(sha256_file(p).encode())
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def git_sha(short: bool = True) -> str:
    """Current commit SHA, or 'unknown' outside a git checkout. Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short" if short else "--verify", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
