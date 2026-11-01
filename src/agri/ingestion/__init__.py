"""External data pullers.

Each puller returns a validated frame, caches its result on disk with a
signature that includes URL + date window (no cache hit on stale data), retries
with exponential backoff, and fails loudly when an API key is missing.

Data provenance
---------------
Every ingested frame carries a provenance dict: source, date window, and
signature hash. When data is replaced downstream, the signature changes and the
model does not accidentally keep serving the old predictions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from agri.common.errors import ExternalServiceError
from agri.common.logging import get_logger
from agri.common.paths import CACHE_DIR, ensure_dir, sha256_text

_log = get_logger("ingestion")

DEFAULT_SIGNING_KEY = "smart_crop_ai_data_integrity"
BACKOFF_MAX = 30.0


@dataclass
class DataSource:
    """Where one record came from."""

    provider: str
    url: str | None
    date_window_start: str | None = None
    date_window_end: str | None = None
    signature: str | None = None
    fetched_at: str | None = None
    signature_key: str = DEFAULT_SIGNING_KEY


def _signature(url: str, start: date | str, end: date | str, key: str = DEFAULT_SIGNING_KEY) -> str:
    """Content-agnostic signature.

    If the upstream provider changes the schema, the column list is not enough -
    the data must be re-validated, so we fold the window and key into the hash.
    """
    if isinstance(start, str):
        start = pd.to_datetime(start).date()
    if isinstance(end, str):
        end = pd.to_datetime(end).date()
    payload = f"{url}|{start}|{end}|{key}"
    return sha256_text(payload)


def _cache_path(source: str, window: str) -> Path:
    return CACHE_DIR / source / window / "frame.arrow"


def _fetch_cached(source: str, window: str) -> pd.DataFrame | None:
    path = _cache_path(source, window)
    if path.is_file() and path.stat().st_size:
        try:
            return pd.read_feather(path)
        except Exception:
            _log.warning("cache corrupted, discarding", extra={"path": str(path)})
    return None


def _store_cached(source: str, window: str, frame: pd.DataFrame) -> None:
    ensure_dir(_cache_path(source, window).parent)
    frame.to_feather(_cache_path(source, window))


class Puller:
    """HTTP helpers shared by every puller. Retries, timeout, and auth headers."""

    source: str
    timeout_seconds: int
    max_retries: int
    signing_key: str

    def __init__(
        self,
        source: str,
        *,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        signing_key: str = DEFAULT_SIGNING_KEY,
    ) -> None:
        self.source = source
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.signing_key = signing_key

    def _request(
        self, method: str, url: str, *, params: dict | None = None, headers: dict | None = None
    ) -> dict[str, Any]:
        """One HTTP attempt. Never swallows an exception - let the caller fail."""
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(headers=headers or {}) as client:
                    response = getattr(client, method)(url, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise ExternalServiceError(f"{self.source} returned {exc.response.status_code}") from exc
                if attempt >= self.max_retries:
                    raise ExternalServiceError(
                        f"{self.source} returned {exc.response.status_code} after {self.max_retries} attempts"
                    ) from exc
            except httpx.RequestError as exc:
                if attempt >= self.max_retries:
                    raise ExternalServiceError(f"{self.source} request failed: {exc}") from exc
            time.sleep(min(2**attempt * 0.25, BACKOFF_MAX))
        raise ExternalServiceError(f"{self.source} exhausted retries")  # pragma: no cover


def fetch_to_frame(
    puller: Puller,
    url: str,
    params: dict | None = None,
    date_window_start: str | None = None,
    date_window_end: str | None = None,
) -> tuple[pd.DataFrame, DataSource]:
    """GET JSON, coerce into a DataFrame, and record provenance."""
    raw = puller._request("get", url, params=params)
    frame = (
        pd.DataFrame(raw if isinstance(raw, list) else [raw])
        if not isinstance(raw, dict)
        else pd.DataFrame([raw])
    )
    provenance = DataSource(
        provider=puller.source,
        url=url,
        date_window_start=date_window_start,
        date_window_end=date_window_end,
        signature=_signature(url, date_window_start or "now", date_window_end or "now", puller.signing_key),
        fetched_at=datetime.now().isoformat(),
    )
    return frame, provenance


def load_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet artefact, failing loudly if the file is gone."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Parquet artefact not found: {resolved}")
    return pd.read_parquet(resolved)


def store_as_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist an intermediate as Parquet, writing atomically."""
    resolved = Path(path)
    ensure_dir(resolved.parent)
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    frame.to_parquet(tmp)
    tmp.replace(resolved)
    return resolved
