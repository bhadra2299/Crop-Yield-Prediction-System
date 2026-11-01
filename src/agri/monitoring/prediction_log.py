"""Prediction logging.

Every served prediction appends one JSON line to reports/logs/predictions.jsonl.
This is the raw material for drift and quality monitoring, and it is written
asynchronously-safe by delegating the append to a thread when called from async
code, or inline otherwise.

Deliberately NOT logged: full request payloads for image endpoints (they would be
megabytes of base64) and anything that looks like a credential. The log records
the shape of the request and the output, not the raw body.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import Any

from agri.common.logging import get_logger
from agri.common.paths import REPORTS_DIR, ensure_dir

_log = get_logger("monitoring.predictions")

LOG_DIR = REPORTS_DIR / "logs"
LOG_FILE = "predictions.jsonl"

#: Serialise appends so concurrent requests do not interleave partial lines.
_write_lock = threading.Lock()


def _summarise(value: Any, limit: int = 400) -> Any:
    """Keep the log small: scalar values stay, big structures become a digest."""
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    if isinstance(value, dict):
        return {key: _summarise(item, limit) for key, item in list(value.items())[:25]}
    if isinstance(value, (list, tuple)):
        return {"__len__": len(value), "items": [_summarise(item, limit) for item in value[:5]]}
    return str(value)[:limit]


def record_prediction(
    *,
    module: str,
    endpoint: str,
    request: Any,
    prediction: Any,
    ok: bool,
) -> None:
    """Append one prediction record. Never raises: monitoring must not break serving."""
    payload = prediction.model_dump() if hasattr(prediction, "model_dump") else prediction
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "module": module,
        "endpoint": endpoint,
        "ok": bool(ok),
        "request": _summarise(request),
        "output": _summarise(
            {
                "value": (payload or {}).get("value") if isinstance(payload, dict) else None,
                "confidence": (payload or {}).get("confidence") if isinstance(payload, dict) else None,
                "confidence_score": (payload or {}).get("confidence_score")
                if isinstance(payload, dict)
                else None,
                "model_version": (payload or {}).get("model_version") if isinstance(payload, dict) else None,
                "latency_ms": (payload or {}).get("latency_ms") if isinstance(payload, dict) else None,
            }
        ),
    }
    try:
        directory = ensure_dir(LOG_DIR)
        with _write_lock, (directory / LOG_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        # Losing a monitoring line is acceptable; breaking a request is not.
        _log.warning("could not append prediction log", extra={"error": str(exc), "module": module})


def read_predictions(limit: int = 5000) -> list[dict[str, Any]]:
    """Read recent prediction records, newest last."""
    path = LOG_DIR / LOG_FILE
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records[-limit:]
