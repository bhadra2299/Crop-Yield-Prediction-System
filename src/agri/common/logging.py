"""Structured logging with request correlation.

Production emits one JSON object per line so a log shipper can index fields.
Development emits a readable single line.

Structured context is passed the normal way:

    log.info("gate evaluated", extra={"module": "risk", "passed": False})

Several of the keys that callers naturally reach for - module, name, args,
filename, process, thread - are reserved by LogRecord, and passing them raises
KeyError from deep inside the stdlib. Instead of asking every call site to
remember that, LogRecordSafe stashes the colliding keys and the formatter merges
them back, so the JSON output keeps the field name the caller intended.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_CONFIGURED = False
_LOG_LEVEL_APPLIED = -1

#: Attributes the stdlib sets on every record; writing any of them via extra
#: is a KeyError, so they are diverted instead.
_RESERVED_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

#: Where diverted keys are parked until the formatter picks them up.
_STASH = "_ctx_fields"


class LogRecordSafe(logging.Logger):
    """Logger that tolerates structured keys colliding with LogRecord fields."""

    def makeRecord(
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: Any,
        args: Any,
        exc_info: Any,
        func: str | None = None,
        extra: dict[str, Any] | None = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord:
        if extra:
            extra = dict(extra)
            diverted = {key: extra.pop(key) for key in list(extra) if key in _RESERVED_KEYS}
            if diverted:
                # Merge rather than overwrite: nested helpers may add context too.
                merged = dict(extra.get(_STASH) or {})
                merged.update(diverted)
                extra[_STASH] = merged
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)


# Install before any logger is created, so module-level loggers are safe too.
logging.setLoggerClass(LogRecordSafe)


class JsonFormatter(logging.Formatter):
    """One JSON object per record, including anything passed via extra."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        diverted = record.__dict__.get(_STASH) or {}
        for key, value in record.__dict__.items():
            if key not in _RESERVED_KEYS and not key.startswith("_") and key != "request_id":
                payload[key] = value
        for key, value in diverted.items():
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable line with a short request id and inline context."""

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_ctx.get()
        rid = rid[:8] if rid != "-" else "-"
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} "
            f"{rid:<8} {record.name} :: {record.getMessage()}"
        )
        diverted = record.__dict__.get(_STASH) or {}
        if diverted:
            rendered = " ".join(f"{key}={value}" for key, value in sorted(diverted.items()))
            base += "  [" + rendered + "]"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(level: str | None = None, *, json_output: bool | None = None) -> None:
    """Configure the root logger. Safe to call repeatedly; re-applies a new level."""
    global _CONFIGURED, _LOG_LEVEL_APPLIED
    resolved_level = (level or os.getenv("AGRI_LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)
    if json_output is None:
        json_output = (os.getenv("AGRI_LOG_JSON") or "false").strip().lower() in {"1", "true", "yes"}

    if _CONFIGURED and numeric_level == _LOG_LEVEL_APPLIED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)
    for noisy in ("uvicorn.access", "httpx", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
    _LOG_LEVEL_APPLIED = numeric_level


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    if not _CONFIGURED:
        setup_logging()
    logger = logging.getLogger(name if name.startswith("agri") else "agri." + name)
    # A logger may pre-date the class swap (e.g. created by an import cycle).
    if not isinstance(logger, LogRecordSafe):
        logger.__class__ = LogRecordSafe
    return logger


def new_request_id() -> str:
    """Generate and bind a request id for the current context."""
    rid = uuid.uuid4().hex
    request_id_ctx.set(rid)
    return rid
