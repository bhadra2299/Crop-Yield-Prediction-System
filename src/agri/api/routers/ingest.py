"""Ingestion endpoints.

An irrigation event is a pump run, a rain gauge reading, or a corrected sensor
value that the platform did not observe itself. Recording it matters for two
reasons: it keeps the water balance honest, and it accumulates exactly the field
ground truth that the irrigation engine currently lacks.

Nothing is written to an external store yet; events append to a JSONL file under
reports/events. That is a deliberate, stated limitation rather than a silent one -
see docs/MODULES/irrigation.md.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from agri.api.security import enforce_rate_limit, require_api_key
from agri.common.logging import get_logger
from agri.common.paths import REPORTS_DIR, ensure_dir

_log = get_logger("api.ingest")

router = APIRouter(
    prefix="/ingest",
    tags=["ingest"],
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)

EVENTS_FILE = "events/irrigation_events.jsonl"


class IrrigationEvent(BaseModel):
    """One observed irrigation or rainfall event."""

    field_id: str = Field(..., min_length=1, description="caller's identifier for the field")
    event_type: str = Field("irrigation", description="irrigation or rainfall")
    occurred_on: str = Field(..., description="ISO date the event occurred")
    depth_mm: float = Field(..., gt=0, le=500, description="applied depth or measured rainfall, mm")
    source: str = Field("manual", description="manual, sensor, or gauge")
    notes: str | None = Field(None, max_length=500)


@router.post("/irrigation-event", status_code=status.HTTP_201_CREATED, summary="Record an irrigation event")
async def irrigation_event(event: IrrigationEvent) -> dict:
    """Append the event to the local journal and echo it back."""
    directory = ensure_dir(REPORTS_DIR / "events")
    path = directory / "irrigation_events.jsonl"
    record = {**event.model_dump(), "recorded_at": datetime.now(UTC).isoformat()}
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:
        # A journal write failure must be visible, not swallowed.
        _log.warning("could not append irrigation event", extra={"error": str(exc), "path": str(path)})
        return {"ok": False, "detail": f"Event could not be persisted: {exc}", "event": record}

    _log.info(
        "irrigation event recorded",
        extra={"field_id": event.field_id, "event_type": event.event_type, "depth_mm": event.depth_mm},
    )
    return {"ok": True, "path": str(path), "event": record}


@router.get("/irrigation-events", summary="Read back recorded irrigation events")
async def list_events(field_id: str | None = None, limit: int = 100) -> dict:
    path = REPORTS_DIR / EVENTS_FILE
    if not path.is_file():
        return {"count": 0, "events": [], "detail": "No irrigation events have been recorded yet."}

    events: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if field_id and record.get("field_id") != field_id:
                continue
            events.append(record)

    return {"count": len(events), "events": events[-max(1, min(limit, 1000)) :]}
