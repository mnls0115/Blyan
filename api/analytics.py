#!/usr/bin/env python3
"""
Analytics API endpoints
- POST /analytics/event: ingest client events (page views, timings)
- GET /analytics/summary: return aggregated metrics for a time window

Constraints:
- No PII, small payloads, CORS controlled by global middleware
- No new dependencies (use Pydantic already present in project)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from backend.analytics.store import get_store


router = APIRouter(prefix="/analytics", tags=["analytics"])


class AnalyticsEvent(BaseModel):
    event_type: str = Field(..., description="Type of event, e.g., page_view, inference_client_timing")
    client_id: Optional[str] = Field(None, description="Anonymous UUID stored in localStorage")
    route: Optional[str] = Field(None, description="Frontend route or page path")
    model: Optional[str] = Field(None, description="Model or variant identifier if applicable")
    status: Optional[str] = Field(None, description="success|error for inference events")
    duration_ms: Optional[float] = Field(None, ge=0, description="Duration in ms for timing events")
    meta: Optional[Dict[str, Any]] = Field(None, description="Small metadata map; prompt text not allowed")
    ts_ms: Optional[int] = Field(None, description="Client timestamp (ms)")


def _analytics_enabled() -> bool:
    return os.getenv("ANALYTICS_ENABLED", "true").lower() in ("1", "true", "yes")


@router.post("/event")
async def ingest_event(evt: AnalyticsEvent, bg: BackgroundTasks) -> Dict[str, Any]:
    if not _analytics_enabled():
        raise HTTPException(status_code=503, detail="Analytics disabled")

    # Enforce size and content constraints
    if evt.meta is not None:
        # Disallow raw prompt or response text
        banned_keys = {"prompt", "response", "messages", "content"}
        if any(k in evt.meta for k in banned_keys):
            raise HTTPException(status_code=400, detail="Invalid meta keys")

    # Clamp route length
    if evt.route and len(evt.route) > 256:
        evt.route = evt.route[:256]

    # Clamp model length
    if evt.model and len(evt.model) > 128:
        evt.model = evt.model[:128]

    # Server-side timestamp fallback
    ts_ms = evt.ts_ms or int(time.time() * 1000)

    # Background write to avoid blocking
    def _write():
        get_store().record_event(
            event_type=evt.event_type,
            client_id=evt.client_id,
            route=evt.route,
            model=evt.model,
            status=evt.status,
            duration_ms=evt.duration_ms,
            meta=evt.meta,
            ts_ms=ts_ms,
        )

    bg.add_task(_write)
    return {"status": "ok"}


@router.get("/summary")
async def get_summary(window: str = "24h") -> Dict[str, Any]:
    if not _analytics_enabled():
        raise HTTPException(status_code=503, detail="Analytics disabled")

    # parse window like 1h, 24h, 7d
    try:
        unit = window[-1].lower()
        val = int(window[:-1])
        if unit == "h":
            seconds = val * 3600
        elif unit == "d":
            seconds = val * 86400
        elif unit == "m":
            seconds = val * 60
        else:
            seconds = int(window)
    except Exception:
        seconds = 24 * 3600

    return get_store().summary(window_seconds=seconds)
