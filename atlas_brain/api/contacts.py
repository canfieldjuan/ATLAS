"""
Contact and call transcript endpoints.

Provides:
- GET /contacts/{id}/timeline -- unified chronological view of all customer activity
- GET /comms/calls/search -- search call transcripts by keyword, date, contact, intent
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..services.customer_context import get_customer_context_service
from ..storage.repositories.call_transcript import get_call_transcript_repo

logger = logging.getLogger("atlas.api.contacts")

router = APIRouter(tags=["contacts"])


# ---------------------------------------------------------------------------
# Contact timeline
# ---------------------------------------------------------------------------

@router.get("/contacts/{contact_id}/timeline")
async def contact_timeline(
    contact_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    """Unified chronological timeline for a contact.

    Merges interactions, call transcripts, appointments, and emails
    into a single sorted feed.
    """
    svc = get_customer_context_service()
    ctx = await svc.get_context(
        contact_id=contact_id,
        max_interactions=limit,
        max_calls=limit,
        max_appointments=limit,
        max_emails=limit,
    )
    if ctx.is_empty:
        raise HTTPException(status_code=404, detail="Contact not found")

    events: list[dict] = []

    # Interactions
    for ix in ctx.interactions:
        ts = ix.get("occurred_at")
        events.append({
            "type": "interaction",
            "subtype": ix.get("interaction_type", ""),
            "timestamp": _to_iso(ts),
            "summary": ix.get("summary", ""),
            "id": str(ix.get("id", "")),
        })

    # Call transcripts
    for call in ctx.call_transcripts:
        ts = call.get("created_at")
        events.append({
            "type": "call",
            "subtype": call.get("status", ""),
            "timestamp": _to_iso(ts),
            "summary": call.get("summary") or f"Call ({call.get('duration_seconds', 0)}s)",
            "id": str(call.get("id", "")),
            "from_number": call.get("from_number", ""),
            "duration_seconds": call.get("duration_seconds"),
            "intent": (call.get("extracted_data") or {}).get("intent", ""),
        })

    # Appointments
    for appt in ctx.appointments:
        ts = appt.get("start_time")
        events.append({
            "type": "appointment",
            "subtype": appt.get("status", ""),
            "timestamp": _to_iso(ts),
            "summary": appt.get("service_type") or "Appointment",
            "id": str(appt.get("id", "")),
            "notes": appt.get("notes", ""),
        })

    # Sent emails
    for em in ctx.sent_emails:
        ts = em.get("sent_at")
        events.append({
            "type": "email_sent",
            "subtype": "",
            "timestamp": _to_iso(ts),
            "summary": em.get("subject", ""),
            "id": str(em.get("id", "")),
        })

    # Inbox emails
    for em in ctx.inbox_emails:
        ts = em.get("date") or em.get("received_at")
        events.append({
            "type": "email_received",
            "subtype": "",
            "timestamp": _to_iso(ts),
            "summary": em.get("subject", ""),
            "id": str(em.get("id", "")),
        })

    # Sort descending by timestamp (most recent first)
    events.sort(key=lambda e: e.get("timestamp") or "", reverse=True)

    # Apply limit
    events = events[:limit]

    return {
        "contact_id": contact_id,
        "contact_name": ctx.display_name,
        "total_events": len(events),
        "events": events,
    }


# ---------------------------------------------------------------------------
# Call transcript search
# ---------------------------------------------------------------------------

@router.get("/comms/calls/search")
async def search_calls(
    q: Optional[str] = Query(default=None, description="Keyword search on transcript and summary"),
    contact_id: Optional[str] = Query(default=None, description="Filter by CRM contact ID"),
    intent: Optional[str] = Query(default=None, description="Filter by extracted intent"),
    from_date: Optional[str] = Query(default=None, description="Start date (ISO 8601)"),
    to_date: Optional[str] = Query(default=None, description="End date (ISO 8601)"),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Search call transcripts with keyword and structured filters."""
    repo = get_call_transcript_repo()

    from_dt = _parse_date(from_date) if from_date else None
    to_dt = _parse_date(to_date) if to_date else None

    results = await repo.search(
        keyword=q,
        contact_id=contact_id,
        intent=intent,
        from_date=from_dt,
        to_date=to_dt,
        limit=limit,
    )

    # Build lightweight result list (don't return full transcripts in search)
    items = []
    for r in results:
        items.append({
            "id": str(r.get("id", "")),
            "call_sid": r.get("call_sid", ""),
            "from_number": r.get("from_number", ""),
            "to_number": r.get("to_number", ""),
            "duration_seconds": r.get("duration_seconds"),
            "summary": r.get("summary", ""),
            "intent": (r.get("extracted_data") or {}).get("intent", ""),
            "customer_name": (r.get("extracted_data") or {}).get("customer_name", ""),
            "status": r.get("status", ""),
            "contact_id": str(r.get("contact_id", "")) if r.get("contact_id") else None,
            "created_at": _to_iso(r.get("created_at")),
        })

    return {
        "total": len(items),
        "results": items,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_iso(val) -> str:
    """Convert a datetime or string to ISO format string."""
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _parse_date(s: str) -> Optional[datetime]:
    """Parse an ISO 8601 date string."""
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None
