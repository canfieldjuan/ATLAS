"""
Atlas Calendar MCP Server.

Provider-agnostic MCP server exposing calendar operations to any MCP-compatible
client (Claude Desktop, Cursor, custom agents, etc.).

Providers (configured via env vars — no code changes needed to swap):
    Google Calendar — ATLAS_TOOLS_CALENDAR_ENABLED=true
                      (run scripts/setup_google_oauth.py once to get tokens)
    CalDAV          — ATLAS_TOOLS_CALDAV_URL + ATLAS_TOOLS_CALDAV_USERNAME
                      + ATLAS_TOOLS_CALDAV_PASSWORD
                      Compatible with Nextcloud, Apple Calendar, Fastmail,
                      Proton Calendar, SOGo, Baikal, Radicale, …

Tools:
    list_calendars    — list all calendars available on the account
    list_events       — list events in an ISO-8601 date/time range
    get_event         — fetch a specific event by ID
    create_event      — create a new event
    update_event      — update fields on an existing event
    delete_event      — delete / cancel an event
    find_free_slots   — find open scheduling windows (avoids conflicts)
    sync_appointment  — sync a local DB appointment → calendar event

Run:
    python -m atlas_brain.mcp.calendar_server          # stdio (Claude Desktop / Cursor)
    python -m atlas_brain.mcp.calendar_server --sse    # SSE HTTP transport (port 8059)
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.calendar")


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from ..storage.database import init_database, close_database
    await init_database()
    logger.info("Calendar MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-calendar",
    instructions=(
        "Calendar server for Atlas. "
        "Use list_calendars first to discover available calendars. "
        "Always call find_free_slots before scheduling new appointments to avoid "
        "double-booking. "
        "Use sync_appointment to keep local DB appointments in sync with the calendar "
        "after creating or updating them via the CRM / scheduling tools."
    ),
    lifespan=_lifespan,
)


def _provider():
    from ..services.calendar_provider import get_calendar_provider

    return get_calendar_provider()


def _parse_dt(dt_str: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware datetime."""
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Tool: list_calendars
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_calendars() -> str:
    """List all calendars available to the configured provider account.

    Returns a JSON array of calendar objects with id, name, primary, read_only.
    Use the 'id' field as calendar_id in other tools to target a specific calendar.
    """
    try:
        calendars = await _provider().list_calendars()
        result = [
            {
                "id": c.id,
                "name": c.name,
                "primary": c.primary,
                "read_only": c.read_only,
            }
            for c in calendars
        ]
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.exception("list_calendars error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: list_events
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_events(
    start: str,
    end: str,
    calendar_id: Optional[str] = None,
) -> str:
    """List calendar events in a time range.

    Args:
        start:       ISO-8601 range start (e.g. '2025-03-01T00:00:00Z')
        end:         ISO-8601 range end   (e.g. '2025-03-07T23:59:59Z')
        calendar_id: Optional calendar ID to filter (from list_calendars).
                     Defaults to the primary / default calendar.
    """
    try:
        events = await _provider().list_events(
            start=_parse_dt(start),
            end=_parse_dt(end),
            calendar_id=calendar_id,
        )
        result = [
            {
                "id": e.uid,
                "summary": e.summary,
                "start": e.start.isoformat(),
                "end": e.end.isoformat(),
                "location": e.location,
                "description": e.description,
                "all_day": e.all_day,
                "status": e.status,
                "calendar_id": e.calendar_id,
            }
            for e in events
        ]
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.exception("list_events error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: get_event
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_event(event_id: str, calendar_id: Optional[str] = None) -> str:
    """Get a specific calendar event by ID.

    Args:
        event_id:    Event UID / ID (from list_events)
        calendar_id: Optional calendar ID (required for CalDAV servers that
                     don't support global event lookup)
    """
    try:
        event = await _provider().get_event(event_id, calendar_id=calendar_id)
        if event is None:
            return json.dumps({"error": "Event not found", "event_id": event_id})
        return json.dumps(
            {
                "id": event.uid,
                "summary": event.summary,
                "start": event.start.isoformat(),
                "end": event.end.isoformat(),
                "location": event.location,
                "description": event.description,
                "all_day": event.all_day,
                "status": event.status,
                "calendar_id": event.calendar_id,
            },
            indent=2,
        )
    except Exception as exc:
        logger.exception("get_event error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: create_event
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_event(
    summary: str,
    start: str,
    end: str,
    calendar_id: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
    all_day: bool = False,
) -> str:
    """Create a new calendar event.

    Args:
        summary:     Event title (e.g. 'House cleaning — Smith residence')
        start:       ISO-8601 start datetime (e.g. '2025-03-10T09:00:00-06:00')
        end:         ISO-8601 end datetime   (e.g. '2025-03-10T11:00:00-06:00')
        calendar_id: Optional calendar to create in (default: primary)
        location:    Optional location / address
        description: Optional notes or description
        all_day:     True for all-day events (start/end should be dates only)
    """
    from ..services.calendar_provider import CalendarEvent

    try:
        event = CalendarEvent(
            uid="",
            summary=summary,
            start=_parse_dt(start),
            end=_parse_dt(end),
            location=location,
            description=description,
            all_day=all_day,
        )
        created = await _provider().create_event(event, calendar_id=calendar_id)
        return json.dumps(
            {
                "id": created.uid,
                "summary": created.summary,
                "start": created.start.isoformat(),
                "end": created.end.isoformat(),
                "location": created.location,
                "calendar_id": created.calendar_id,
                "status": "created",
            },
            indent=2,
        )
    except Exception as exc:
        logger.exception("create_event error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: update_event
# ---------------------------------------------------------------------------


@mcp.tool()
async def update_event(
    event_id: str,
    summary: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    calendar_id: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Update an existing calendar event. Only provided fields are changed.

    Args:
        event_id:    Event UID / ID (from list_events or get_event)
        summary:     New event title
        start:       New ISO-8601 start datetime
        end:         New ISO-8601 end datetime
        calendar_id: Calendar the event lives in
        location:    New location / address
        description: New notes or description
    """
    from ..services.calendar_provider import CalendarEvent

    try:
        existing = await _provider().get_event(event_id, calendar_id=calendar_id)
        if existing is None:
            return json.dumps({"error": "Event not found", "event_id": event_id})

        updated = CalendarEvent(
            uid=event_id,
            summary=summary if summary is not None else existing.summary,
            start=_parse_dt(start) if start else existing.start,
            end=_parse_dt(end) if end else existing.end,
            calendar_id=existing.calendar_id,
            location=location if location is not None else existing.location,
            description=description if description is not None else existing.description,
            all_day=existing.all_day,
            status=existing.status,
        )
        result = await _provider().update_event(updated, calendar_id=calendar_id)
        return json.dumps(
            {
                "id": result.uid,
                "summary": result.summary,
                "start": result.start.isoformat(),
                "end": result.end.isoformat(),
                "status": "updated",
            },
            indent=2,
        )
    except Exception as exc:
        logger.exception("update_event error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: delete_event
# ---------------------------------------------------------------------------


@mcp.tool()
async def delete_event(event_id: str, calendar_id: Optional[str] = None) -> str:
    """Delete or cancel a calendar event.

    Args:
        event_id:    Event UID / ID (from list_events)
        calendar_id: Calendar the event lives in
    """
    try:
        success = await _provider().delete_event(event_id, calendar_id=calendar_id)
        return json.dumps({"deleted": success, "event_id": event_id})
    except Exception as exc:
        logger.exception("delete_event error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: find_free_slots
# ---------------------------------------------------------------------------


@mcp.tool()
async def find_free_slots(
    start: str,
    end: str,
    duration_minutes: int = 60,
    calendar_id: Optional[str] = None,
    start_hour: int = 8,
    end_hour: int = 18,
) -> str:
    """Find available (unbooked) time slots within a date range.

    Fetches existing events and returns gaps large enough to fit a new
    appointment of the requested duration.  Use this before create_event or
    sync_appointment to avoid double-booking.

    Args:
        start:            ISO-8601 range start (e.g. '2025-03-10T00:00:00Z')
        end:              ISO-8601 range end   (e.g. '2025-03-14T23:59:59Z')
        duration_minutes: Minimum slot size in minutes (default: 60)
        calendar_id:      Optional calendar to check (default: primary)
        start_hour:       Earliest hour to offer slots (0-23, default: 8)
        end_hour:         Latest hour to end a slot   (0-23, default: 18)

    Returns up to 20 free slots to keep LLM context manageable.
    """
    try:
        range_start = _parse_dt(start)
        range_end = _parse_dt(end)

        events = await _provider().list_events(
            start=range_start,
            end=range_end,
            calendar_id=calendar_id,
        )

        slot_duration = timedelta(minutes=duration_minutes)
        free_slots: list[dict] = []

        current_day = range_start.replace(hour=0, minute=0, second=0, microsecond=0)
        while current_day.date() <= range_end.date():
            tz = current_day.tzinfo
            day_start = current_day.replace(hour=start_hour, minute=0, second=0)
            day_end = current_day.replace(hour=end_hour, minute=0, second=0)

            # Collect busy windows for this day
            busy: list[tuple[datetime, datetime]] = []
            for ev in events:
                ev_start = ev.start if ev.start.tzinfo else ev.start.replace(tzinfo=tz)
                ev_end = ev.end if ev.end.tzinfo else ev.end.replace(tzinfo=tz)
                if ev.all_day:
                    if ev_start.date() <= current_day.date() <= ev_end.date():
                        busy.append((day_start, day_end))
                elif ev_start < day_end and ev_end > day_start:
                    busy.append((max(ev_start, day_start), min(ev_end, day_end)))

            busy.sort()

            # Walk through the day and collect gaps
            cursor = day_start
            for b_start, b_end in busy:
                while cursor + slot_duration <= b_start:
                    free_slots.append(
                        {
                            "start": cursor.isoformat(),
                            "end": (cursor + slot_duration).isoformat(),
                        }
                    )
                    cursor += slot_duration
                cursor = max(cursor, b_end)

            # Remaining slots after the last busy block
            while cursor + slot_duration <= day_end:
                free_slots.append(
                    {
                        "start": cursor.isoformat(),
                        "end": (cursor + slot_duration).isoformat(),
                    }
                )
                cursor += slot_duration

            current_day += timedelta(days=1)

        return json.dumps(
            {
                "free_slots": free_slots[:20],
                "total_found": len(free_slots),
                "duration_minutes": duration_minutes,
            },
            indent=2,
        )
    except Exception as exc:
        logger.exception("find_free_slots error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: sync_appointment
# ---------------------------------------------------------------------------


@mcp.tool()
async def sync_appointment(
    appointment_id: str,
    calendar_id: Optional[str] = None,
) -> str:
    """Sync a local DB appointment to the calendar provider.

    Looks up the appointment by UUID, creates or updates the corresponding
    calendar event, and writes the event ID back to appointments.calendar_event_id
    so the two stay linked.

    Use this after booking a new appointment or rescheduling an existing one
    to keep the calendar in sync with the CRM / scheduling database.

    Args:
        appointment_id: UUID of the appointment in the local database
        calendar_id:    Optional calendar to sync into (default: primary)
    """
    from ..services.calendar_provider import CalendarEvent

    try:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT id, customer_name, customer_address, start_time, end_time,
                   notes, calendar_event_id
            FROM appointments
            WHERE id = $1
            """,
            appointment_id,
        )

        if not row:
            return json.dumps({"error": "Appointment not found", "appointment_id": appointment_id})

        summary = f"Cleaning - {row['customer_name']}"
        description = row["notes"] or ""
        location = row["customer_address"] or ""
        existing_event_id = row["calendar_event_id"]

        def _make_aware(dt: datetime) -> datetime:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        event = CalendarEvent(
            uid=existing_event_id or "",
            summary=summary,
            start=_make_aware(row["start_time"]),
            end=_make_aware(row["end_time"]),
            location=location,
            description=description,
        )

        if existing_event_id:
            result = await _provider().update_event(event, calendar_id=calendar_id)
            action = "updated"
        else:
            result = await _provider().create_event(event, calendar_id=calendar_id)
            action = "created"

        # Write calendar_event_id back to DB so appointments stay linked
        if result.uid:
            await pool.execute(
                "UPDATE appointments SET calendar_event_id = $1 WHERE id = $2",
                result.uid,
                appointment_id,
            )

        return json.dumps(
            {
                "appointment_id": appointment_id,
                "calendar_event_id": result.uid,
                "summary": result.summary,
                "start": result.start.isoformat(),
                "end": result.end.isoformat(),
                "status": action,
            },
            indent=2,
        )
    except Exception as exc:
        logger.exception("sync_appointment error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    sse_mode = "--sse" in sys.argv
    if sse_mode:
        from ..config import settings
        from .auth import run_sse_with_auth

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.calendar_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.calendar_port)
    else:
        mcp.run(transport="stdio")
