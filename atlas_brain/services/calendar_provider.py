"""
Provider-agnostic calendar service.

Providers
---------
GoogleCalendarProvider  — Google Calendar API v3 via OAuth2 (existing token-store pattern)
CalDAVCalendarProvider  — CalDAV / RFC 4791 via httpx (Nextcloud, Apple Calendar,
                          Fastmail, Proton Calendar, SOGo, Baikal, …)

Factory
-------
get_calendar_provider() — returns active provider based on config.

Priority: CalDAV (if ATLAS_TOOLS_CALDAV_URL is set) > Google Calendar
(if ATLAS_TOOLS_CALENDAR_ENABLED=true) > RuntimeError
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol, runtime_checkable

import httpx

from ..config import settings

logger = logging.getLogger("atlas.services.calendar")

# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------

@dataclass
class CalendarEvent:
    """Normalised calendar event across all providers."""
    uid: str
    summary: str
    start: datetime
    end: datetime
    calendar_id: str = ""
    location: Optional[str] = None
    description: Optional[str] = None
    all_day: bool = False
    status: str = "confirmed"   # confirmed / tentative / cancelled


@dataclass
class CalendarInfo:
    """Metadata for a single calendar / collection."""
    id: str
    name: str
    primary: bool = False
    read_only: bool = False


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CalendarProvider(Protocol):
    async def list_calendars(self) -> list[CalendarInfo]: ...
    async def list_events(
        self, start: datetime, end: datetime, calendar_id: Optional[str] = None
    ) -> list[CalendarEvent]: ...
    async def get_event(
        self, event_id: str, calendar_id: Optional[str] = None
    ) -> Optional[CalendarEvent]: ...
    async def create_event(
        self, event: CalendarEvent, calendar_id: Optional[str] = None
    ) -> CalendarEvent: ...
    async def update_event(
        self, event: CalendarEvent, calendar_id: Optional[str] = None
    ) -> CalendarEvent: ...
    async def delete_event(
        self, event_id: str, calendar_id: Optional[str] = None
    ) -> bool: ...


# ---------------------------------------------------------------------------
# Google Calendar Provider
# ---------------------------------------------------------------------------

_GCAL_BASE = "https://www.googleapis.com/calendar/v3"
_GCAL_TOKEN_URL = "https://oauth2.googleapis.com/token"


class GoogleCalendarProvider:
    """Google Calendar v3, reusing the existing google_oauth token-store."""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expires: float = 0.0
        self._refresh_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # HTTP / auth helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                limits=httpx.Limits(max_keepalive_connections=5),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _refresh_token(self, force: bool = False) -> str:
        import time

        async with self._refresh_lock:
            if not force and self._access_token and time.time() < self._token_expires - 60:
                return self._access_token  # type: ignore[return-value]

            from .google_oauth import get_google_token_store

            store = get_google_token_store()
            creds = store.get_credentials("calendar")
            if not creds:
                raise RuntimeError(
                    "Google Calendar OAuth not configured. "
                    "Run: python scripts/setup_google_oauth.py"
                )

            client = await self._ensure_client()
            resp = await client.post(
                _GCAL_TOKEN_URL,
                data={
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                    "refresh_token": creds.refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            if resp.status_code in (400, 401):
                raise RuntimeError(
                    f"Google Calendar refresh token rejected (HTTP {resp.status_code}). "
                    "Re-run: python scripts/setup_google_oauth.py"
                )
            resp.raise_for_status()
            token_data = resp.json()

            self._access_token = token_data["access_token"]
            self._token_expires = time.time() + token_data.get("expires_in", 3600)

            new_refresh = token_data.get("refresh_token")
            if new_refresh and new_refresh != creds.refresh_token:
                store.persist_refresh_token("calendar", new_refresh)

            return self._access_token  # type: ignore[return-value]

    async def _auth_headers(self, force_refresh: bool = False) -> dict[str, str]:
        import time

        if force_refresh or not self._access_token or time.time() >= self._token_expires - 60:
            await self._refresh_token(force=force_refresh)
        return {"Authorization": f"Bearer {self._access_token}"}

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_event(
        self, item: dict[str, Any], calendar_id: str = ""
    ) -> Optional[CalendarEvent]:
        try:
            start_data = item.get("start", {})
            end_data = item.get("end", {})
            if "dateTime" in start_data:
                start = datetime.fromisoformat(start_data["dateTime"])
                end = datetime.fromisoformat(end_data["dateTime"])
                all_day = False
            elif "date" in start_data:
                start = datetime.fromisoformat(start_data["date"] + "T00:00:00+00:00")
                end = datetime.fromisoformat(end_data["date"] + "T23:59:59+00:00")
                all_day = True
            else:
                return None
            return CalendarEvent(
                uid=item.get("id", ""),
                summary=item.get("summary", "Untitled"),
                start=start,
                end=end,
                calendar_id=calendar_id,
                location=item.get("location"),
                description=item.get("description"),
                all_day=all_day,
                status=item.get("status", "confirmed"),
            )
        except Exception as exc:
            logger.warning("Failed to parse Google Calendar event: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    async def list_calendars(self) -> list[CalendarInfo]:
        client = await self._ensure_client()
        headers = await self._auth_headers()
        resp = await client.get(f"{_GCAL_BASE}/users/me/calendarList", headers=headers)
        resp.raise_for_status()
        result: list[CalendarInfo] = []
        for item in resp.json().get("items", []):
            cal_id = item.get("id", "")
            if "holiday@group" in cal_id:
                continue
            result.append(
                CalendarInfo(
                    id=cal_id,
                    name=item.get("summary", ""),
                    primary=item.get("primary", False),
                    read_only=item.get("accessRole", "") in ("reader", "freeBusyReader"),
                )
            )
        return result

    async def list_events(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> list[CalendarEvent]:
        client = await self._ensure_client()
        cal_id = calendar_id or "primary"
        params = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "maxResults": 50,
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        headers = await self._auth_headers()
        resp = await client.get(f"{_GCAL_BASE}/calendars/{cal_id}/events", headers=headers, params=params)
        if resp.status_code == 401:
            headers = await self._auth_headers(force_refresh=True)
            resp = await client.get(f"{_GCAL_BASE}/calendars/{cal_id}/events", headers=headers, params=params)
        resp.raise_for_status()
        events = [self._parse_event(item, calendar_id=cal_id) for item in resp.json().get("items", [])]
        return [e for e in events if e is not None]

    async def get_event(
        self, event_id: str, calendar_id: Optional[str] = None
    ) -> Optional[CalendarEvent]:
        client = await self._ensure_client()
        cal_id = calendar_id or "primary"
        headers = await self._auth_headers()
        resp = await client.get(f"{_GCAL_BASE}/calendars/{cal_id}/events/{event_id}", headers=headers)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return self._parse_event(resp.json(), calendar_id=cal_id)

    async def create_event(
        self, event: CalendarEvent, calendar_id: Optional[str] = None
    ) -> CalendarEvent:
        client = await self._ensure_client()
        cal_id = calendar_id or "primary"
        body: dict[str, Any] = {
            "summary": event.summary,
            "start": (
                {"date": event.start.date().isoformat()} if event.all_day
                else {"dateTime": event.start.isoformat()}
            ),
            "end": (
                {"date": event.end.date().isoformat()} if event.all_day
                else {"dateTime": event.end.isoformat()}
            ),
        }
        if event.location:
            body["location"] = event.location
        if event.description:
            body["description"] = event.description

        headers = await self._auth_headers()
        headers["Content-Type"] = "application/json"
        resp = await client.post(f"{_GCAL_BASE}/calendars/{cal_id}/events", headers=headers, json=body)
        if resp.status_code == 401:
            headers = await self._auth_headers(force_refresh=True)
            headers["Content-Type"] = "application/json"
            resp = await client.post(f"{_GCAL_BASE}/calendars/{cal_id}/events", headers=headers, json=body)
        resp.raise_for_status()
        return self._parse_event(resp.json(), calendar_id=cal_id) or event

    async def update_event(
        self, event: CalendarEvent, calendar_id: Optional[str] = None
    ) -> CalendarEvent:
        client = await self._ensure_client()
        cal_id = calendar_id or "primary"
        body: dict[str, Any] = {
            "summary": event.summary,
            "start": (
                {"date": event.start.date().isoformat()} if event.all_day
                else {"dateTime": event.start.isoformat()}
            ),
            "end": (
                {"date": event.end.date().isoformat()} if event.all_day
                else {"dateTime": event.end.isoformat()}
            ),
        }
        if event.location:
            body["location"] = event.location
        if event.description:
            body["description"] = event.description

        headers = await self._auth_headers()
        headers["Content-Type"] = "application/json"
        resp = await client.put(
            f"{_GCAL_BASE}/calendars/{cal_id}/events/{event.uid}",
            headers=headers,
            json=body,
        )
        if resp.status_code == 401:
            headers = await self._auth_headers(force_refresh=True)
            headers["Content-Type"] = "application/json"
            resp = await client.put(
                f"{_GCAL_BASE}/calendars/{cal_id}/events/{event.uid}",
                headers=headers,
                json=body,
            )
        resp.raise_for_status()
        return self._parse_event(resp.json(), calendar_id=cal_id) or event

    async def delete_event(
        self, event_id: str, calendar_id: Optional[str] = None
    ) -> bool:
        client = await self._ensure_client()
        cal_id = calendar_id or "primary"
        headers = await self._auth_headers()
        resp = await client.delete(f"{_GCAL_BASE}/calendars/{cal_id}/events/{event_id}", headers=headers)
        return resp.status_code in (200, 204, 410)


# ---------------------------------------------------------------------------
# iCalendar helpers (minimal, stdlib-only, used by CalDAV provider)
# ---------------------------------------------------------------------------

_ICAL_DT_FMT = "%Y%m%dT%H%M%SZ"
_ICAL_D_FMT = "%Y%m%d"


def _ical_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


_UNESCAPE_RE = re.compile(r"\\(.)")


def _ical_unescape(s: str) -> str:
    """Single-pass iCal property value unescaping (avoids double-unescaping \\n)."""

    def _repl(m: re.Match) -> str:
        ch = m.group(1)
        if ch == "n":
            return "\n"
        if ch in ("\\", ",", ";"):
            return ch
        return m.group(0)  # unknown escape — leave as-is

    return _UNESCAPE_RE.sub(_repl, s)


def _ical_parse_dt(value: str) -> datetime:
    """Parse an iCal DTSTART/DTEND property value into a datetime."""
    # Strip property parameters (e.g. TZID=America/Chicago)
    value = value.split(":")[-1].strip()
    if value.endswith("Z"):
        return datetime.strptime(value, _ICAL_DT_FMT).replace(tzinfo=timezone.utc)
    if "T" in value:
        return datetime.strptime(value, "%Y%m%dT%H%M%S")
    return datetime.strptime(value, _ICAL_D_FMT)


def _parse_ical_events(ical_text: str, calendar_id: str = "") -> list[CalendarEvent]:
    """Extract VEVENT blocks from raw iCalendar text."""
    events: list[CalendarEvent] = []
    for vevent_block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", ical_text, re.DOTALL):
        props: dict[str, str] = {}
        for line in vevent_block.strip().splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            # Store under the base property name (without parameters like ;TZID=)
            base_key = key.split(";")[0].upper()
            # For DTSTART/DTEND preserve the full "key:value" so _ical_parse_dt
            # can handle TZID parameters embedded in the value field
            props[base_key] = f"{key}:{val}".partition(":")[2].strip() if ";" in key else val.strip()

        uid = props.get("UID", str(uuid.uuid4()))
        summary = _ical_unescape(props.get("SUMMARY", "Untitled"))
        location = _ical_unescape(props.get("LOCATION", "")) or None
        description = _ical_unescape(props.get("DESCRIPTION", "")) or None
        status = props.get("STATUS", "CONFIRMED").lower()

        if status == "cancelled":
            continue

        try:
            start_raw = props.get("DTSTART", "")
            end_raw = props.get("DTEND", props.get("DUE", ""))
            all_day = "T" not in start_raw
            start = _ical_parse_dt(start_raw)
            end = _ical_parse_dt(end_raw) if end_raw else start + timedelta(hours=1)
        except Exception as exc:
            logger.debug("Skipping unparseable VEVENT uid=%s: %s", uid, exc)
            continue

        events.append(
            CalendarEvent(
                uid=uid,
                summary=summary,
                start=start,
                end=end,
                calendar_id=calendar_id,
                location=location,
                description=description,
                all_day=all_day,
                status=status,
            )
        )
    return events


def _build_ical_event(event: CalendarEvent) -> str:
    """Render a CalendarEvent as an iCalendar VCALENDAR string."""
    now_str = datetime.now(timezone.utc).strftime(_ICAL_DT_FMT)
    if event.all_day:
        start_line = f"DTSTART;VALUE=DATE:{event.start.strftime(_ICAL_D_FMT)}"
        end_line = f"DTEND;VALUE=DATE:{event.end.strftime(_ICAL_D_FMT)}"
    else:
        s = event.start.astimezone(timezone.utc)
        e = event.end.astimezone(timezone.utc)
        start_line = f"DTSTART:{s.strftime(_ICAL_DT_FMT)}"
        end_line = f"DTEND:{e.strftime(_ICAL_DT_FMT)}"

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Atlas Brain//Atlas Calendar//EN",
        "BEGIN:VEVENT",
        f"UID:{event.uid}",
        f"DTSTAMP:{now_str}",
        f"CREATED:{now_str}",
        f"LAST-MODIFIED:{now_str}",
        start_line,
        end_line,
        f"SUMMARY:{_ical_escape(event.summary)}",
        f"STATUS:{event.status.upper()}",
    ]
    if event.location:
        lines.append(f"LOCATION:{_ical_escape(event.location)}")
    if event.description:
        lines.append(f"DESCRIPTION:{_ical_escape(event.description)}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines)


# ---------------------------------------------------------------------------
# CalDAV Provider (httpx + stdlib iCal, zero new deps)
# ---------------------------------------------------------------------------

_CALDAV_REPORT_TMPL = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
    "  <d:prop><d:getetag/><c:calendar-data/></d:prop>"
    "  <c:filter>"
    '    <c:comp-filter name="VCALENDAR">'
    '      <c:comp-filter name="VEVENT">'
    '        <c:time-range start="{start}" end="{end}"/>'
    "      </c:comp-filter>"
    "    </c:comp-filter>"
    "  </c:filter>"
    "</c:calendar-query>"
)

_CALDAV_PROPFIND_HOME = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
    "  <d:prop><c:calendar-home-set/><d:displayname/></d:prop>"
    "</d:propfind>"
)

_CALDAV_PROPFIND_CALS = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
    "  <d:prop>"
    "    <d:displayname/>"
    "    <d:resourcetype/>"
    "    <c:supported-calendar-component-set/>"
    "  </d:prop>"
    "</d:propfind>"
)

_XML_HEADERS = {"Content-Type": "application/xml; charset=utf-8"}


class CalDAVCalendarProvider:
    """CalDAV / RFC 4791 calendar provider using httpx (no extra dependencies).

    Compatible with: Nextcloud, Apple Calendar, Fastmail, Proton Calendar,
    SOGo, Baikal, Radicale, and any RFC 4791-compliant CalDAV server.
    """

    def __init__(self) -> None:
        cfg = settings.tools
        self._base_url = (cfg.caldav_url or "").rstrip("/")
        self._username = cfg.caldav_username or ""
        self._password = cfg.caldav_password or ""
        if self._base_url and not self._username:
            logger.warning("CalDAV URL set but ATLAS_TOOLS_CALDAV_USERNAME is empty")
        if self._base_url and not self._password:
            logger.warning("CalDAV URL set but ATLAS_TOOLS_CALDAV_PASSWORD is empty")
        self._calendar_url: Optional[str] = cfg.caldav_calendar_url or None
        self._client: Optional[httpx.AsyncClient] = None

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _client_(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                auth=(self._username, self._password),
                timeout=15.0,
                headers=_XML_HEADERS,
                follow_redirects=True,
            )
        return self._client

    async def _discover_calendar_url(self) -> str:
        """Discover the default calendar collection URL via PROPFIND."""
        if self._calendar_url:
            return self._calendar_url

        client = self._client_()
        well_known = f"{self._base_url}/.well-known/caldav"

        resp = await client.request(
            "PROPFIND",
            well_known,
            content=_CALDAV_PROPFIND_HOME,
            headers={**_XML_HEADERS, "Depth": "0"},
        )

        if resp.status_code == 207:
            # Try to extract calendar-home-set href
            m = re.search(
                r"<[^>]*:?href[^>]*>\s*([^<]*calendar[^<]*)\s*</",
                resp.text,
                re.IGNORECASE,
            )
            if m:
                path = m.group(1).strip()
                self._calendar_url = (
                    path if path.startswith("http")
                    else f"{self._base_url}/{path.lstrip('/')}"
                )

        if not self._calendar_url:
            # Fallback: conventional Nextcloud / DAViCal path
            self._calendar_url = f"{self._base_url}/calendars/{self._username}/"

        return self._calendar_url

    async def list_calendars(self) -> list[CalendarInfo]:
        client = self._client_()
        cal_url = await self._discover_calendar_url()

        resp = await client.request(
            "PROPFIND",
            cal_url,
            content=_CALDAV_PROPFIND_CALS,
            headers={**_XML_HEADERS, "Depth": "1"},
        )
        if resp.status_code not in (200, 207):
            raise RuntimeError(f"CalDAV PROPFIND failed: HTTP {resp.status_code}")

        calendars: list[CalendarInfo] = []
        for m in re.finditer(
            r"<d:href>([^<]+)</d:href>.*?<d:displayname>([^<]*)</d:displayname>",
            resp.text,
            re.DOTALL,
        ):
            href, name = m.group(1).strip(), m.group(2).strip()
            if not name or href.rstrip("/") == cal_url.rstrip("/"):
                continue
            cal_id = href if href.startswith("http") else f"{self._base_url}{href}"
            calendars.append(CalendarInfo(id=cal_id, name=name))

        if not calendars:
            # Server returned a single calendar at this URL
            name_m = re.search(r"<d:displayname>([^<]+)</d:displayname>", resp.text)
            calendars.append(
                CalendarInfo(
                    id=cal_url,
                    name=name_m.group(1) if name_m else "Calendar",
                    primary=True,
                )
            )
        return calendars

    async def list_events(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> list[CalendarEvent]:
        client = self._client_()
        cal_url = calendar_id or await self._discover_calendar_url()

        start_str = start.astimezone(timezone.utc).strftime(_ICAL_DT_FMT)
        end_str = end.astimezone(timezone.utc).strftime(_ICAL_DT_FMT)
        query = _CALDAV_REPORT_TMPL.format(start=start_str, end=end_str)

        resp = await client.request(
            "REPORT",
            cal_url,
            content=query,
            headers={**_XML_HEADERS, "Depth": "1"},
        )
        if resp.status_code not in (200, 207):
            raise RuntimeError(f"CalDAV REPORT failed: HTTP {resp.status_code}")

        events = _parse_ical_events(resp.text, calendar_id=cal_url)
        events.sort(key=lambda e: e.start)
        return events

    async def get_event(
        self, event_id: str, calendar_id: Optional[str] = None
    ) -> Optional[CalendarEvent]:
        client = self._client_()
        cal_url = calendar_id or await self._discover_calendar_url()
        url = event_id if event_id.startswith("http") else f"{cal_url.rstrip('/')}/{event_id}.ics"

        resp = await client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        events = _parse_ical_events(resp.text, calendar_id=cal_url)
        return events[0] if events else None

    async def create_event(
        self, event: CalendarEvent, calendar_id: Optional[str] = None
    ) -> CalendarEvent:
        client = self._client_()
        cal_url = calendar_id or await self._discover_calendar_url()
        if not event.uid:
            event.uid = str(uuid.uuid4())

        url = f"{cal_url.rstrip('/')}/{event.uid}.ics"
        resp = await client.put(
            url,
            content=_build_ical_event(event),
            headers={"Content-Type": "text/calendar; charset=utf-8"},
        )
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(f"CalDAV PUT failed: HTTP {resp.status_code}")

        event.calendar_id = cal_url
        return event

    async def update_event(
        self, event: CalendarEvent, calendar_id: Optional[str] = None
    ) -> CalendarEvent:
        # CalDAV PUT with existing UID acts as an upsert
        return await self.create_event(event, calendar_id=calendar_id)

    async def delete_event(
        self, event_id: str, calendar_id: Optional[str] = None
    ) -> bool:
        client = self._client_()
        cal_url = calendar_id or await self._discover_calendar_url()
        url = event_id if event_id.startswith("http") else f"{cal_url.rstrip('/')}/{event_id}.ics"

        resp = await client.delete(url)
        return resp.status_code in (200, 204, 404)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider_instance: Optional[CalendarProvider] = None


def get_calendar_provider() -> CalendarProvider:
    """Return the active CalendarProvider based on environment config.

    Priority:
        1. CalDAV  — when ATLAS_TOOLS_CALDAV_URL is set
        2. Google  — when ATLAS_TOOLS_CALENDAR_ENABLED=true
    """
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    cfg = settings.tools
    if cfg.caldav_url:
        logger.info("Calendar provider: CalDAV (%s)", cfg.caldav_url)
        _provider_instance = CalDAVCalendarProvider()
    elif cfg.calendar_enabled:
        logger.info("Calendar provider: Google Calendar")
        _provider_instance = GoogleCalendarProvider()
    else:
        raise RuntimeError(
            "No calendar provider configured. "
            "Set ATLAS_TOOLS_CALDAV_URL or ATLAS_TOOLS_CALENDAR_ENABLED=true."
        )
    return _provider_instance
