"""
Google Calendar tool with aggressive caching for lowest latency.

Uses Google Calendar API with OAuth2 or service account credentials.
Caches events in memory with configurable TTL.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from ..config import settings
from ..services.google_oauth import get_google_token_store
from .base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.calendar")


class CalendarAuthError(Exception):
    """Raised when Google OAuth refresh token is invalid or revoked."""


# Google Calendar API endpoints
CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"


@dataclass
class CachedEvent:
    """Cached calendar event."""
    id: str
    summary: str
    start: datetime
    end: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    all_day: bool = False


@dataclass
class CalendarInfo:
    """Cached calendar metadata."""
    id: str
    name: str
    primary: bool = False


@dataclass
class EventCache:
    """In-memory event cache with TTL."""
    events: list[CachedEvent] = field(default_factory=list)
    last_updated: float = 0.0
    ttl_seconds: float = 300.0  # 5 minutes default

    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        return time.time() - self.last_updated < self.ttl_seconds

    def update(self, events: list[CachedEvent]) -> None:
        """Update cache with new events."""
        self.events = events
        self.last_updated = time.time()

    def get_upcoming(self, hours: int = 24) -> list[CachedEvent]:
        """Get events in the next N hours from cache."""
        now = datetime.now().astimezone()
        cutoff = now + timedelta(hours=hours)
        return [
            e for e in self.events
            if e.start >= now and e.start <= cutoff
        ]


@dataclass
class CalendarListCache:
    """Cache for calendar list."""
    calendars: list[CalendarInfo] = field(default_factory=list)
    last_updated: float = 0.0
    ttl_seconds: float = 3600.0  # 1 hour

    def is_valid(self) -> bool:
        return time.time() - self.last_updated < self.ttl_seconds

    def update(self, calendars: list[CalendarInfo]) -> None:
        self.calendars = calendars
        self.last_updated = time.time()


class CalendarTool:
    """Google Calendar tool with caching for low latency."""

    def __init__(self) -> None:
        self._config = settings.tools
        self._client: Optional[httpx.AsyncClient] = None
        self._cache = EventCache(ttl_seconds=self._config.calendar_cache_ttl)
        self._calendar_list_cache = CalendarListCache()
        self._access_token: Optional[str] = None
        self._token_expires: float = 0.0
        self._refresh_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "get_calendar"

    @property
    def description(self) -> str:
        return "Get upcoming calendar events and appointments"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="hours_ahead",
                param_type="int",
                description="Hours to look ahead (default: 24)",
                required=False,
                default=24,
            ),
            ToolParameter(
                name="max_results",
                param_type="int",
                description="Maximum events to return (default: 10)",
                required=False,
                default=10,
            ),
            ToolParameter(
                name="calendar_name",
                param_type="string",
                description="Specific calendar name to query (e.g., 'family', 'work')",
                required=False,
                default=None,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["calendar", "events", "schedule", "appointments", "agenda"]

    @property
    def category(self) -> str:
        return "utility"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _refresh_token(self, force: bool = False) -> str:
        """Refresh OAuth2 access token using refresh token."""
        async with self._refresh_lock:
            # Double-check after acquiring lock (skip if forced)
            if not force and self._access_token and time.time() < self._token_expires - 60:
                return self._access_token

            # Load credentials from token store (file first, .env fallback)
            store = get_google_token_store()
            creds = store.get_credentials("calendar")
            if not creds:
                raise CalendarAuthError(
                    "Calendar OAuth not configured. "
                    "Run: python scripts/setup_google_oauth.py"
                )

            client = await self._ensure_client()

            data = {
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "refresh_token": creds.refresh_token,
                "grant_type": "refresh_token",
            }

            response = await client.post(TOKEN_URL, data=data)
            if response.status_code in (400, 401):
                logger.critical(
                    "Calendar refresh token is INVALID (HTTP %d). "
                    "Re-run: python scripts/setup_google_oauth.py",
                    response.status_code,
                )
                raise CalendarAuthError(
                    f"Refresh token rejected (HTTP {response.status_code}). "
                    "Token may be expired or revoked."
                )
            response.raise_for_status()
            token_data = response.json()

            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires = time.time() + expires_in

            # Auto-persist rotated refresh token
            new_refresh = token_data.get("refresh_token")
            if new_refresh and new_refresh != creds.refresh_token:
                store.persist_refresh_token("calendar", new_refresh)

            logger.debug("Refreshed Google Calendar access token")
            return self._access_token

    async def _get_auth_header(self, force_refresh: bool = False) -> dict[str, str]:
        """Get authorization header, refreshing token if needed."""
        if force_refresh or not self._access_token or time.time() >= self._token_expires - 60:
            await self._refresh_token(force=force_refresh)
        return {"Authorization": f"Bearer {self._access_token}"}

    def _invalidate_access_token(self) -> None:
        """Mark current access token as expired so next call refreshes."""
        self._access_token = None
        self._token_expires = 0.0

    async def _get_calendar_list(self) -> list[CalendarInfo]:
        """Get list of calendars, using cache if valid."""
        if self._calendar_list_cache.is_valid():
            return self._calendar_list_cache.calendars

        client = await self._ensure_client()
        headers = await self._get_auth_header()

        url = f"{CALENDAR_API_BASE}/users/me/calendarList"
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        calendars = []
        for item in response.json().get("items", []):
            cal_id = item.get("id", "")
            # Skip holiday calendars
            if "holiday@group" in cal_id:
                continue
            calendars.append(CalendarInfo(
                id=cal_id,
                name=item.get("summary", ""),
                primary=item.get("primary", False),
            ))

        self._calendar_list_cache.update(calendars)
        logger.debug("Cached %d calendars", len(calendars))
        return calendars

    def _match_calendar(self, query: str, calendars: list[CalendarInfo]) -> Optional[CalendarInfo]:
        """Match a calendar name from query text."""
        query_lower = query.lower()

        # Common words to ignore when matching
        ignore_words = {
            "calendar", "schedule", "appointment", "appointments",
            "event", "events", "meeting", "meetings", "the", "my",
        }

        # First pass: direct full name match
        for cal in calendars:
            cal_name_lower = cal.name.lower()
            if cal_name_lower in query_lower:
                return cal

        # Second pass: significant word match
        for cal in calendars:
            cal_name_lower = cal.name.lower()
            cal_words = cal_name_lower.replace("-", " ").split()

            for word in cal_words:
                # Skip short words and common words
                if len(word) <= 3 or word in ignore_words:
                    continue
                if word in query_lower:
                    return cal

        return None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute calendar query with caching."""
        if not self._config.calendar_enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Calendar tool is disabled",
            )

        if not self._config.calendar_refresh_token:
            return ToolResult(
                success=False,
                error="NOT_CONFIGURED",
                message="Calendar not configured. Run calendar setup first.",
            )

        hours_ahead = params.get("hours_ahead", 24)
        max_results = min(params.get("max_results", 10), 25)
        calendar_name = params.get("calendar_name")

        try:
            # Get calendar list and try to match specific calendar
            calendars = await self._get_calendar_list()
            matched_calendar = None

            if calendar_name:
                matched_calendar = self._match_calendar(calendar_name, calendars)
                if matched_calendar:
                    logger.info("Matched calendar: %s", matched_calendar.name)

            # Fetch events (filtered by calendar if matched)
            events = await self._fetch_events(
                hours_ahead, max_results, target_calendar=matched_calendar
            )

            return ToolResult(
                success=True,
                data={
                    "events": [self._event_to_dict(e) for e in events],
                    "count": len(events),
                    "cached": self._cache.is_valid(),
                },
                message=self._format_message(events, hours_ahead),
            )

        except CalendarAuthError as e:
            return ToolResult(
                success=False,
                error="AUTH_ERROR",
                message="Calendar authentication failed. Refresh token needs renewal.",
            )
        except httpx.HTTPStatusError as e:
            logger.error("Calendar API HTTP error: %s", e)
            # Try to use stale cache on error
            if self._cache.events:
                events = self._cache.get_upcoming(hours_ahead)[:max_results]
                return ToolResult(
                    success=True,
                    data={
                        "events": [self._event_to_dict(e) for e in events],
                        "count": len(events),
                        "cached": True,
                        "stale": True,
                    },
                    message=self._format_message(events, hours_ahead) + " (cached)",
                )
            return ToolResult(
                success=False,
                error="API_ERROR",
                message=f"Calendar API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.exception("Calendar tool error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    async def _fetch_events(
        self,
        hours_ahead: int,
        max_results: int,
        target_calendar: Optional[CalendarInfo] = None,
    ) -> list[CachedEvent]:
        """Fetch events from Google Calendars."""
        client = await self._ensure_client()
        headers = await self._get_auth_header()

        now = datetime.now().astimezone()
        time_max = now + timedelta(days=7)

        # Determine which calendars to query
        if target_calendar:
            calendars_to_query = [target_calendar]
        else:
            calendars_to_query = await self._get_calendar_list()

        all_events = []

        for cal in calendars_to_query:
            try:
                query_params = {
                    "timeMin": now.isoformat(),
                    "timeMax": time_max.isoformat(),
                    "maxResults": 25,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                }

                url = f"{CALENDAR_API_BASE}/calendars/{cal.id}/events"
                response = await client.get(url, headers=headers, params=query_params)

                # Retry once on 401 with fresh token
                if response.status_code == 401:
                    logger.warning("Calendar API 401 -- forcing token refresh")
                    self._invalidate_access_token()
                    headers = await self._get_auth_header(force_refresh=True)
                    response = await client.get(url, headers=headers, params=query_params)

                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("items", []):
                        event = self._parse_event(item, calendar_name=cal.name)
                        if event:
                            all_events.append(event)
            except CalendarAuthError:
                raise  # Propagate auth errors
            except Exception as e:
                logger.debug("Failed to fetch calendar %s: %s", cal.id, e)

        # Sort all events by start time
        all_events.sort(key=lambda e: e.start)

        # Only update full cache when querying all calendars
        if not target_calendar:
            self._cache.update(all_events)

        cal_count = len(calendars_to_query)
        logger.info("Fetched %d events from %d calendar(s)", len(all_events), cal_count)

        # Return only requested window
        cutoff = now + timedelta(hours=hours_ahead)
        return [e for e in all_events if e.start <= cutoff][:max_results]

    def _parse_event(self, item: dict[str, Any], calendar_name: str = "") -> Optional[CachedEvent]:
        """Parse API response item into CachedEvent."""
        try:
            start_data = item.get("start", {})
            end_data = item.get("end", {})

            # Handle all-day vs timed events
            if "dateTime" in start_data:
                start = datetime.fromisoformat(start_data["dateTime"])
                end = datetime.fromisoformat(end_data["dateTime"])
                all_day = False
            elif "date" in start_data:
                start = datetime.fromisoformat(start_data["date"]).replace(
                    hour=0, minute=0, tzinfo=datetime.now().astimezone().tzinfo
                )
                end = datetime.fromisoformat(end_data["date"]).replace(
                    hour=23, minute=59, tzinfo=datetime.now().astimezone().tzinfo
                )
                all_day = True
            else:
                return None

            return CachedEvent(
                id=item.get("id", ""),
                summary=item.get("summary", "Untitled"),
                start=start,
                end=end,
                location=item.get("location"),
                description=item.get("description"),
                all_day=all_day,
            )
        except Exception as e:
            logger.warning("Failed to parse event: %s", e)
            return None

    def _event_to_dict(self, event: CachedEvent) -> dict[str, Any]:
        """Convert CachedEvent to dict for response."""
        return {
            "id": event.id,
            "summary": event.summary,
            "start": event.start.isoformat(),
            "end": event.end.isoformat(),
            "location": event.location,
            "all_day": event.all_day,
        }

    def _format_message(self, events: list[CachedEvent], hours: int) -> str:
        """Format events as human-readable message."""
        if not events:
            return f"No events in the next {hours} hours."

        if len(events) == 1:
            e = events[0]
            time_str = self._format_event_time(e)
            return f"You have {e.summary} {time_str}."

        # Multiple events
        lines = [f"You have {len(events)} events coming up:"]
        for e in events[:5]:  # Limit spoken list
            time_str = self._format_event_time(e)
            lines.append(f"{e.summary} {time_str}")

        if len(events) > 5:
            lines.append(f"...and {len(events) - 5} more.")

        return " ".join(lines)

    def _format_event_time(self, event: CachedEvent) -> str:
        """Format event time for speech."""
        now = datetime.now().astimezone()
        start = event.start

        if event.all_day:
            if start.date() == now.date():
                return "today, all day"
            elif start.date() == (now + timedelta(days=1)).date():
                return "tomorrow, all day"
            else:
                return f"on {start.strftime('%A')}, all day"

        # Timed event
        if start.date() == now.date():
            return f"today at {start.strftime('%-I:%M %p')}"
        elif start.date() == (now + timedelta(days=1)).date():
            return f"tomorrow at {start.strftime('%-I:%M %p')}"
        else:
            return f"on {start.strftime('%A')} at {start.strftime('%-I:%M %p')}"

    async def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        location: Optional[str] = None,
        description: Optional[str] = None,
        calendar_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Create a new calendar event.

        Args:
            summary: Event title
            start: Event start datetime (timezone-aware)
            end: Event end datetime (timezone-aware)
            location: Optional location string
            description: Optional description
            calendar_id: Calendar to create in (default: primary)

        Returns:
            ToolResult with created event details
        """
        if not self._config.calendar_enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Calendar tool is disabled",
            )

        if not self._config.calendar_refresh_token:
            return ToolResult(
                success=False,
                error="NOT_CONFIGURED",
                message="Calendar not configured. Run calendar setup first.",
            )

        try:
            client = await self._ensure_client()
            headers = await self._get_auth_header()
            headers["Content-Type"] = "application/json"

            cal_id = calendar_id or "primary"

            event_body = {
                "summary": summary,
                "start": {"dateTime": start.isoformat()},
                "end": {"dateTime": end.isoformat()},
            }

            if location:
                event_body["location"] = location
            if description:
                event_body["description"] = description

            url = f"{CALENDAR_API_BASE}/calendars/{cal_id}/events"
            response = await client.post(url, headers=headers, json=event_body)

            # Retry once on 401 with fresh token
            if response.status_code == 401:
                logger.warning("Calendar create 401 -- forcing token refresh")
                self._invalidate_access_token()
                headers = await self._get_auth_header(force_refresh=True)
                headers["Content-Type"] = "application/json"
                response = await client.post(url, headers=headers, json=event_body)

            response.raise_for_status()

            created = response.json()
            event_id = created.get("id", "")

            # Invalidate cache since we added an event
            self._cache.last_updated = 0.0

            logger.info("Created calendar event: %s", event_id)

            return ToolResult(
                success=True,
                data={
                    "event_id": event_id,
                    "summary": summary,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "location": location,
                },
                message=f"Created event: {summary}",
            )

        except CalendarAuthError as e:
            return ToolResult(
                success=False,
                error="AUTH_ERROR",
                message="Calendar authentication failed. Refresh token needs renewal.",
            )
        except httpx.HTTPStatusError as e:
            logger.error("Calendar create event HTTP error: %s", e)
            return ToolResult(
                success=False,
                error="API_ERROR",
                message=f"Calendar API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.exception("Calendar create event error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    async def verify_credentials(self) -> bool:
        """Verify refresh token is valid. Call on startup."""
        if not self._config.calendar_enabled:
            return True
        if not self._config.calendar_refresh_token:
            logger.warning("Calendar enabled but no refresh token configured")
            return False

        try:
            await self._refresh_token(force=True)
            logger.info("Calendar OAuth credentials verified")
            return True
        except CalendarAuthError:
            # Already logged as CRITICAL in _refresh_token
            return False
        except Exception as e:
            logger.error("Calendar credential check failed: %s", e)
            return False

    async def prefetch(self) -> None:
        """Pre-fetch events to warm the cache. Call on startup."""
        if not self._config.calendar_enabled:
            return
        if not self._config.calendar_refresh_token:
            return

        try:
            await self._fetch_events(hours_ahead=168, max_results=50)  # 7 days
            logger.info("Calendar cache warmed")
        except CalendarAuthError:
            pass  # Already logged as CRITICAL
        except Exception as e:
            logger.warning("Calendar prefetch failed: %s", e)


class CreateCalendarEventTool:
    """Tool for creating calendar events via natural language."""

    @property
    def name(self) -> str:
        return "create_calendar_event"

    @property
    def description(self) -> str:
        return "Create a new calendar event. Use when user says 'add a meeting', 'schedule an event', 'put on my calendar'."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="summary",
                param_type="string",
                description="Event title (e.g., 'Team meeting', 'Lunch with Sarah')",
                required=True,
            ),
            ToolParameter(
                name="start_time",
                param_type="string",
                description="When the event starts (e.g., 'tomorrow at 3pm', 'next Tuesday 10am')",
                required=True,
            ),
            ToolParameter(
                name="duration_minutes",
                param_type="int",
                description="Event duration in minutes (default: 60)",
                required=False,
                default=60,
            ),
            ToolParameter(
                name="location",
                param_type="string",
                description="Event location",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["create event", "add event", "new event", "schedule event"]

    @property
    def category(self) -> str:
        return "utility"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Create a calendar event from natural language parameters."""
        import dateparser as dp

        summary = params.get("summary", "").strip()
        start_text = params.get("start_time", "").strip()
        duration = int(params.get("duration_minutes", 60))
        location = params.get("location")

        if not summary:
            return ToolResult(
                success=False, error="MISSING_SUMMARY",
                message="What should the event be called?",
            )
        if not start_text:
            return ToolResult(
                success=False, error="MISSING_TIME",
                message="When should the event be?",
            )

        start = dp.parse(start_text, settings={
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": settings.reminder.default_timezone,
        })
        if not start:
            return ToolResult(
                success=False, error="INVALID_TIME",
                message=f"I couldn't understand '{start_text}'. Try 'tomorrow at 3pm'.",
            )

        end = start + timedelta(minutes=duration)
        return await calendar_tool.create_event(
            summary=summary, start=start, end=end, location=location,
        )


# Module-level instances
calendar_tool = CalendarTool()
create_calendar_event_tool = CreateCalendarEventTool()
