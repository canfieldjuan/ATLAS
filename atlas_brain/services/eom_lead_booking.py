"""Durable, idempotent estimate bookings for active EOM leads."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from ..eom_api.config import EOMFunnelConfig, funnel_settings
from ..storage.database import get_db_pool
from ..tools.base import ToolResult
from ..tools.calendar import CalendarTool, calendar_tool
from .crm_provider import _transaction_connection
from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

_MAX_ERROR_LENGTH = 1000
_PROJECTION_LEASE = timedelta(minutes=2)
_TERMINAL_CALENDAR_FAILURE_STATUS = "calendar_rejected"
_PERMANENT_CALENDAR_RESULT_ERRORS = {"AUTH_ERROR", "NOT_CONFIGURED", "TOOL_DISABLED"}
_PERMANENT_CALENDAR_HTTP_STATUSES = {400, 401, 403, 404, 410}

logger = logging.getLogger("atlas.services.eom_lead_booking")


class EOMLeadBookingError(ValueError):
    """Base error for a caller-correctable booking rejection."""

    status_code = 400


class EOMLeadBookingNotFoundError(EOMLeadBookingError):
    """The contact is not an active EOM lead eligible for booking."""

    status_code = 404


class EOMLeadBookingConflictError(EOMLeadBookingError):
    """The command conflicts with existing lead or operation state."""

    status_code = 409


class EOMLeadBookingProjectionError(EOMLeadBookingError):
    """The external Calendar projection was not confirmed."""

    status_code = 502


@dataclass(frozen=True)
class EstimateBookingCommand:
    """Validated command received from the authenticated office proxy."""

    contact_id: UUID
    idempotency_key: str
    actor_id: int
    actor_name: str
    start_time: datetime
    duration_minutes: int
    service_type: str = "estimate"
    location: str | None = None
    notes: str = ""

    @property
    def actor(self) -> str:
        return f"employee:{self.actor_id}:{self.actor_name}"

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(minutes=self.duration_minutes)

    @property
    def request_fingerprint(self) -> str:
        canonical = {
            "contact_id": str(self.contact_id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "service_type": self.service_type,
            "location": self.location or "",
            "notes": self.notes,
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class EstimateBookingResult:
    """Closed response returned by the private EOM funnel route."""

    operation_id: UUID
    appointment_id: UUID | None
    calendar_event_id: str
    status: str
    idempotent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": str(self.operation_id),
            "appointment_id": str(self.appointment_id) if self.appointment_id else None,
            "calendar_event_id": self.calendar_event_id,
            "status": self.status,
            "idempotent": self.idempotent,
        }


class EOMLeadBookingService:
    """Coordinates one durable booking operation, Calendar event, and CRM state."""

    def __init__(
        self,
        *,
        pool: Any | None = None,
        calendar: CalendarTool | Any | None = None,
        config: EOMFunnelConfig | None = None,
    ) -> None:
        self._pool = pool or get_db_pool()
        self._calendar = calendar or calendar_tool
        self._config = config or funnel_settings

    @staticmethod
    def _event_id(operation_id: UUID) -> str:
        """Return a Google-safe stable event ID for this operation UUID."""
        return f"eom{operation_id.hex}"

    @staticmethod
    def _result(operation: Any, *, idempotent: bool) -> EstimateBookingResult:
        return EstimateBookingResult(
            operation_id=operation["id"],
            appointment_id=operation["appointment_id"],
            calendar_event_id=operation["calendar_event_id"],
            status=operation["status"],
            idempotent=idempotent,
        )

    @staticmethod
    def _contact_snapshot(contact: Any) -> str:
        return json.dumps(
            {
                "full_name": contact["full_name"],
                "phone": contact["phone"] or "",
                "email": contact["email"],
                "address": contact["address"],
            },
            sort_keys=True,
        )

    @staticmethod
    async def _load_same_key_operation(
        conn: Any,
        command: EstimateBookingCommand,
    ) -> Any:
        return await conn.fetchrow(
            """
            SELECT *
            FROM eom_lead_estimate_booking_operations
            WHERE contact_id = $1 AND idempotency_key = $2
            FOR UPDATE
            """,
            command.contact_id,
            command.idempotency_key,
        )

    @staticmethod
    def _validate_same_key_operation(
        operation: Any,
        command: EstimateBookingCommand,
    ) -> None:
        if operation["request_fingerprint"] != command.request_fingerprint:
            raise EOMLeadBookingConflictError(
                "Idempotency key was already used for different estimate details"
            )

    @staticmethod
    def _calendar_failure_status_code(result: ToolResult) -> int | None:
        raw_status = (result.data or {}).get("status_code")
        if raw_status is not None:
            try:
                return int(raw_status)
            except (TypeError, ValueError):
                return None
        status_match = re.search(r"\b([1-5][0-9]{2})\b", result.message or "")
        if status_match is None and result.error:
            status_match = re.search(r"\b([1-5][0-9]{2})\b", result.error)
        return int(status_match.group(1)) if status_match is not None else None

    @classmethod
    def _is_terminal_calendar_failure(cls, result: ToolResult) -> bool:
        if (result.data or {}).get("calendar_event_status") == "cancelled":
            return True
        if result.error in _PERMANENT_CALENDAR_RESULT_ERRORS:
            return True
        if result.error != "API_ERROR":
            return False
        status_code = cls._calendar_failure_status_code(result)
        return status_code in _PERMANENT_CALENDAR_HTTP_STATUSES

    @classmethod
    def _proves_calendar_event_settled(cls, result: ToolResult) -> bool:
        if (result.data or {}).get("calendar_event_status") == "cancelled":
            return True
        status_code = cls._calendar_failure_status_code(result)
        return status_code in {404, 410}

    @staticmethod
    def _operation_contact_snapshot(operation: Any) -> dict[str, Any]:
        snapshot = operation["contact_snapshot"]
        if isinstance(snapshot, str):
            snapshot = json.loads(snapshot)
        return dict(snapshot)

    @classmethod
    def _calendar_payload_for_operation(cls, operation: Any) -> dict[str, Any]:
        snapshot = cls._operation_contact_snapshot(operation)
        description_lines = [
            "EOM office estimate booking",
            f"Lead: {snapshot['full_name']}",
        ]
        if snapshot.get("phone"):
            description_lines.append(f"Phone: {snapshot['phone']}")
        if snapshot.get("email"):
            description_lines.append(f"Email: {snapshot['email']}")
        if operation["notes"]:
            description_lines.extend(("", operation["notes"]))
        return {
            "summary": f"Estimate: {snapshot['full_name']}"[:256],
            "start": operation["start_time"],
            "end": operation["end_time"],
            "location": operation["location"] or snapshot.get("address") or None,
            "description": "\n".join(description_lines)[:4000],
        }

    async def _reconcile_existing_calendar_event(self, operation: Any) -> bool | None:
        """Return True when the deterministic event is live, False when absent.

        ``None`` means Calendar could not prove either state, so the service
        must keep the operation retryable instead of terminally releasing the
        lead for a new key that could create a second live event.
        """
        get_event = getattr(self._calendar, "get_event", None)
        if get_event is None:
            return None
        try:
            result: ToolResult = await get_event(
                event_id=operation["calendar_event_id"],
                calendar_id=operation["calendar_id"],
            )
        except Exception:
            return None
        if (
            result.success
            and (result.data or {}).get("event_id") == operation["calendar_event_id"]
        ):
            mismatches = CalendarTool._calendar_event_mismatches(
                result.data or {},
                **self._calendar_payload_for_operation(operation),
            )
            if mismatches:
                logger.warning(
                    "Recovered Calendar event %s does not match booking operation %s: %s",
                    operation["calendar_event_id"],
                    operation["id"],
                    ", ".join(mismatches),
                )
                return None
            return True
        if self._proves_calendar_event_settled(result):
            return False
        return None

    async def _create_or_load_operation(
        self,
        command: EstimateBookingCommand,
    ) -> tuple[Any, bool]:
        """Persist or load one operation while serializing on the contact."""
        async with _transaction_connection(self._pool) as conn:
            existing = await self._load_same_key_operation(conn, command)
            if existing is not None:
                self._validate_same_key_operation(existing, command)
                return existing, False

            contact = await conn.fetchrow(
                """
                SELECT id, full_name, phone, email, address, lead_stage, status
                FROM contacts
                WHERE id = $1
                  AND business_context_id = $2
                  AND contact_type = 'lead'
                FOR UPDATE
                """,
                command.contact_id,
                EOM_BUSINESS_CONTEXT_ID,
            )
            if contact is None:
                raise EOMLeadBookingNotFoundError("EOM lead was not found")
            if contact["status"] != "active":
                raise EOMLeadBookingConflictError(
                    "Only active EOM leads can be booked for an estimate"
                )
            if contact["lead_stage"] != "new":
                raise EOMLeadBookingConflictError(
                    "Only EOM leads in the new stage can be booked for an estimate"
                )

            existing = await self._load_same_key_operation(conn, command)
            if existing is not None:
                self._validate_same_key_operation(existing, command)
                return existing, False

            active_operation = await conn.fetchrow(
                """
                SELECT id
                FROM eom_lead_estimate_booking_operations
                WHERE contact_id = $1
                  AND status <> $2
                FOR UPDATE
                """,
                command.contact_id,
                _TERMINAL_CALENDAR_FAILURE_STATUS,
            )
            if active_operation is not None:
                raise EOMLeadBookingConflictError(
                    "An estimate booking already exists for this lead"
                )

            operation_id = uuid4()
            operation = await conn.fetchrow(
                """
                INSERT INTO eom_lead_estimate_booking_operations (
                    id, contact_id, idempotency_key, request_fingerprint, actor,
                    start_time, end_time, service_type, location, notes,
                    contact_snapshot, calendar_id, calendar_event_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11::jsonb, $12, $13
                )
                RETURNING *
                """,
                operation_id,
                command.contact_id,
                command.idempotency_key,
                command.request_fingerprint,
                command.actor,
                command.start_time,
                command.end_time,
                command.service_type,
                command.location,
                command.notes,
                self._contact_snapshot(contact),
                self._config.estimate_calendar_id.strip() or "primary",
                self._event_id(operation_id),
            )
            return operation, True

    async def _claim_calendar_projection(self, operation_id: UUID) -> Any:
        """Give one request a short lease to project the durable command."""
        async with _transaction_connection(self._pool) as conn:
            operation = await conn.fetchrow(
                """
                SELECT *
                FROM eom_lead_estimate_booking_operations
                WHERE id = $1
                FOR UPDATE
                """,
                operation_id,
            )
            if operation is None:
                raise EOMLeadBookingNotFoundError("Booking operation no longer exists")
            if operation["appointment_id"] is not None:
                return operation
            if operation["status"] == _TERMINAL_CALENDAR_FAILURE_STATUS:
                raise EOMLeadBookingConflictError(
                    "Calendar permanently rejected this booking command; "
                    "submit a corrected booking request with a new idempotency key"
                )
            if (
                operation["status"] == "projecting"
                and operation["projection_started_at"] is not None
                and datetime.now(timezone.utc) - operation["projection_started_at"]
                < _PROJECTION_LEASE
            ):
                raise EOMLeadBookingConflictError(
                    "Estimate calendar projection is already in progress; retry the same key shortly"
                )
            projection_token = uuid4()
            return await conn.fetchrow(
                """
                UPDATE eom_lead_estimate_booking_operations
                SET status = 'projecting',
                    projection_started_at = NOW(),
                    projection_token = $2,
                    last_error = NULL,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
                """,
                operation_id,
                projection_token,
            )

    async def _mark_projection_failed(
        self,
        operation: Any,
        error: str,
        *,
        terminal: bool = False,
    ) -> None:
        await self._pool.execute(
            """
            UPDATE eom_lead_estimate_booking_operations
            SET status = $3,
                last_error = $2,
                projection_token = NULL,
                updated_at = NOW()
            WHERE id = $1
              AND appointment_id IS NULL
              AND status = 'projecting'
              AND projection_token = $4
            """,
            operation["id"],
            error[:_MAX_ERROR_LENGTH],
            (
                _TERMINAL_CALENDAR_FAILURE_STATUS
                if terminal
                else "calendar_failed"
            ),
            operation["projection_token"],
        )

    async def _refresh_projection_lease_before_calendar_write(
        self,
        operation: Any,
    ) -> Any:
        """Prove this holder still owns an unexpired lease before side effects."""
        refreshed = await self._pool.fetchrow(
            """
            UPDATE eom_lead_estimate_booking_operations
            SET projection_started_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
              AND appointment_id IS NULL
              AND status = 'projecting'
              AND projection_token = $2
              AND (
                    projection_started_at IS NULL
                    OR NOW() - projection_started_at < $3
                  )
            RETURNING *
            """,
            operation["id"],
            operation["projection_token"],
            _PROJECTION_LEASE,
        )
        if refreshed is None:
            raise EOMLeadBookingConflictError(
                "Estimate calendar projection lease expired before Calendar write; "
                "retry the same key"
            )
        return refreshed

    async def _project_calendar(self, operation: Any) -> None:
        operation = await self._refresh_projection_lease_before_calendar_write(
            operation
        )
        calendar_payload = self._calendar_payload_for_operation(operation)

        result: ToolResult = await self._calendar.create_event(
            **calendar_payload,
            calendar_id=operation["calendar_id"],
            event_id=operation["calendar_event_id"],
        )
        if result.success:
            event_id = (result.data or {}).get("event_id")
            if event_id == operation["calendar_event_id"]:
                return
            error = "Calendar projection returned a different event ID"
            terminal = True
        else:
            error = result.message or result.error or "Calendar projection failed"
            terminal = self._is_terminal_calendar_failure(result)
            if terminal:
                reconciled = await self._reconcile_existing_calendar_event(operation)
                if reconciled is True:
                    return
                if reconciled is None:
                    terminal = False
                    error = (
                        f"{error}; deterministic Calendar event could not be reconciled"
                    )
        await self._mark_projection_failed(operation, str(error), terminal=terminal)
        raise EOMLeadBookingProjectionError(str(error)[:_MAX_ERROR_LENGTH])

    async def _complete_operation(self, operation_id: UUID, projection_token: UUID) -> Any:
        """Link appointment, advance lead stage, and append evidence together."""
        async with _transaction_connection(self._pool) as conn:
            operation_contact = await conn.fetchrow(
                """
                SELECT contact_id
                FROM eom_lead_estimate_booking_operations
                WHERE id = $1
                """,
                operation_id,
            )
            if operation_contact is None:
                raise EOMLeadBookingNotFoundError("Booking operation no longer exists")

            # Customer approval locks the contact before checking the booking
            # operation, so completion must take the same order after Calendar
            # projection returns.  That keeps approval/completion overlap from
            # deadlocking after the external event already exists.
            contact = await conn.fetchrow(
                """
                SELECT id
                FROM contacts
                WHERE id = $1
                FOR UPDATE
                """,
                operation_contact["contact_id"],
            )
            if contact is None:
                raise EOMLeadBookingNotFoundError("Booking contact no longer exists")

            operation = await conn.fetchrow(
                """
                SELECT *
                FROM eom_lead_estimate_booking_operations
                WHERE id = $1
                FOR UPDATE
                """,
                operation_id,
            )
            if operation is None:
                raise EOMLeadBookingNotFoundError("Booking operation no longer exists")
            if operation["status"] == _TERMINAL_CALENDAR_FAILURE_STATUS:
                raise EOMLeadBookingConflictError(
                    "Calendar permanently rejected this booking command; "
                    "submit a corrected booking request with a new idempotency key"
                )
            if operation["appointment_id"] is not None:
                return operation
            if operation["projection_token"] != projection_token:
                raise EOMLeadBookingConflictError(
                    "Estimate calendar projection lease expired; retry the same key"
                )

            snapshot = operation["contact_snapshot"]
            if isinstance(snapshot, str):
                snapshot = json.loads(snapshot)
            appointment = await conn.fetchrow(
                """
                INSERT INTO appointments (
                    start_time, end_time, duration_minutes, service_type, notes,
                    customer_name, customer_phone, customer_email, customer_address,
                    calendar_event_id, business_context_id, status, contact_id,
                    eom_estimate_booking_operation_id, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                    'confirmed', $12, $13, $14::jsonb
                )
                ON CONFLICT (eom_estimate_booking_operation_id)
                    WHERE eom_estimate_booking_operation_id IS NOT NULL
                    DO NOTHING
                RETURNING id
                """,
                operation["start_time"],
                operation["end_time"],
                int((operation["end_time"] - operation["start_time"]).total_seconds() // 60),
                operation["service_type"],
                operation["notes"],
                snapshot["full_name"],
                snapshot.get("phone") or "",
                snapshot.get("email"),
                snapshot.get("address"),
                operation["calendar_event_id"],
                EOM_BUSINESS_CONTEXT_ID,
                operation["contact_id"],
                operation["id"],
                json.dumps(
                    {
                        "source": "eom_lead_funnel",
                        "operation_id": str(operation["id"]),
                    }
                ),
            )
            if appointment is None:
                appointment = await conn.fetchrow(
                    """
                    SELECT id
                    FROM appointments
                    WHERE eom_estimate_booking_operation_id = $1
                    """,
                    operation["id"],
                )
            if appointment is None:
                raise RuntimeError("EOM estimate appointment link was not persisted")

            transitioned = await conn.fetchrow(
                """
                UPDATE contacts
                SET lead_stage = 'estimate_booked', updated_at = NOW()
                WHERE id = $1
                  AND business_context_id = $2
                  AND contact_type = 'lead'
                  AND status = 'active'
                  AND lead_stage = 'new'
                RETURNING id
                """,
                operation["contact_id"],
                EOM_BUSINESS_CONTEXT_ID,
            )
            if transitioned is None:
                raise EOMLeadBookingConflictError(
                    "Lead changed before its estimate booking could be completed"
                )

            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor, source,
                    operation_key, metadata
                ) VALUES ($1, 'estimate_booked', 'new', 'estimate_booked', $2,
                          'eom_lead_funnel', $3, $4::jsonb)
                ON CONFLICT (contact_id, event_type, operation_key)
                    WHERE operation_key IS NOT NULL
                    DO NOTHING
                """,
                operation["contact_id"],
                operation["actor"],
                operation["idempotency_key"],
                json.dumps(
                    {
                        "operation_id": str(operation["id"]),
                        "appointment_id": str(appointment["id"]),
                        "calendar_event_id": operation["calendar_event_id"],
                    }
                ),
            )
            return await conn.fetchrow(
                """
                UPDATE eom_lead_estimate_booking_operations
                SET appointment_id = $2,
                    status = 'completed',
                    last_error = NULL,
                    projection_token = NULL,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING *
                """,
                operation["id"],
                appointment["id"],
            )

    async def book_estimate(
        self,
        command: EstimateBookingCommand,
    ) -> EstimateBookingResult:
        """Create or replay exactly one first estimate booking command."""
        operation, created = await self._create_or_load_operation(command)
        if operation["appointment_id"] is not None:
            return self._result(operation, idempotent=True)
        projection = await self._claim_calendar_projection(operation["id"])
        if projection["appointment_id"] is not None:
            return self._result(projection, idempotent=True)
        await self._project_calendar(projection)
        completed = await self._complete_operation(
            projection["id"],
            projection["projection_token"],
        )
        return self._result(completed, idempotent=not created)
