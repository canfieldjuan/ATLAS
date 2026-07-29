"""Real-Postgres proof for durable EOM estimate booking."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.eom_api.config import EOMFunnelConfig  # noqa: E402
from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402
from atlas_brain.services.eom_lead_booking import (  # noqa: E402
    EOMLeadBookingConflictError,
    EOMLeadBookingProjectionError,
    EOMLeadBookingService,
    EstimateBookingCommand,
)
from atlas_brain.storage.migrations import (  # noqa: E402
    _contains_executable_concurrently,
    _split_sql_statements,
)
from atlas_brain.services.eom_lead_conversion import EOMLeadConversionError  # noqa: E402
from atlas_brain.tools.base import ToolResult  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
_NOCODB_TEST_PASSWORD = "test-only-nocodb-password"


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
    return database_url


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


async def _require_disposable_role_administration(conn) -> None:
    can_administer_roles = await conn.fetchval(
        """
        SELECT rolsuper OR rolcreaterole
        FROM pg_roles
        WHERE rolname = current_user
        """
    )
    if not can_administer_roles:
        pytest.skip("privilege migration proof requires disposable role administration")


async def _provision_nocodb_login(conn) -> None:
    await _require_disposable_role_administration(conn)
    existing_role = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'atlas_nocodb')"
    )
    if existing_role:
        await conn.execute(
            "ALTER ROLE atlas_nocodb LOGIN NOINHERIT "
            f"PASSWORD '{_NOCODB_TEST_PASSWORD}'"
        )
    else:
        await conn.execute(
            "CREATE ROLE atlas_nocodb LOGIN NOINHERIT "
            f"PASSWORD '{_NOCODB_TEST_PASSWORD}'"
        )
    database_name = await conn.fetchval("SELECT current_database()")
    await conn.execute(
        f"GRANT CONNECT ON DATABASE {_quote_ident(database_name)} TO atlas_nocodb"
    )


async def _execute_migration_file(conn, name: str) -> None:
    sql = (MIGRATIONS / name).read_text()
    if _contains_executable_concurrently(sql):
        for statement in _split_sql_statements(sql):
            await conn.execute(statement)
    else:
        await conn.execute(sql)


async def _prepare_schema(
    conn,
    schema: str,
    *,
    apply_privilege_migration: bool = False,
) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}", public')
    migration_names = [
        "012_appointments.sql",
        "035_contacts.sql",
        "256_contact_interaction_dedupe.sql",
        "346_contact_lead_pipeline.sql",
        "348_appointment_operating_fields.sql",
        "351_eom_lead_lifecycle_events.sql",
        "353_eom_customer_handoffs.sql",
    ]
    if apply_privilege_migration:
        await _provision_nocodb_login(conn)
        migration_names.append("354_eom_customer_handoff_privileges.sql")
    migration_names.append(
        "356_eom_lead_estimate_booking_operations.sql",
    )
    migration_names.append(
        "358_eom_estimate_booking_appointment_link_index.sql",
    )
    for name in migration_names:
        await _execute_migration_file(conn, name)


async def _insert_lead(conn, *, lead_stage: str = "new") -> UUID:
    contact_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO contacts (
            id, full_name, email, phone, address, business_context_id,
            contact_type, lead_stage, status, source
        )
        VALUES (
            $1, 'Estimate Lead', 'estimate@example.com', '2175550101',
            '100 Main St', 'effingham_maids', 'lead', $2, 'active', 'web'
        )
        """,
        contact_id,
        lead_stage,
    )
    return contact_id


class _Calendar:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create_event(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(
            success=True,
            data={"event_id": kwargs["event_id"]},
            message="created",
        )


class _RejectingCalendar:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.gets: list[dict[str, object]] = []

    async def create_event(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(
            success=False,
            data={"status_code": 404},
            error="API_ERROR",
            message="Calendar API error: 404",
        )

    async def get_event(self, **kwargs) -> ToolResult:
        self.gets.append(kwargs)
        return ToolResult(
            success=False,
            data={"status_code": 404, "event_id": kwargs["event_id"]},
            error="API_ERROR",
            message="Calendar event not found",
        )


class _RecoveringCalendar:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.gets: list[dict[str, object]] = []

    async def create_event(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(
            success=False,
            error="AUTH_ERROR",
            message="Calendar authentication failed",
        )

    async def get_event(self, **kwargs) -> ToolResult:
        self.gets.append(kwargs)
        return ToolResult(
            success=True,
            data={
                "event_id": kwargs["event_id"],
                "summary": "Estimate: Estimate Lead",
                "start": {"dateTime": "2026-08-01T15:00:00+00:00"},
                "end": {"dateTime": "2026-08-01T16:00:00+00:00"},
                "location": "100 Main St",
                "description": (
                    "EOM office estimate booking\n"
                    "Lead: Estimate Lead\n"
                    "Phone: 2175550101\n"
                    "Email: estimate@example.com\n"
                    "\n"
                    "Use side door"
                ),
                "calendar_event_status": "confirmed",
            },
            message="fetched",
        )


class _UnverifiableCalendar(_RecoveringCalendar):
    async def get_event(self, **kwargs) -> ToolResult:
        self.gets.append(kwargs)
        return ToolResult(
            success=False,
            error="AUTH_ERROR",
            message="Calendar authentication failed",
        )


@pytest.mark.asyncio
async def test_booking_operation_calendar_appointment_stage_and_approval_chain():
    database_url = _database_url_or_skip()
    schema = f"test_eom_estimate_booking_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        contact_id = await _insert_lead(conn)
        calendar = _Calendar()
        service = EOMLeadBookingService(
            pool=conn,
            calendar=calendar,
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )
        command = EstimateBookingCommand(
            contact_id=contact_id,
            idempotency_key="estimate-booking-key-0001",
            actor_id=7,
            actor_name="Mayra",
            start_time=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            duration_minutes=60,
            service_type="Cleaning estimate",
            location="100 Main St",
            notes="Use side door",
        )

        created = await service.book_estimate(command)
        replay = await service.book_estimate(command)

        assert created.status == "completed"
        assert created.appointment_id is not None
        assert created.calendar_event_id.startswith("eom")
        assert replay.idempotent is True
        assert replay.appointment_id == created.appointment_id
        assert len(calendar.calls) == 1
        assert calendar.calls[0]["calendar_id"] == "estimate-calendar"
        assert calendar.calls[0]["event_id"] == created.calendar_event_id

        changed = EstimateBookingCommand(
            contact_id=contact_id,
            idempotency_key=command.idempotency_key,
            actor_id=7,
            actor_name="Mayra",
            start_time=command.start_time,
            duration_minutes=90,
            service_type=command.service_type,
            location=command.location,
            notes=command.notes,
        )
        with pytest.raises(EOMLeadBookingConflictError):
            await service.book_estimate(changed)
        assert len(calendar.calls) == 1

        contact_stage = await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1",
            contact_id,
        )
        appointment_count = await conn.fetchval(
            "SELECT COUNT(*) FROM appointments WHERE contact_id = $1",
            contact_id,
        )
        lifecycle = await conn.fetchrow(
            """
            SELECT from_stage, to_stage, actor, metadata->>'calendar_event_id' AS event_id
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'estimate_booked'
            """,
            contact_id,
        )
        assert contact_stage == "estimate_booked"
        assert appointment_count == 1
        assert dict(lifecycle) == {
            "from_stage": "new",
            "to_stage": "estimate_booked",
            "actor": "employee:7:Mayra",
            "event_id": created.calendar_event_id,
        }

        provider = DatabaseCRMProvider(pool=conn)
        review_rows = await provider.list_eom_new_lead_review_items(limit=10)
        assert [row["contact_id"] for row in review_rows] == [contact_id]

        handoff = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key="fixture",
            actor_id=9,
            actor_name="Juan",
        )
        assert handoff["contact_id"] == str(contact_id)
        approved_from_stage = await conn.fetchval(
            """
            SELECT from_stage
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'customer_approved'
            """,
            contact_id,
        )
        assert approved_from_stage == "estimate_booked"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_expired_projection_retry_reconciles_stale_calendar_success():
    database_url = _database_url_or_skip()
    schema = f"test_eom_stale_projection_reconcile_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        contact_id = await _insert_lead(conn)
        command = EstimateBookingCommand(
            contact_id=contact_id,
            idempotency_key="estimate-booking-stale-success",
            actor_id=7,
            actor_name="Mayra",
            start_time=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            duration_minutes=60,
            service_type="Cleaning estimate",
            location="100 Main St",
            notes="Use side door",
        )
        service = EOMLeadBookingService(
            pool=conn,
            calendar=_Calendar(),
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )
        operation, _ = await service._create_or_load_operation(command)
        leased = await service._claim_calendar_projection(operation["id"])
        await conn.execute(
            """
            UPDATE eom_lead_estimate_booking_operations
            SET status = 'projecting',
                projection_started_at = NOW() - INTERVAL '3 minutes',
                projection_token = $2
            WHERE id = $1
            """,
            operation["id"],
            leased["projection_token"],
        )

        calendar = _RecoveringCalendar()
        retry_service = EOMLeadBookingService(
            pool=conn,
            calendar=calendar,
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )

        result = await retry_service.book_estimate(command)

        assert result.status == "completed"
        assert result.appointment_id is not None
        assert len(calendar.calls) == 1
        assert calendar.gets == [
            {
                "event_id": operation["calendar_event_id"],
                "calendar_id": operation["calendar_id"],
            }
        ]
        stored = await conn.fetchrow(
            """
            SELECT status, appointment_id, calendar_event_id
            FROM eom_lead_estimate_booking_operations
            WHERE id = $1
            """,
            operation["id"],
        )
        assert stored["status"] == "completed"
        assert stored["appointment_id"] == result.appointment_id
        assert stored["calendar_event_id"] == operation["calendar_event_id"]
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_terminal_calendar_failure_without_reconciliation_stays_retryable():
    database_url = _database_url_or_skip()
    schema = f"test_eom_unverified_projection_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        contact_id = await _insert_lead(conn)
        calendar = _UnverifiableCalendar()
        service = EOMLeadBookingService(
            pool=conn,
            calendar=calendar,
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )
        command = EstimateBookingCommand(
            contact_id=contact_id,
            idempotency_key="estimate-booking-unverified",
            actor_id=7,
            actor_name="Mayra",
            start_time=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            duration_minutes=60,
            service_type="Cleaning estimate",
            location="100 Main St",
            notes="Use side door",
        )

        with pytest.raises(EOMLeadBookingProjectionError, match="could not be reconciled"):
            await service.book_estimate(command)

        operation = await conn.fetchrow(
            """
            SELECT id, status, appointment_id, last_error
            FROM eom_lead_estimate_booking_operations
            WHERE contact_id = $1
            """,
            contact_id,
        )
        assert operation["status"] == "calendar_failed"
        assert operation["appointment_id"] is None
        assert "could not be reconciled" in operation["last_error"]
        corrected = EstimateBookingCommand(
            contact_id=contact_id,
            idempotency_key="estimate-booking-new-key-blocked",
            actor_id=7,
            actor_name="Mayra",
            start_time=command.start_time,
            duration_minutes=command.duration_minutes,
            service_type=command.service_type,
            location=command.location,
            notes=command.notes,
        )
        with pytest.raises(EOMLeadBookingConflictError, match="already exists"):
            await service.book_estimate(corrected)
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_expired_projection_holder_cannot_issue_calendar_write_after_reclaim():
    database_url = _database_url_or_skip()
    schema = f"test_eom_stale_projection_write_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        contact_id = await _insert_lead(conn)
        operation_id = uuid.uuid4()
        stale_projection_token = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO eom_lead_estimate_booking_operations (
                id, contact_id, idempotency_key, request_fingerprint, actor,
                start_time, end_time, service_type, notes, contact_snapshot,
                calendar_id, calendar_event_id, status, projection_started_at,
                projection_token
            ) VALUES (
                $1, $2, 'stale-side-effect-holder', $3, 'employee:7:Mayra',
                $4, $5, 'Cleaning estimate', '', $6::jsonb,
                'estimate-calendar', $7, 'projecting',
                NOW() - INTERVAL '3 minutes', $8
            )
            """,
            operation_id,
            contact_id,
            "3" * 64,
            datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
            '{"full_name":"Estimate Lead","phone":"","email":"estimate@example.com","address":"100 Main St"}',
            f"eom{operation_id.hex}",
            stale_projection_token,
        )
        stale_operation = await conn.fetchrow(
            "SELECT * FROM eom_lead_estimate_booking_operations WHERE id = $1",
            operation_id,
        )

        rejecting_calendar = _RejectingCalendar()
        reclaiming_service = EOMLeadBookingService(
            pool=conn,
            calendar=rejecting_calendar,
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )
        reclaimed = await reclaiming_service._claim_calendar_projection(operation_id)
        assert reclaimed["projection_token"] != stale_projection_token
        with pytest.raises(EOMLeadBookingProjectionError, match="Calendar API error: 404"):
            await reclaiming_service._project_calendar(reclaimed)
        assert len(rejecting_calendar.calls) == 1

        stale_calendar = _Calendar()
        stale_service = EOMLeadBookingService(
            pool=conn,
            calendar=stale_calendar,
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )
        with pytest.raises(EOMLeadBookingConflictError, match="lease expired"):
            await stale_service._project_calendar(stale_operation)
        assert stale_calendar.calls == []
        stored = await conn.fetchrow(
            """
            SELECT status, projection_token, appointment_id
            FROM eom_lead_estimate_booking_operations
            WHERE id = $1
            """,
            operation_id,
        )
        assert stored["status"] == "calendar_rejected"
        assert stored["projection_token"] is None
        assert stored["appointment_id"] is None
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_permanent_calendar_failure_can_be_corrected_with_new_key():
    database_url = _database_url_or_skip()
    schema = f"test_eom_estimate_calendar_reject_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        contact_id = await _insert_lead(conn)
        rejecting_calendar = _RejectingCalendar()
        rejecting_service = EOMLeadBookingService(
            pool=conn,
            calendar=rejecting_calendar,
            config=EOMFunnelConfig(estimate_calendar_id="deleted-calendar"),
        )
        rejected_command = EstimateBookingCommand(
            contact_id=contact_id,
            idempotency_key="estimate-booking-key-rejected",
            actor_id=7,
            actor_name="Mayra",
            start_time=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            duration_minutes=60,
            service_type="Cleaning estimate",
            location="100 Main St",
            notes="Use side door",
        )

        with pytest.raises(
            EOMLeadBookingProjectionError,
            match="Calendar API error: 404",
        ):
            await rejecting_service.book_estimate(rejected_command)
        assert len(rejecting_calendar.calls) == 1
        rejected_operation = await conn.fetchrow(
            """
            SELECT id, status, appointment_id
            FROM eom_lead_estimate_booking_operations
            WHERE contact_id = $1 AND idempotency_key = $2
            """,
            contact_id,
            rejected_command.idempotency_key,
        )
        assert rejected_operation is not None
        assert rejected_operation["status"] == "calendar_rejected"
        assert rejected_operation["appointment_id"] is None

        with pytest.raises(EOMLeadBookingConflictError, match="permanently rejected"):
            await rejecting_service.book_estimate(rejected_command)
        assert len(rejecting_calendar.calls) == 1

        correcting_calendar = _Calendar()
        correcting_service = EOMLeadBookingService(
            pool=conn,
            calendar=correcting_calendar,
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )
        corrected_command = EstimateBookingCommand(
            contact_id=contact_id,
            idempotency_key="estimate-booking-key-corrected",
            actor_id=7,
            actor_name="Mayra",
            start_time=rejected_command.start_time,
            duration_minutes=rejected_command.duration_minutes,
            service_type=rejected_command.service_type,
            location=rejected_command.location,
            notes=rejected_command.notes,
        )

        corrected = await correcting_service.book_estimate(corrected_command)

        assert corrected.status == "completed"
        assert corrected.appointment_id is not None
        assert len(correcting_calendar.calls) == 1
        statuses = await conn.fetch(
            """
            SELECT idempotency_key, status
            FROM eom_lead_estimate_booking_operations
            WHERE contact_id = $1
            """,
            contact_id,
        )
        assert {row["idempotency_key"]: row["status"] for row in statuses} == {
            "estimate-booking-key-rejected": "calendar_rejected",
            "estimate-booking-key-corrected": "completed",
        }
        provider = DatabaseCRMProvider(pool=conn)
        handoff = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key="corrected-after-terminal-calendar-reject",
            actor_id=9,
            actor_name="Juan",
        )
        assert handoff["contact_id"] == str(contact_id)
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_booking_rejects_non_new_stage_before_calendar_projection():
    database_url = _database_url_or_skip()
    schema = f"test_eom_estimate_reject_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        contact_id = await _insert_lead(conn, lead_stage="estimate_booked")
        calendar = _Calendar()
        service = EOMLeadBookingService(
            pool=conn,
            calendar=calendar,
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )

        with pytest.raises(EOMLeadBookingConflictError):
            await service.book_estimate(
                EstimateBookingCommand(
                    contact_id=contact_id,
                    idempotency_key="estimate-booking-key-0002",
                    actor_id=7,
                    actor_name="Mayra",
                    start_time=datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
                    duration_minutes=60,
                )
            )
        assert calendar.calls == []
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_expired_projection_holder_cannot_fail_or_complete_reclaimed_lease():
    database_url = _database_url_or_skip()
    schema = f"test_eom_estimate_projection_lease_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        contact_id = await _insert_lead(conn)
        operation_id = uuid.uuid4()
        stale_projection_token = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO eom_lead_estimate_booking_operations (
                id, contact_id, idempotency_key, request_fingerprint, actor,
                start_time, end_time, service_type, notes, contact_snapshot,
                calendar_id, calendar_event_id, status, projection_started_at,
                projection_token
            ) VALUES (
                $1, $2, 'stale-projection-holder', $3, 'employee:7:Mayra',
                $4, $5, 'Cleaning estimate', '', $6::jsonb,
                'estimate-calendar', $7, 'projecting',
                NOW() - INTERVAL '3 minutes', $8
            )
            """,
            operation_id,
            contact_id,
            "2" * 64,
            datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
            '{"full_name":"Estimate Lead","phone":"","email":"estimate@example.com","address":"100 Main St"}',
            f"eom{operation_id.hex}",
            stale_projection_token,
        )
        service = EOMLeadBookingService(
            pool=conn,
            calendar=_Calendar(),
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )
        stale_operation = await conn.fetchrow(
            "SELECT * FROM eom_lead_estimate_booking_operations WHERE id = $1",
            operation_id,
        )
        fresh_projection = await service._claim_calendar_projection(operation_id)
        assert fresh_projection["projection_token"] != stale_projection_token

        await service._mark_projection_failed(
            stale_operation,
            "Calendar API error: 404",
            terminal=True,
        )
        still_projecting = await conn.fetchrow(
            """
            SELECT status, last_error, projection_token
            FROM eom_lead_estimate_booking_operations
            WHERE id = $1
            """,
            operation_id,
        )
        assert still_projecting["status"] == "projecting"
        assert still_projecting["last_error"] is None
        assert still_projecting["projection_token"] == fresh_projection["projection_token"]

        with pytest.raises(EOMLeadBookingConflictError, match="lease expired"):
            await service._complete_operation(operation_id, stale_projection_token)
        completed = await service._complete_operation(
            operation_id,
            fresh_projection["projection_token"],
        )
        assert completed["status"] == "completed"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_customer_handoff_rejects_pending_estimate_booking_operation():
    database_url = _database_url_or_skip()
    schema = f"test_eom_estimate_handoff_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        contact_id = await _insert_lead(conn)
        await conn.execute(
            """
            INSERT INTO eom_lead_estimate_booking_operations (
                id, contact_id, idempotency_key, request_fingerprint, actor,
                start_time, end_time, service_type, notes, contact_snapshot,
                calendar_id, calendar_event_id, status, projection_started_at
            ) VALUES (
                $1, $2, 'pending-booking', $3, 'employee:7:Mayra',
                $4, $5, 'Cleaning estimate', '', $6::jsonb,
                'estimate-calendar', $7, 'projecting', NOW()
            )
            """,
            uuid.uuid4(),
            contact_id,
            "0" * 64,
            datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
            '{"full_name":"Estimate Lead","phone":"","email":"estimate@example.com","address":"100 Main St"}',
            f"eom{uuid.uuid4().hex}",
        )

        provider = DatabaseCRMProvider(pool=conn)
        with pytest.raises(EOMLeadConversionError, match="booking must complete"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key="fixture",
                actor_id=9,
                actor_name="Juan",
            )

        contact_state = await conn.fetchrow(
            """
            SELECT contact_type, lead_stage
            FROM contacts
            WHERE id = $1
            """,
            contact_id,
        )
        assert dict(contact_state) == {
            "contact_type": "lead",
            "lead_stage": "new",
        }
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1",
                contact_id,
            )
            == 0
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_nocodb_cannot_write_estimate_booking_operation_link():
    database_url = _database_url_or_skip()
    schema = f"test_eom_estimate_nocodb_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    nocodb_conn = None
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=True)

        assert await conn.fetchval(
            "SELECT has_column_privilege('atlas_nocodb', 'appointments', 'notes', 'UPDATE')"
        )
        assert not await conn.fetchval(
            """
            SELECT has_column_privilege(
                'atlas_nocodb',
                'appointments',
                'eom_estimate_booking_operation_id',
                'INSERT'
            )
            """
        )
        assert not await conn.fetchval(
            """
            SELECT has_column_privilege(
                'atlas_nocodb',
                'appointments',
                'eom_estimate_booking_operation_id',
                'UPDATE'
            )
            """
        )

        nocodb_conn = await asyncpg.connect(
            database_url,
            user="atlas_nocodb",
            password=_NOCODB_TEST_PASSWORD,
        )
        await nocodb_conn.execute(f"SET search_path TO {_quote_ident(schema)}, public")
        await nocodb_conn.execute("UPDATE appointments SET notes = notes")
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await nocodb_conn.execute(
                "UPDATE appointments SET eom_estimate_booking_operation_id = NULL"
            )
    finally:
        if nocodb_conn is not None:
            await nocodb_conn.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_pending_estimate_booking_fences_contact_state_until_projection_finishes():
    database_url = _database_url_or_skip()
    schema = f"test_eom_estimate_contact_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    nocodb_conn = None
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=True)
        contact_id = await _insert_lead(conn)
        operation_id = uuid.uuid4()
        projection_token = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO eom_lead_estimate_booking_operations (
                id, contact_id, idempotency_key, request_fingerprint, actor,
                start_time, end_time, service_type, notes, contact_snapshot,
                calendar_id, calendar_event_id, status, projection_started_at,
                projection_token
            ) VALUES (
                $1, $2, 'pending-contact-fence', $3, 'employee:7:Mayra',
                $4, $5, 'Cleaning estimate', '', $6::jsonb,
                'estimate-calendar', $7, 'projecting', NOW(), $8
            )
            """,
            operation_id,
            contact_id,
            "1" * 64,
            datetime(2026, 8, 1, 15, tzinfo=timezone.utc),
            datetime(2026, 8, 1, 16, tzinfo=timezone.utc),
            '{"full_name":"Estimate Lead","phone":"","email":"estimate@example.com","address":"100 Main St"}',
            f"eom{operation_id.hex}",
            projection_token,
        )

        provider = DatabaseCRMProvider(pool=conn)
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await provider.update_contact(str(contact_id), {"status": "archived"})
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await provider.delete_contact(str(contact_id))

        nocodb_conn = await asyncpg.connect(
            database_url,
            user="atlas_nocodb",
            password=_NOCODB_TEST_PASSWORD,
        )
        await nocodb_conn.execute(f"SET search_path TO {_quote_ident(schema)}, public")
        await nocodb_conn.execute(
            "UPDATE contacts SET notes = 'ordinary edit' WHERE id = $1",
            contact_id,
        )
        await nocodb_conn.execute(
            """
            CREATE TEMP TABLE eom_lead_estimate_booking_operations (
                id UUID,
                contact_id UUID,
                appointment_id UUID,
                status TEXT
            )
            """
        )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await nocodb_conn.execute(
                "UPDATE contacts SET status = 'archived' WHERE id = $1",
                contact_id,
            )
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await nocodb_conn.execute(
                "DELETE FROM contacts WHERE id = $1",
                contact_id,
            )

        service = EOMLeadBookingService(
            pool=conn,
            calendar=_Calendar(),
            config=EOMFunnelConfig(estimate_calendar_id="estimate-calendar"),
        )
        completed = await service._complete_operation(operation_id, projection_token)
        assert completed["status"] == "completed"
        assert await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1",
            contact_id,
        ) == "estimate_booked"
    finally:
        if nocodb_conn is not None:
            await nocodb_conn.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
