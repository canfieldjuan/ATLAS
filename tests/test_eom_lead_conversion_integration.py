"""Real-Postgres proof for the EOM customer handoff transaction."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.services.crm_provider import (  # noqa: E402
    DatabaseCRMProvider,
    _eom_identity_lock_key,
)
from atlas_brain.services.eom_crm_mutations import (  # noqa: E402
    EOMOperatorContactMutation,
    EOMOperatorContactMutationError,
    mutate_eom_operator_contact,
)
from atlas_brain.services.eom_estimate_booking import (  # noqa: E402
    deterministic_eom_estimate_calendar_event_id,
    deterministic_eom_first_clean_calendar_event_id,
)
from atlas_brain.services.eom_lead_conversion import (  # noqa: E402
    EOMLeadConversionError,
)
from atlas_brain.services.eom_lead_ingress import (  # noqa: E402
    resolve_or_create_eom_inbound_lead,
)

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
_NOCODB_TEST_PASSWORD = "test-only-nocodb-password"
_NON_SUPERUSER_TEST_PASSWORD = "test-only-migrator-password"


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
    return database_url


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


async def _require_disposable_role_administration(conn) -> None:
    can_administer_roles = await conn.fetchval("""
        SELECT rolsuper OR rolcreaterole
        FROM pg_roles
        WHERE rolname = current_user
        """)
    if not can_administer_roles:
        pytest.skip("privilege migration proof requires disposable role administration")


async def _provision_nocodb_login(conn) -> None:
    """Mirror the DBA-only NocoDB provision step without sourcing a real secret."""
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


async def _provision_handoff_guard(conn) -> None:
    await _require_disposable_role_administration(conn)
    await conn.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'atlas_eom_handoff_owner'
            ) THEN
                CREATE ROLE atlas_eom_handoff_owner NOLOGIN NOINHERIT;
            END IF;
        END;
        $$;
        """)


async def _prepare_schema(
    conn,
    schema: str,
    *,
    apply_privilege_migration: bool = True,
    apply_lifecycle_sequence_migration: bool = True,
) -> None:
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}", public')
    await conn.execute("""
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
        """)
    await conn.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
    await conn.execute("CREATE TABLE api_keys (id UUID PRIMARY KEY)")
    await conn.execute("CREATE TABLE byok_keys (id UUID PRIMARY KEY)")
    await conn.execute("CREATE TABLE scoped_mailbox_credentials (id UUID PRIMARY KEY)")
    # Referenced by sent_emails (016) foreign keys only; the funnel never
    # touches them.
    await conn.execute("CREATE TABLE sessions (id UUID PRIMARY KEY)")
    await conn.execute("CREATE TABLE users (id UUID PRIMARY KEY)")
    migration_names = [
        "016_sent_emails.sql",
        "035_contacts.sql",
        "256_contact_interaction_dedupe.sql",
        "346_contact_lead_pipeline.sql",
        "348_appointment_operating_fields.sql",
        "349_sent_emails_business_context.sql",
        "351_eom_lead_lifecycle_events.sql",
        "352_eom_inbound_delivery_receipts.sql",
        "353_eom_customer_handoffs.sql",
        "360_eom_onboarding_email_drafts.sql",
        "361_eom_onboarding_draft_actor_bigint.sql",
    ]
    if apply_lifecycle_sequence_migration:
        migration_names.append("363_eom_lead_lifecycle_sequence.sql")
    if apply_privilege_migration:
        await _provision_nocodb_login(conn)
        migration_names.append("354_eom_customer_handoff_privileges.sql")
    for name in migration_names:
        await conn.execute((MIGRATIONS / name).read_text())


async def _insert_contact(
    conn,
    *,
    contact_id: uuid.UUID,
    business_context_id: str = "effingham_maids",
    contact_type: str = "lead",
    lead_stage: str | None = "new",
    status: str = "active",
    full_name: str = "Approved Estimate",
    email: str | None = None,
    phone: str | None = None,
    address: str | None = None,
    source: str | None = "web",
    created_at: datetime | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO contacts (
            id, full_name, email, phone, address, source, created_at,
            business_context_id, contact_type, lead_stage, status
        ) VALUES ($1, $2, $3, $4, $5, $6, COALESCE($7, NOW()), $8, $9, $10, $11)
        """,
        contact_id,
        full_name,
        email,
        phone,
        address,
        source,
        created_at,
        business_context_id,
        contact_type,
        lead_stage,
        status,
    )


async def _contact_state(
    conn, contact_id: uuid.UUID
) -> tuple[dict[str, object], int, int]:
    contact = await conn.fetchrow(
        """
        SELECT business_context_id, contact_type, lead_stage, status
        FROM contacts WHERE id = $1
        """,
        contact_id,
    )
    assert contact is not None
    customer_approval_count = await conn.fetchval(
        """
        SELECT COUNT(*) FROM eom_lead_lifecycle_events
        WHERE contact_id = $1 AND event_type = 'customer_approved'
        """,
        contact_id,
    )
    handoff_count = await conn.fetchval(
        "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1",
        contact_id,
    )
    return dict(contact), int(customer_approval_count), int(handoff_count)


def _approval_key() -> str:
    return f"office-handoff-{uuid.uuid4().hex}"


def _metadata_dict(value):
    if isinstance(value, str):
        return json.loads(value)
    return dict(value or {})


class _IdentityLockGateConnection:
    """Hold the first shared identity lock while a second writer reaches it."""

    def __init__(
        self,
        connection,
        *,
        lock_key: str,
        first_locked: asyncio.Event,
        second_waiting: asyncio.Event,
        release_first: asyncio.Event,
    ) -> None:
        self._connection = connection
        self._lock_key = lock_key
        self._first_locked = first_locked
        self._second_waiting = second_waiting
        self._release_first = release_first

    def __getattr__(self, name):
        return getattr(self._connection, name)

    async def execute(self, *args):
        sql = str(args[0]) if args else ""
        lock_key = str(args[1]) if len(args) > 1 else ""
        if "pg_advisory_xact_lock" in sql and lock_key == self._lock_key:
            if not self._first_locked.is_set():
                result = await self._connection.execute(*args)
                self._first_locked.set()
                await self._release_first.wait()
                return result
            self._second_waiting.set()
        return await self._connection.execute(*args)


class _IdentityLockGatePool:
    def __init__(
        self,
        pool,
        *,
        lock_key: str,
        first_locked: asyncio.Event,
        second_waiting: asyncio.Event,
        release_first: asyncio.Event,
    ) -> None:
        self._pool = pool
        self._lock_key = lock_key
        self._first_locked = first_locked
        self._second_waiting = second_waiting
        self._release_first = release_first

    @asynccontextmanager
    async def transaction(self):
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield _IdentityLockGateConnection(
                    connection,
                    lock_key=self._lock_key,
                    first_locked=self._first_locked,
                    second_waiting=self._second_waiting,
                    release_first=self._release_first,
                )


@pytest.mark.asyncio
async def test_operator_contact_mutation_creates_replays_and_records_actor():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_create_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        operation_key = f"operator-create-{uuid.uuid4().hex}"
        command = EOMOperatorContactMutation.from_raw(
            operation_key=operation_key,
            actor_id=7,
            actor_name="Mayra Canfield",
            source_channel="time_tracker",
            source_ref="customer:88",
            contact_type="customer",
            fields={
                "full_name": "Operator Created",
                "email": "OPERATOR@EXAMPLE.COM",
                "phone": "(217) 555-0100",
                "address": "",
            },
        )

        created = await mutate_eom_operator_contact(provider, command)

        assert created["operation"] == "contact_created"
        assert created["idempotent"] is False
        contact_id = uuid.UUID(created["contact_id"])
        contact = await conn.fetchrow("SELECT * FROM contacts WHERE id = $1", contact_id)
        assert contact["business_context_id"] == "effingham_maids"
        assert contact["contact_type"] == "customer"
        assert contact["lead_stage"] is None
        assert contact["email"] == "operator@example.com"
        assert contact["phone"] == "2175550100"
        assert contact["address"] is None
        assert contact["source"] == "manual"
        assert contact["source_ref"] == "time_tracker:customer:88"
        event = await conn.fetchrow(
            """
            SELECT actor, source, operation_key, metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'contact_created'
            """,
            contact_id,
        )
        assert event["actor"] == "employee:7:Mayra Canfield"
        assert event["source"] == "eom_office"
        assert event["operation_key"] == operation_key
        metadata = _metadata_dict(event["metadata"])
        assert metadata["source_channel"] == "time_tracker"
        assert metadata["source_ref"] == "customer:88"
        assert metadata["field_names"] == [
            "address",
            "email",
            "full_name",
            "phone",
        ]

        replay = await mutate_eom_operator_contact(provider, command)

        assert replay["contact_id"] == str(contact_id)
        assert replay["idempotent"] is True
        assert await conn.fetchval("SELECT COUNT(*) FROM contacts") == 1
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE event_type = 'contact_created' AND operation_key = $1
            """,
            operation_key,
        ) == 1

        conflict = EOMOperatorContactMutation.from_raw(
            operation_key=operation_key,
            actor_id=7,
            actor_name="Mayra Canfield",
            source_channel="time_tracker",
            source_ref="customer:88",
            contact_type="customer",
            fields={
                "full_name": "Different Operator",
                "email": "operator@example.com",
            },
        )
        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, conflict)
        assert exc_info.value.status_code == 409
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_operator_contact_mutation_updates_exact_match_and_claims_legacy():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_update_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO contacts (
                id, full_name, phone, business_context_id, contact_type, status, source
            ) VALUES ($1, 'Legacy Match', '1 (217) 555-0100', NULL, 'customer', 'active', 'legacy')
            """,
            contact_id,
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-update-{uuid.uuid4().hex}",
            actor_id=8,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:44",
            fields={
                "phone": "217-555-0100",
                "email": "LEGACY.AFTER@EXAMPLE.COM",
                "notes": "",
            },
        )

        updated = await mutate_eom_operator_contact(provider, command)

        assert updated["operation"] == "contact_updated"
        assert updated["idempotent"] is False
        assert updated["contact_id"] == str(contact_id)
        row = await conn.fetchrow("SELECT * FROM contacts WHERE id = $1", contact_id)
        assert row["business_context_id"] == "effingham_maids"
        assert row["email"] == "legacy.after@example.com"
        assert row["phone"] == "2175550100"
        assert row["notes"] is None
        assert row["source"] == "legacy"
        metadata = _metadata_dict(row["metadata"])
        assert metadata["eom_operator_contact_sources"] == {
            "time_tracker:customer:44": {
                "source": "manual",
                "source_channel": "time_tracker",
                "source_ref": "customer:44",
            }
        }
        event = await conn.fetchrow(
            """
            SELECT actor, metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'contact_updated'
            """,
            contact_id,
        )
        assert event["actor"] == "employee:8:Juan Canfield"
        metadata = _metadata_dict(event["metadata"])
        assert metadata["field_names"] == ["email", "notes", "phone"]
        assert metadata["changed_fields"] == ["email", "phone"]
        assert metadata["source_ref"] == "customer:44"

        source_ref_only = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-source-ref-only-{uuid.uuid4().hex}",
            actor_id=8,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:44",
            fields={"full_name": "Legacy Match Renamed"},
        )

        rediscovered = await mutate_eom_operator_contact(provider, source_ref_only)

        assert rediscovered["operation"] == "contact_updated"
        assert rediscovered["contact_id"] == str(contact_id)
        assert await conn.fetchval("SELECT COUNT(*) FROM contacts") == 1
        assert await conn.fetchval(
            "SELECT full_name FROM contacts WHERE id = $1", contact_id
        ) == "Legacy Match Renamed"
        assert await conn.fetchval("SELECT COUNT(*) FROM contacts") == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_inbound_atomic_uses_ascii_phone_normalizer_for_relay_identity():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_inbound_unicode_phone_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)

        result = await resolve_or_create_eom_inbound_lead(
            provider,
            full_name="Unicode Phone",
            phone="٢١٧٥٥٥٠١٠٠",
            email=None,
            address=None,
            source="web",
            source_ref="submitted-unicode-phone",
            relay_event_id="relay-unicode-phone",
        )

        assert result["_was_created"] is True
        row = await conn.fetchrow("SELECT phone, source, source_ref FROM contacts")
        assert row["phone"] is None
        assert row["source"] == "web"
        assert row["source_ref"] == "relay-unicode-phone"
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_inbound_delivery_receipts
            WHERE source = 'web' AND delivery_id = 'relay-unicode-phone'
            """
        ) == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_operator_and_inbound_contact_writers_share_phone_identity_lock():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_cross_writer_lock_{uuid.uuid4().hex}"
    setup = await asyncpg.connect(database_url)
    pool = None
    release_first = asyncio.Event()
    try:
        await _prepare_schema(setup, schema, apply_privilege_migration=False)
        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=2,
            server_settings={"search_path": f'"{schema}", public'},
        )
        first_locked = asyncio.Event()
        second_waiting = asyncio.Event()
        provider = DatabaseCRMProvider(
            pool=_IdentityLockGatePool(
                pool,
                lock_key=_eom_identity_lock_key("phone", "2175550198"),
                first_locked=first_locked,
                second_waiting=second_waiting,
                release_first=release_first,
            )
        )

        async def operator_create():
            command = EOMOperatorContactMutation.from_raw(
                operation_key=f"operator-cross-writer-{uuid.uuid4().hex}",
                actor_id=8,
                actor_name="Juan Canfield",
                source_channel="time_tracker",
                source_ref="customer:98",
                fields={
                    "full_name": "Cross Writer Operator",
                    "phone": "217-555-0198",
                },
            )
            return await mutate_eom_operator_contact(provider, command)

        async def inbound_create():
            return await resolve_or_create_eom_inbound_lead(
                provider,
                full_name="Cross Writer Inbound",
                phone="(217) 555-0198",
                email=None,
                address=None,
                source="web",
                source_ref="web-cross-writer",
            )

        operator_task = asyncio.create_task(operator_create())
        await asyncio.wait_for(first_locked.wait(), timeout=2)
        inbound_task = asyncio.create_task(inbound_create())
        await asyncio.wait_for(second_waiting.wait(), timeout=2)
        release_first.set()
        operator_result, inbound_result = await asyncio.gather(
            operator_task,
            inbound_task,
        )

        assert operator_result["contact_id"] == str(inbound_result["id"])
        async with pool.acquire() as check:
            assert await check.fetchval("SELECT COUNT(*) FROM contacts") == 1
            assert await check.fetchval(
                """
                SELECT COUNT(*)
                FROM contacts
                WHERE RIGHT(REGEXP_REPLACE(COALESCE(phone, ''), '[^0-9]', '', 'g'), 10)
                    = '2175550198'
                """
            ) == 1
    finally:
        release_first.set()
        if pool is not None:
            await pool.close()
        await setup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await setup.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "claimed_value", "original_value"),
    (
        ("phone", "217-555-0197", "217-555-0196"),
        ("email", "owned@example.com", "target@example.com"),
    ),
)
async def test_operator_contact_mutation_rejects_explicit_id_identity_collision(
    field: str,
    claimed_value: str,
    original_value: str,
):
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_identity_collision_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        target_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=target_id,
            full_name="Target Contact",
            phone=original_value if field == "phone" else "2175550196",
            email=original_value if field == "email" else "target@example.com",
        )
        await _insert_contact(
            conn,
            contact_id=owner_id,
            full_name="Identity Owner",
            phone=claimed_value if field == "phone" else "2175550197",
            email=claimed_value if field == "email" else "owned@example.com",
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-explicit-collision-{uuid.uuid4().hex}",
            actor_id=8,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref=f"customer:collision-{field}",
            contact_id=str(target_id),
            fields={field: claimed_value},
        )

        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, command)

        assert exc_info.value.status_code == 409
        target = await conn.fetchrow("SELECT phone, email FROM contacts WHERE id = $1", target_id)
        owner = await conn.fetchrow("SELECT phone, email FROM contacts WHERE id = $1", owner_id)
        assert target[field] == original_value
        assert owner[field] == claimed_value
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE event_type = 'contact_updated'
            """
        ) == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_operator_contact_mutation_rejects_ambiguous_exact_identity():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_ambiguous_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        await _insert_contact(
            conn,
            contact_id=uuid.uuid4(),
            full_name="First Exact Phone",
            phone="12175550100",
            contact_type="customer",
            lead_stage=None,
        )
        await _insert_contact(
            conn,
            contact_id=uuid.uuid4(),
            full_name="Second Exact Phone",
            phone="2175550100",
            contact_type="customer",
            lead_stage=None,
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-ambiguous-{uuid.uuid4().hex}",
            actor_id=9,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:45",
            fields={"phone": "(217) 555-0100", "email": "new@example.com"},
        )

        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, command)

        assert exc_info.value.status_code == 409
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE event_type IN ('contact_created', 'contact_updated')
            """
        ) == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_operator_contact_mutation_rejects_unsupported_existing_contact_type():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_bad_type_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=contact_id,
            business_context_id=None,
            full_name="Legacy Vendor",
            contact_type="vendor",
            lead_stage=None,
            phone="2175550100",
            email="vendor@example.com",
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-bad-type-{uuid.uuid4().hex}",
            actor_id=10,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:46",
            contact_id=str(contact_id),
            fields={"phone": "217-555-0100", "notes": "not an EOM customer"},
        )

        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, command)

        assert exc_info.value.status_code == 409
        row = await conn.fetchrow("SELECT * FROM contacts WHERE id = $1", contact_id)
        assert row["business_context_id"] is None
        assert row["contact_type"] == "vendor"
        assert row["phone"] == "2175550100"
        assert row["notes"] is None
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE event_type IN ('contact_created', 'contact_updated')
            """
        ) == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_eom_lead_review_projection_is_closed_filtered_and_read_only():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_lead_review_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        eligible_id = uuid.uuid4()
        newer_eligible_id = uuid.uuid4()
        created_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        newer_created_at = datetime(2026, 7, 27, 13, 0, tzinfo=timezone.utc)
        await _insert_contact(
            conn,
            contact_id=eligible_id,
            full_name="Eligible Earlier",
            email="earlier@example.com",
            phone="2175550100",
            address="100 Main St",
            source="web",
            created_at=created_at,
            lead_stage="estimate_booked",
        )
        await _insert_contact(
            conn,
            contact_id=newer_eligible_id,
            full_name="Eligible Newer",
            email="newer@example.com",
            phone="2175550101",
            address="101 Main St",
            source="web",
            created_at=newer_created_at,
        )
        await conn.execute(
            """
            INSERT INTO contact_interactions (
                id, contact_id, interaction_type, summary, intent, occurred_at, metadata
            ) VALUES (
                $1, $2, 'web_form', 'older callback', 'estimate_request', $3,
                '{"submitted_email":"old-callback@example.com","submitted_phone":"2175550102"}'::jsonb
            )
            """,
            uuid.uuid4(),
            newer_eligible_id,
            datetime(2026, 7, 27, 12, 30, tzinfo=timezone.utc),
        )
        await conn.execute(
            """
            INSERT INTO contact_interactions (
                id, contact_id, interaction_type, summary, intent, occurred_at, metadata
            ) VALUES (
                $1, $2, 'web_form', 'latest callback', 'estimate_request', $3,
                '{"submitted_email":"latest-callback@example.com","submitted_phone":"2175550199"}'::jsonb
            )
            """,
            uuid.uuid4(),
            newer_eligible_id,
            datetime(2026, 7, 27, 13, 30, tzinfo=timezone.utc),
        )
        for business_context_id, contact_type, lead_stage, status in (
            ("other_business", "lead", "new", "active"),
            ("effingham_maids", "customer", None, "active"),
            ("effingham_maids", "lead", "qualified", "active"),
            ("effingham_maids", "lead", "new", "inactive"),
        ):
            await _insert_contact(
                conn,
                contact_id=uuid.uuid4(),
                business_context_id=business_context_id,
                contact_type=contact_type,
                lead_stage=lead_stage,
                status=status,
            )

        before_counts = {
            "contacts": await conn.fetchval("SELECT COUNT(*) FROM contacts"),
            "events": await conn.fetchval(
                "SELECT COUNT(*) FROM eom_lead_lifecycle_events"
            ),
            "handoffs": await conn.fetchval(
                "SELECT COUNT(*) FROM eom_customer_handoffs"
            ),
        }
        rows = await provider.list_eom_new_lead_review_items(limit=10)
        second_page = await provider.list_eom_new_lead_review_items(
            limit=1,
            cursor_created_at=newer_created_at,
            cursor_contact_id=newer_eligible_id,
        )

        assert rows == [
            {
                "contact_id": newer_eligible_id,
                "full_name": "Eligible Newer",
                "email": "latest-callback@example.com",
                "phone": "2175550199",
                "address": "101 Main St",
                "source": "web",
                "lead_stage": "new",
                "created_at": newer_created_at,
            },
            {
                "contact_id": eligible_id,
                "full_name": "Eligible Earlier",
                "email": "earlier@example.com",
                "phone": "2175550100",
                "address": "100 Main St",
                "source": "web",
                "lead_stage": "estimate_booked",
                "created_at": created_at,
            },
        ]
        assert second_page == [
            {
                "contact_id": eligible_id,
                "full_name": "Eligible Earlier",
                "email": "earlier@example.com",
                "phone": "2175550100",
                "address": "100 Main St",
                "source": "web",
                "lead_stage": "estimate_booked",
                "created_at": created_at,
            }
        ]
        assert set(rows[0]) == {
            "contact_id",
            "full_name",
            "email",
            "phone",
            "address",
            "source",
            "lead_stage",
            "created_at",
        }
        assert {
            "contacts": await conn.fetchval("SELECT COUNT(*) FROM contacts"),
            "events": await conn.fetchval(
                "SELECT COUNT(*) FROM eom_lead_lifecycle_events"
            ),
            "handoffs": await conn.fetchval(
                "SELECT COUNT(*) FROM eom_customer_handoffs"
            ),
        } == before_counts
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_intake_and_handoff_share_the_authoritative_postgres_provider():
    """A real web lead can be finalized without copying it to another database."""
    from atlas_brain.api.leads import LeadIntakeRequest, _process_lead_intake

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_intake_handoff_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)

        async def no_prior_submissions(_email: str, _phone: str) -> int:
            return 0

        intake = await _process_lead_intake(
            LeadIntakeRequest(
                name="Approved Estimate",
                phone="2175550100",
                service="Recurring residential cleaning",
                frequency="Every two weeks",
                source_page="/request-an-estimate",
            ),
            crm=provider,
            email_provider=object(),
            daily_count=no_prior_submissions,
        )
        assert intake == {"success": True, "email_sent": False}

        contact = await conn.fetchrow(
            """
            SELECT id, business_context_id, contact_type, lead_stage, status
            FROM contacts
            WHERE phone = $1
            """,
            "2175550100",
        )
        assert contact is not None
        assert {
            key: contact[key]
            for key in (
                "business_context_id",
                "contact_type",
                "lead_stage",
                "status",
            )
        } == {
            "business_context_id": "effingham_maids",
            "contact_type": "lead",
            "lead_stage": "new",
            "status": "active",
        }
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM contact_interactions WHERE contact_id = $1",
                contact["id"],
            )
            == 1
        )

        handoff = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact["id"]),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=_approval_key(),
            actor_id=1,
            actor_name="Juan Canfield",
        )

        assert handoff["idempotent"] is False
        assert await _contact_state(conn, contact["id"]) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
                "lead_stage": None,
                "status": "active",
            },
            1,
            1,
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_estimate_booking_lifecycle_is_idempotent_and_keeps_lead_approvable():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_estimate_booking_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        booking_key = f"office-booking-{uuid.uuid4().hex}"
        calendar_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=booking_key,
        )
        start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        await _insert_contact(
            conn,
            contact_id=contact_id,
            full_name="Booked Estimate",
            address="100 Main St",
        )

        prepared = await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        requested_metadata = await conn.fetchval(
            """
            SELECT metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'estimate_booking_requested'
              AND operation_key = $2
            """,
            contact_id,
            booking_key,
        )
        requested_metadata = _metadata_dict(requested_metadata)
        assert requested_metadata["calendar_event"] == {
            "summary": "Estimate: Booked Estimate",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "location": "100 Main St",
            "description": (
                "Scheduled from the private EOM lead funnel.\n\n"
                "Bring estimate worksheet"
            ),
            "calendar_id": "estimate-calendar",
            "event_id": calendar_event_id,
        }
        await conn.execute(
            """
            UPDATE contacts
            SET full_name = 'Edited Before Retry',
                address = '999 Changed Ave'
            WHERE id = $1
            """,
            contact_id,
        )
        pending_replay = await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        contact_after_prepare = await conn.fetchrow(
            "SELECT contact_type, lead_stage FROM contacts WHERE id = $1",
            contact_id,
        )
        completed = await provider.complete_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        replay = await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        lifecycle_counts = {
            row["event_type"]: int(row["count"])
            for row in await conn.fetch(
                """
                SELECT event_type, COUNT(*) AS count
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type IN ('estimate_booking_requested', 'estimate_booked')
                GROUP BY event_type
                """,
                contact_id,
            )
        }

        assert prepared["idempotent"] is False
        assert prepared["status"] == "calendar_pending"
        assert pending_replay["idempotent"] is True
        assert pending_replay["status"] == "calendar_pending"
        assert pending_replay["calendar_event"]["summary"] == "Estimate: Booked Estimate"
        assert pending_replay["calendar_event"]["location"] == "100 Main St"
        assert dict(contact_after_prepare) == {
            "contact_type": "lead",
            "lead_stage": "new",
        }
        assert completed["idempotent"] is False
        assert completed["status"] == "estimate_booked"
        assert completed["calendar_event_id"] == calendar_event_id
        assert replay["idempotent"] is True
        assert replay["status"] == "estimate_booked"
        assert replay["calendar_event"]["summary"] == "Estimate: Booked Estimate"
        await provider.mark_eom_estimate_booking_calendar_ambiguous(
            contact_id=str(contact_id),
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            observed_calendar_event_id="",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert (
            await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND operation_key = $2
                  AND event_type = 'estimate_booking_calendar_ambiguous'
                """,
                contact_id,
                booking_key,
            )
            == 0
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor,
                source, operation_key, metadata
            )
            VALUES ($1::uuid, 'estimate_booking_calendar_ambiguous', 'new', 'new',
                    'employee:1:Juan Canfield', 'eom_office', $2::varchar,
                    jsonb_build_object(
                        'expected_calendar_event_id', $3::text,
                        'observed_calendar_event_id', ''
                    ))
            """,
            contact_id,
            booking_key,
            calendar_event_id,
        )
        replay_with_historical_ambiguity = await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert replay_with_historical_ambiguity["idempotent"] is True
        assert replay_with_historical_ambiguity["status"] == "estimate_booked"
        assert lifecycle_counts == {
            "estimate_booking_requested": 1,
            "estimate_booked": 1,
        }

        handoff = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=_approval_key(),
            actor_id=1,
            actor_name="Juan Canfield",
        )
        approval_from_stage = await conn.fetchval(
            """
            SELECT from_stage
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'customer_approved'
            """,
            contact_id,
        )

        assert handoff["idempotent"] is False
        assert approval_from_stage == "estimate_booked"
        assert await _contact_state(conn, contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
                "lead_stage": None,
                "status": "active",
            },
            1,
            1,
        )
        post_handoff_replay = await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert post_handoff_replay["idempotent"] is True
        assert post_handoff_replay["status"] == "estimate_booked"
        assert post_handoff_replay["calendar_event"]["summary"] == (
            "Estimate: Booked Estimate"
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_estimate_booking_rejects_conflicting_replay_before_new_lifecycle_event():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_estimate_booking_conflict_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        booking_key = f"office-booking-{uuid.uuid4().hex}"
        calendar_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=booking_key,
        )
        start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        await _insert_contact(conn, contact_id=contact_id)

        await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        with pytest.raises(
            EOMLeadConversionError,
            match="different estimate booking",
        ):
            await provider.prepare_eom_estimate_booking(
                contact_id=str(contact_id),
                scheduled_start=start,
                scheduled_end=end,
                calendar_id="second-calendar",
                notes="Bring estimate worksheet",
                booking_key=booking_key,
                expected_calendar_event_id=calendar_event_id,
                actor_id=1,
                actor_name="Juan Canfield",
            )
        with pytest.raises(
            EOMLeadConversionError,
            match="different estimate booking",
        ):
            different_booking_key = f"office-booking-{uuid.uuid4().hex}"
            await provider.prepare_eom_estimate_booking(
                contact_id=str(contact_id),
                scheduled_start=start,
                scheduled_end=end,
                calendar_id="estimate-calendar",
                notes="Bring estimate worksheet",
                booking_key=different_booking_key,
                expected_calendar_event_id=deterministic_eom_estimate_calendar_event_id(
                    contact_id=str(contact_id),
                    booking_key=different_booking_key,
                ),
                actor_id=1,
                actor_name="Juan Canfield",
            )

        assert (
            await conn.fetchval(
                """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'estimate_booking_requested'
            """,
                contact_id,
            )
            == 1
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_pending_estimate_booking_blocks_handoff_until_terminal_event():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_estimate_booking_handoff_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        failed_key = f"office-booking-{uuid.uuid4().hex}"
        corrected_key = f"office-booking-{uuid.uuid4().hex}"
        start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        failed_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=failed_key,
        )
        corrected_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=corrected_key,
        )
        await _insert_contact(conn, contact_id=contact_id)

        await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="mistyped-calendar",
            notes="Bring estimate worksheet",
            booking_key=failed_key,
            expected_calendar_event_id=failed_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        with pytest.raises(EOMLeadConversionError, match="still pending"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            )

        await provider.mark_eom_estimate_booking_calendar_failed(
            contact_id=str(contact_id),
            booking_key=failed_key,
            expected_calendar_event_id=failed_event_id,
            calendar_error="API_ERROR",
            calendar_message="Calendar API error: 404",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        corrected = await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=corrected_key,
            expected_calendar_event_id=corrected_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        lifecycle_counts = {
            row["event_type"]: int(row["count"])
            for row in await conn.fetch(
                """
                SELECT event_type, COUNT(*) AS count
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type IN (
                      'estimate_booking_requested',
                      'estimate_booking_calendar_failed'
                  )
                GROUP BY event_type
                """,
                contact_id,
            )
        }

        assert corrected["idempotent"] is False
        assert corrected["status"] == "calendar_pending"
        assert lifecycle_counts == {
            "estimate_booking_requested": 2,
            "estimate_booking_calendar_failed": 1,
        }
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_handoff_stays_fenced_while_a_same_key_booking_is_executing():
    """A terminal failed marker alone must not admit handoff mid-execution.

    The booking executor holds a session advisory lock on
    eom-estimate-booking:execution:<key> for its whole prepare -> Calendar ->
    complete span. While a concurrent same-key retry is still talking to
    Calendar, the ledger's failed marker is not yet the settled outcome, so
    finalize must stay fenced until the executor releases the lock.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_estimate_booking_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    executor_conn = None
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        booking_key = f"office-booking-{uuid.uuid4().hex}"
        execution_lock_key = f"eom-estimate-booking:execution:{booking_key}"
        calendar_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=booking_key,
        )
        start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        await _insert_contact(conn, contact_id=contact_id)

        await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await provider.mark_eom_estimate_booking_calendar_failed(
            contact_id=str(contact_id),
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            calendar_error="API_ERROR",
            calendar_message="Calendar API error: 404",
            actor_id=1,
            actor_name="Juan Canfield",
        )

        executor_conn = await asyncpg.connect(database_url)
        assert await executor_conn.fetchval(
            "SELECT pg_try_advisory_lock(hashtextextended($1, 0))",
            execution_lock_key,
        )

        with pytest.raises(EOMLeadConversionError, match="still executing"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            )

        with pytest.raises(EOMLeadConversionError, match="already executing"):
            async with provider.eom_estimate_booking_execution_lock(
                booking_key=booking_key
            ):
                pass  # pragma: no cover - the lock must refuse entry

        assert await _contact_state(conn, contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "lead",
                "lead_stage": "new",
                "status": "active",
            },
            0,
            0,
        )

        assert await executor_conn.fetchval(
            "SELECT pg_advisory_unlock(hashtextextended($1, 0))",
            execution_lock_key,
        )

        settled = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=_approval_key(),
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert settled["idempotent"] is False
        assert await _contact_state(conn, contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
                "lead_stage": None,
                "status": "active",
            },
            1,
            1,
        )
    finally:
        if executor_conn is not None:
            await executor_conn.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_booking_completion_settles_after_mid_execution_status_flip():
    """NocoDB holds an UPDATE (status) grant, so an operator can archive the
    lead while the Calendar call is in flight. Admission was validated at
    prepare time and the Calendar event exists, so completion must still
    record the booked outcome; the active-status admission lives downstream
    in review and handoff, which must keep rejecting the inactive contact.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_estimate_booking_status_flip_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        booking_key = f"office-booking-{uuid.uuid4().hex}"
        calendar_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=booking_key,
        )
        start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        await _insert_contact(conn, contact_id=contact_id)

        await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await conn.execute(
            "UPDATE contacts SET status = 'inactive' WHERE id = $1",
            contact_id,
        )

        completed = await provider.complete_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        assert completed["idempotent"] is False
        assert completed["status"] == "estimate_booked"
        contact = await conn.fetchrow(
            "SELECT lead_stage, status FROM contacts WHERE id = $1", contact_id
        )
        assert dict(contact) == {"lead_stage": "estimate_booked", "status": "inactive"}
        assert (
            await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND operation_key = $2
                  AND event_type = 'estimate_booked'
                """,
                contact_id,
                booking_key,
            )
            == 1
        )
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM contacts WHERE id = $1 AND id IN ("
                "SELECT contact_id FROM eom_lead_lifecycle_events)",
                contact_id,
            )
            == 1
        )

        with pytest.raises(EOMLeadConversionError, match="must be active"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            )
        assert await provider.list_eom_new_lead_review_items(limit=10) == []
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_estimate_booking_completion_dominates_failed_and_ambiguous_markers():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_estimate_booking_precedence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        booking_key = f"office-booking-{uuid.uuid4().hex}"
        calendar_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=booking_key,
        )
        start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        await _insert_contact(conn, contact_id=contact_id)

        await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await provider.mark_eom_estimate_booking_calendar_failed(
            contact_id=str(contact_id),
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            calendar_error="API_ERROR",
            calendar_message="Calendar API error: 404",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await provider.mark_eom_estimate_booking_calendar_ambiguous(
            contact_id=str(contact_id),
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            observed_calendar_event_id="",
            actor_id=1,
            actor_name="Juan Canfield",
        )

        with pytest.raises(EOMLeadConversionError, match="calendar reconciliation"):
            await provider.prepare_eom_estimate_booking(
                contact_id=str(contact_id),
                scheduled_start=start,
                scheduled_end=end,
                calendar_id="estimate-calendar",
                notes="Bring estimate worksheet",
                booking_key=booking_key,
                expected_calendar_event_id=calendar_event_id,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        completed = await provider.complete_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await provider.mark_eom_estimate_booking_calendar_failed(
            contact_id=str(contact_id),
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            calendar_error="API_ERROR",
            calendar_message="Calendar API error: 404",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        replay = await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=calendar_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        lifecycle_counts = {
            row["event_type"]: int(row["count"])
            for row in await conn.fetch(
                """
                SELECT event_type, COUNT(*) AS count
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND operation_key = $2
                  AND event_type IN (
                      'estimate_booking_calendar_failed',
                      'estimate_booking_calendar_ambiguous',
                      'estimate_booked'
                  )
                GROUP BY event_type
                """,
                contact_id,
                booking_key,
            )
        }

        assert completed["status"] == "estimate_booked"
        assert replay["idempotent"] is True
        assert replay["status"] == "estimate_booked"
        assert lifecycle_counts == {
            "estimate_booking_calendar_failed": 1,
            "estimate_booking_calendar_ambiguous": 1,
            "estimate_booked": 1,
        }
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_estimate_booking_key_is_owned_across_contacts():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_estimate_booking_key_owner_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        first_contact_id = uuid.uuid4()
        second_contact_id = uuid.uuid4()
        booking_key = f"office-booking-{uuid.uuid4().hex}"
        start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        await _insert_contact(conn, contact_id=first_contact_id)
        await _insert_contact(conn, contact_id=second_contact_id)

        await provider.prepare_eom_estimate_booking(
            contact_id=str(first_contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=booking_key,
            expected_calendar_event_id=deterministic_eom_estimate_calendar_event_id(
                contact_id=str(first_contact_id),
                booking_key=booking_key,
            ),
            actor_id=1,
            actor_name="Juan Canfield",
        )

        with pytest.raises(EOMLeadConversionError, match="different EOM lead"):
            await provider.prepare_eom_estimate_booking(
                contact_id=str(second_contact_id),
                scheduled_start=start,
                scheduled_end=end,
                calendar_id="estimate-calendar",
                notes="Bring estimate worksheet",
                booking_key=booking_key,
                expected_calendar_event_id=deterministic_eom_estimate_calendar_event_id(
                    contact_id=str(second_contact_id),
                    booking_key=booking_key,
                ),
                actor_id=1,
                actor_name="Juan Canfield",
            )

        assert (
            await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = 'estimate_booking_requested'
                """,
                second_contact_id,
            )
            == 0
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_office_handoff_is_atomic_idempotent_and_keeps_rate_schedule_out_of_atlas():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)

        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        approval_key = _approval_key()
        different_approval_key = _approval_key()
        await _insert_contact(conn, contact_id=contact_id)

        first = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=approval_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        retry = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=approval_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        assert first["idempotent"] is False
        assert retry == {**first, "idempotent": True}
        contact = await conn.fetchrow(
            "SELECT contact_type, lead_stage FROM contacts WHERE id = $1", contact_id
        )
        assert dict(contact) == {"contact_type": "customer", "lead_stage": None}
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )
        assert (
            await conn.fetchval(
                """
            SELECT COUNT(*) FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'customer_approved'
            """,
                contact_id,
            )
            == 1
        )

        with pytest.raises(asyncpg.exceptions.RaiseError, match="immutable"):
            await conn.execute(
                """
                UPDATE eom_customer_handoffs
                SET tracker_site_id = 999
                WHERE contact_id = $1
                """,
                contact_id,
            )
        assert (
            await conn.fetchval(
                "SELECT tracker_site_id FROM eom_customer_handoffs WHERE contact_id = $1",
                contact_id,
            )
            == 202
        )

        with pytest.raises(asyncpg.exceptions.RaiseError, match="immutable"):
            await conn.execute(
                "DELETE FROM eom_customer_handoffs WHERE contact_id = $1",
                contact_id,
            )
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )

        with pytest.raises(asyncpg.exceptions.RaiseError, match="immutable"):
            await conn.execute("TRUNCATE TABLE eom_customer_handoffs")
        assert await conn.fetchval("SELECT COUNT(*) FROM eom_customer_handoffs") == 1

        direct_contact_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=direct_contact_id)
        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="matching customer transition and lifecycle evidence",
        ):
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('atlas.eom_customer_handoff_finalization', 'true', true)"
                )
                await conn.execute(
                    """
                    INSERT INTO eom_customer_handoffs (
                        contact_id, approval_key, tracker_customer_id, tracker_site_id,
                        approved_by_employee_id, approved_by_name
                    )
                    VALUES ($1, $2, 303, 404, 1, 'Juan Canfield')
                    """,
                    direct_contact_id,
                    _approval_key(),
                )
        assert await _contact_state(conn, direct_contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "lead",
                "lead_stage": "new",
                "status": "active",
            },
            0,
            0,
        )

        with pytest.raises(EOMLeadConversionError, match="different customer handoff"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=different_approval_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_office_handoff_rejects_an_incomplete_preexisting_replay_row():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_incomplete_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)

        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        approval_key = _approval_key()
        await _insert_contact(conn, contact_id=contact_id)
        # Only the guard-object owner can model malformed legacy data. The
        # runtime and NocoDB roles cannot disable this trigger.
        await conn.execute("SET ROLE atlas_eom_handoff_owner")
        await conn.execute(
            "ALTER TABLE eom_customer_handoffs "
            "DISABLE TRIGGER trg_require_eom_customer_handoff_finalization"
        )
        try:
            await conn.execute(
                """
                INSERT INTO eom_customer_handoffs (
                    contact_id, approval_key, tracker_customer_id, tracker_site_id,
                    approved_by_employee_id, approved_by_name
                )
                VALUES ($1, $2, 101, 202, 1, 'Juan Canfield')
                """,
                contact_id,
                approval_key,
            )
        finally:
            await conn.execute(
                "ALTER TABLE eom_customer_handoffs "
                "ENABLE TRIGGER trg_require_eom_customer_handoff_finalization"
            )
            await conn.execute("RESET ROLE")

        with pytest.raises(
            EOMLeadConversionError, match="not a completed finalization"
        ):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=approval_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        assert await _contact_state(conn, contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "lead",
                "lead_stage": "new",
                "status": "active",
            },
            0,
            1,
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_nocodb_role_cannot_bypass_handoff_or_lifecycle_guards():
    """The real NocoDB login has only the documented CRM-table capability."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_nocodb_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    nocodb_conn = None
    try:
        await _prepare_schema(conn, schema)
        assert not await conn.fetchval(
            "SELECT pg_has_role('atlas_nocodb', 'atlas_eom_handoff_owner', 'MEMBER')"
        )
        assert not await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS nocodb_role ON nocodb_role.oid = membership.member
                WHERE nocodb_role.rolname = 'atlas_nocodb'
            )
            """)

        nocodb_conn = await asyncpg.connect(
            database_url,
            user="atlas_nocodb",
            password=_NOCODB_TEST_PASSWORD,
        )
        try:
            await nocodb_conn.execute(
                f"SET search_path TO {_quote_ident(schema)}, public"
            )
            assert await nocodb_conn.fetchval("SELECT current_user") == "atlas_nocodb"
            blocked_statements = (
                "CREATE TABLE nocodb_bypass (id INTEGER)",
                "ALTER TABLE eom_customer_handoffs DISABLE TRIGGER ALL",
                "INSERT INTO eom_customer_handoffs DEFAULT VALUES",
                "UPDATE eom_customer_handoffs SET approved_by_name = 'Tampered'",
                "DELETE FROM eom_customer_handoffs",
                "INSERT INTO eom_lead_lifecycle_events DEFAULT VALUES",
                "SELECT * FROM schema_migrations",
                "INSERT INTO schema_migrations (version, name) VALUES (999, 'tampered')",
                "SELECT * FROM api_keys",
                "INSERT INTO api_keys (id) VALUES (gen_random_uuid())",
                "SELECT * FROM byok_keys",
                "INSERT INTO byok_keys (id) VALUES (gen_random_uuid())",
                "SELECT * FROM scoped_mailbox_credentials",
                "INSERT INTO scoped_mailbox_credentials (id) VALUES (gen_random_uuid())",
            )
            for statement in blocked_statements:
                with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
                    await nocodb_conn.execute(statement)

            protected_contact_mutations = (
                "UPDATE contacts SET contact_type = 'customer'",
                "UPDATE contacts SET contact_type = 'lead'",
                "UPDATE contacts SET lead_stage = 'qualified'",
                "UPDATE contacts SET lead_stage = NULL",
                "UPDATE contacts SET business_context_id = 'effingham_maids'",
                "UPDATE contacts SET business_context_id = NULL",
                "UPDATE contacts SET contact_type = 'customer', lead_stage = NULL",
                "INSERT INTO contacts (id, full_name, contact_type) "
                "VALUES (gen_random_uuid(), 'Direct customer', 'customer')",
                "INSERT INTO contacts (id, full_name, lead_stage) "
                "VALUES (gen_random_uuid(), 'Direct lead', 'new')",
                "INSERT INTO contacts (id, full_name, business_context_id) "
                "VALUES (gen_random_uuid(), 'Direct EOM', 'effingham_maids')",
            )
            for statement in protected_contact_mutations:
                with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
                    await nocodb_conn.execute(statement)

            contact_id = uuid.uuid4()
            await nocodb_conn.execute(
                "INSERT INTO contacts (id, full_name, notes) VALUES ($1, 'NocoDB CRM', 'before')",
                contact_id,
            )
            await nocodb_conn.execute(
                "UPDATE contacts SET notes = 'ordinary edit' WHERE id = $1",
                contact_id,
            )
            assert (
                await nocodb_conn.fetchval(
                    "SELECT notes FROM contacts WHERE id = $1", contact_id
                )
                == "ordinary edit"
            )
            await nocodb_conn.execute(
                """
                INSERT INTO contact_interactions (id, contact_id, interaction_type)
                VALUES ($1, $2, 'note')
                """,
                uuid.uuid4(),
                contact_id,
            )
            appointment_id = uuid.uuid4()
            await nocodb_conn.execute(
                "INSERT INTO appointments (id) VALUES ($1)", appointment_id
            )
            assert await nocodb_conn.fetchval(
                "SELECT id = $1 FROM appointments WHERE id = $1", appointment_id
            )
        finally:
            await nocodb_conn.close()
            nocodb_conn = None
    finally:
        if nocodb_conn is not None:
            await nocodb_conn.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_privilege_migration_rolls_back_if_ledger_recording_fails():
    """354 cannot revoke its executor's recovery path before it is recorded."""
    from atlas_brain.storage.migrations import run_migrations

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_privilege_rollback_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _require_disposable_role_administration(conn)

        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        runtime_role = await conn.fetchval("SELECT current_user")
        runtime_ident = _quote_ident(runtime_role)
        await _provision_handoff_guard(conn)
        await _provision_nocodb_login(conn)
        await conn.execute(
            f"GRANT atlas_eom_handoff_owner TO {runtime_ident} WITH ADMIN OPTION"
        )
        await conn.execute("""
            CREATE FUNCTION fail_eom_privilege_bookkeeping()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.name = '354_eom_customer_handoff_privileges' THEN
                    RAISE EXCEPTION 'injected privilege ledger failure';
                END IF;
                RETURN NEW;
            END;
            $$;
            CREATE TRIGGER trg_fail_eom_privilege_bookkeeping
            BEFORE INSERT ON schema_migrations
            FOR EACH ROW
            EXECUTE FUNCTION fail_eom_privilege_bookkeeping();
            """)

        class _SchemaPool:
            async def acquire(self):
                return conn

            async def release(self, released):
                assert released is conn

        with pytest.raises(
            asyncpg.exceptions.RaiseError, match="injected privilege ledger failure"
        ):
            await run_migrations(
                _SchemaPool(),
                migrations_dir=MIGRATIONS,
                only={"354_eom_customer_handoff_privileges"},
            )

        assert await conn.fetchval("""
            SELECT pg_get_userbyid(relowner) = current_user
            FROM pg_class
            WHERE oid = 'eom_customer_handoffs'::regclass
            """)
        assert await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = current_user
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """)
        assert not await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM schema_migrations
                WHERE name = '354_eom_customer_handoff_privileges'
            )
            """)

        await conn.execute(
            "DROP TRIGGER trg_fail_eom_privilege_bookkeeping ON schema_migrations"
        )
        await conn.execute("DROP FUNCTION fail_eom_privilege_bookkeeping()")
        await run_migrations(
            _SchemaPool(),
            migrations_dir=MIGRATIONS,
            only={"354_eom_customer_handoff_privileges"},
        )

        assert await conn.fetchval("""
            SELECT pg_get_userbyid(relowner) = 'atlas_eom_handoff_owner'
            FROM pg_class
            WHERE oid = 'eom_customer_handoffs'::regclass
            """)
        assert await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = current_user
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """)
        await conn.execute(
            f"REVOKE atlas_eom_handoff_owner FROM {runtime_ident} CASCADE"
        )
        assert not await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = current_user
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """)
        assert await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM schema_migrations
                WHERE name = '354_eom_customer_handoff_privileges'
            )
            """)
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_privilege_migration_runs_from_a_real_non_superuser_login(monkeypatch):
    """Ownership transfer works without relying on the administrator session."""
    from atlas_brain import main
    from atlas_brain.storage.migrations import run_migrations

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_non_super_{uuid.uuid4().hex}"
    executor_role = f"atlas_eom_migrator_{uuid.uuid4().hex}"
    schema_ident = _quote_ident(schema)
    executor_ident = _quote_ident(executor_role)
    conn = await asyncpg.connect(database_url)
    executor_conn = None
    try:
        await _require_disposable_role_administration(conn)
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        await _provision_handoff_guard(conn)
        await _provision_nocodb_login(conn)

        database_name = await conn.fetchval("SELECT current_database()")
        await conn.execute(
            f"CREATE ROLE {executor_ident} LOGIN NOINHERIT CREATEROLE "
            f"PASSWORD '{_NON_SUPERUSER_TEST_PASSWORD}'"
        )
        await conn.execute(
            f"GRANT CONNECT ON DATABASE {_quote_ident(database_name)} TO {executor_ident}"
        )
        await conn.execute(
            f"GRANT atlas_eom_handoff_owner TO {executor_ident} WITH ADMIN OPTION"
        )
        await conn.execute(f"ALTER SCHEMA {schema_ident} OWNER TO {executor_ident}")
        table_rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = $1", schema
        )
        for table_row in table_rows:
            await conn.execute(
                f"ALTER TABLE {schema_ident}.{_quote_ident(table_row['tablename'])} "
                f"OWNER TO {executor_ident}"
            )
        await conn.execute(
            "ALTER FUNCTION "
            f"{schema_ident}.require_eom_customer_handoff_finalization() "
            f"OWNER TO {executor_ident}"
        )
        await conn.execute(
            "ALTER FUNCTION "
            f"{schema_ident}.prevent_eom_customer_handoff_mutation() "
            f"OWNER TO {executor_ident}"
        )

        executor_conn = await asyncpg.connect(
            database_url,
            user=executor_role,
            password=_NON_SUPERUSER_TEST_PASSWORD,
        )
        await executor_conn.execute(f"SET search_path TO {schema_ident}, public")
        assert not await executor_conn.fetchval(
            "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
        )
        assert await executor_conn.fetchval(
            "SELECT rolcreaterole FROM pg_roles WHERE rolname = current_user"
        )

        class _SchemaPool:
            async def acquire(self):
                return executor_conn

            async def release(self, released):
                assert released is executor_conn

        await run_migrations(
            _SchemaPool(),
            migrations_dir=MIGRATIONS,
            only={"354_eom_customer_handoff_privileges"},
        )

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await executor_conn.fetchval(query))

        monkeypatch.setattr(main, "get_db_pool", lambda: _Pool())
        assert await executor_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = current_user
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """)
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await main._require_eom_funnel_data_store(
                type("Config", (), {"api_enabled": True})(),
                database_enabled=True,
            )
        await executor_conn.close()
        executor_conn = None

        await conn.execute(
            f"REVOKE atlas_eom_handoff_owner FROM {executor_ident} CASCADE"
        )
        assert await conn.fetchval(
            """
            SELECT NOT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = $1
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """,
            executor_role,
        )
        assert await conn.fetchval(
            "SELECT has_schema_privilege('atlas_eom_handoff_owner', $1, 'CREATE')",
            schema,
        )

        verifier_conn = await asyncpg.connect(
            database_url,
            user=executor_role,
            password=_NON_SUPERUSER_TEST_PASSWORD,
        )
        try:
            await verifier_conn.execute(f"SET search_path TO {schema_ident}, public")

            class _VerifiedPool:
                is_initialized = True

                async def fetchval(self, query: str) -> bool:
                    return bool(await verifier_conn.fetchval(query))

            monkeypatch.setattr(main, "get_db_pool", lambda: _VerifiedPool())
            await main._require_eom_funnel_data_store(
                type("Config", (), {"api_enabled": True})(),
                database_enabled=True,
            )

            # #2286: the readiness guard above does not check the runtime's own
            # privileges, so it passed even though 354's pre-transfer self-grant
            # left the runtime with nothing on eom_customer_handoffs. The app
            # finalizes handoffs with a direct INSERT as this login, so assert
            # the DML survives the ownership transfer AND the membership revoke.
            for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
                assert await verifier_conn.fetchval(
                    "SELECT has_table_privilege("
                    "current_user, 'eom_customer_handoffs', $1)",
                    privilege,
                ), (
                    f"runtime login lost {privilege} on eom_customer_handoffs "
                    "after ownership transfer + membership revoke (#2286)"
                )
        finally:
            await verifier_conn.close()

        assert await conn.fetchval(
            """
            SELECT pg_get_userbyid(relowner) = 'atlas_eom_handoff_owner'
            FROM pg_class
            WHERE oid = $1::regclass
            """,
            f"{schema}.eom_customer_handoffs",
        )
    finally:
        if executor_conn is not None:
            await executor_conn.close()
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema_ident} CASCADE")
        database_name = await conn.fetchval("SELECT current_database()")
        await conn.execute(
            f"REVOKE CONNECT ON DATABASE {_quote_ident(database_name)} FROM {executor_ident}"
        )
        await conn.execute(f"REVOKE atlas_eom_handoff_owner FROM {executor_ident}")
        await conn.execute(f"DROP ROLE IF EXISTS {executor_ident}")
        await conn.close()


@pytest.mark.asyncio
async def test_privilege_migration_satisfies_the_enabled_full_app_startup_guard(
    monkeypatch,
):
    """The live preflight query accepts only the protected role arrangement."""
    from atlas_brain import main

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_preflight_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await conn.fetchval(query))

        monkeypatch.setattr(main, "get_db_pool", lambda: _Pool())
        await main._require_eom_funnel_data_store(
            type("Config", (), {"api_enabled": True})(),
            database_enabled=True,
        )

        await conn.execute("ALTER ROLE atlas_nocodb INHERIT")
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await main._require_eom_funnel_data_store(
                type("Config", (), {"api_enabled": True})(),
                database_enabled=True,
            )
        await conn.execute("ALTER ROLE atlas_nocodb NOINHERIT")
        await conn.execute("GRANT atlas_eom_handoff_owner TO atlas_nocodb")
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await main._require_eom_funnel_data_store(
                type("Config", (), {"api_enabled": True})(),
                database_enabled=True,
            )
        await conn.execute("REVOKE atlas_eom_handoff_owner FROM atlas_nocodb")
        await main._require_eom_funnel_data_store(
            type("Config", (), {"api_enabled": True})(),
            database_enabled=True,
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_relation",
    (
        "contacts",
        "eom_lead_lifecycle_events",
        "eom_customer_handoffs",
        "eom_onboarding_email_drafts",
    ),
)
async def test_enabled_shared_guard_returns_controlled_error_when_required_relation_is_absent(
    missing_relation: str,
):
    """A fresh or partial database must fail readiness without relation-lookup errors."""
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_missing_relation_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        await _provision_handoff_guard(conn)
        await _provision_nocodb_login(conn)
        await conn.execute(f"DROP TABLE {_quote_ident(missing_relation)} CASCADE")
        assert await conn.fetchval(
            "SELECT to_regclass($1::text) IS NULL",
            missing_relation,
        )

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await conn.fetchval(query))

        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                type("Config", (), {"api_enabled": True})(),
                database_enabled=True,
                get_db_pool_fn=lambda: _Pool(),
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_enabled_shared_guard_requires_bigint_draft_approver_column():
    """A canonical store with migration 360 but not 361 must fail readiness.

    The funnel actor boundary admits signed-64 ids; against the original
    INTEGER approver column a valid approval would pass HTTP and then fail
    in Postgres after the claim, so the slim funnel refuses to serve until
    the widening is applied.
    """
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_draft_actor_width_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await conn.fetchval(query))

        config = type("Config", (), {"api_enabled": True})()
        await require_eom_funnel_data_store(
            config, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )

        await conn.execute(
            "ALTER TABLE eom_onboarding_email_drafts "
            "ALTER COLUMN approved_by_employee_id TYPE INTEGER"
        )
        with pytest.raises(
            RuntimeError, match="CRM lifecycle and handoff schema"
        ):
            await require_eom_funnel_data_store(
                config, database_enabled=True, get_db_pool_fn=lambda: _Pool()
            )

        await conn.execute(
            "ALTER TABLE eom_onboarding_email_drafts "
            "ALTER COLUMN approved_by_employee_id TYPE BIGINT"
        )
        await require_eom_funnel_data_store(
            config, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_enabled_shared_guard_requires_lifecycle_sequence_column():
    """A canonical store with migration 362 but not 363 must fail readiness."""
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_lifecycle_sequence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await conn.fetchval(query))

        config = type("Config", (), {"api_enabled": True})()
        await require_eom_funnel_data_store(
            config, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )

        await conn.execute(
            "ALTER TABLE eom_lead_lifecycle_events DROP COLUMN lifecycle_sequence"
        )
        with pytest.raises(
            RuntimeError, match="CRM lifecycle and handoff schema"
        ):
            await require_eom_funnel_data_store(
                config, database_enabled=True, get_db_pool_fn=lambda: _Pool()
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_column",
    ("business_context_id", "contact_type", "lead_stage"),
)
async def test_enabled_shared_guard_handles_missing_required_contact_column(
    missing_column: str,
):
    """A partial contacts migration must fail readiness without undefined-column errors."""
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_missing_column_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        await _provision_handoff_guard(conn)
        await _provision_nocodb_login(conn)
        await conn.execute(
            f"ALTER TABLE contacts DROP COLUMN {_quote_ident(missing_column)}"
        )
        assert not await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_attribute
                WHERE attrelid = 'contacts'::regclass
                  AND attname = $1
                  AND NOT attisdropped
            )
            """,
            missing_column,
        )

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await conn.fetchval(query))

        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                type("Config", (), {"api_enabled": True})(),
                database_enabled=True,
                get_db_pool_fn=lambda: _Pool(),
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("business_context_id", "contact_type", "lead_stage", "status", "expected_status"),
    (
        ("other_business", "lead", "new", "active", 404),
        ("effingham_maids", "lead", "new", "inactive", 409),
        ("effingham_maids", "customer", None, "active", 409),
        ("effingham_maids", "lead", "qualified", "active", 409),
    ),
)
async def test_office_handoff_rejects_ineligible_contact_before_any_handoff_write(
    business_context_id: str,
    contact_type: str,
    lead_stage: str | None,
    status: str,
    expected_status: int,
):
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_reject_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)

        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=contact_id,
            business_context_id=business_context_id,
            contact_type=contact_type,
            lead_stage=lead_stage,
            status=status,
        )
        before = await _contact_state(conn, contact_id)

        with pytest.raises(EOMLeadConversionError) as exc_info:
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            )

        assert exc_info.value.status_code == expected_status
        assert await _contact_state(conn, contact_id) == before
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("later_status", ("inactive", "archived"))
async def test_completed_office_handoff_replays_after_later_customer_status_change(
    later_status: str,
):
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_replay_status_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        approval_key = _approval_key()
        await _insert_contact(conn, contact_id=contact_id)

        first = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=approval_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        updated = await provider.update_contact(
            str(contact_id),
            {"status": later_status},
        )
        assert updated is not None
        assert updated["status"] == later_status

        replay = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=approval_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        assert replay == {**first, "idempotent": True}
        assert await _contact_state(conn, contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
                "lead_stage": None,
                "status": later_status,
            },
            1,
            1,
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_office_handoff_serializes_overlapping_key_and_contact_callbacks():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_conversion_concurrent_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    pool = None
    try:
        await _prepare_schema(conn, schema)
        await conn.execute("""
            CREATE OR REPLACE FUNCTION delay_eom_customer_handoff_insert()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$
            BEGIN
                PERFORM pg_sleep(0.05);
                RETURN NEW;
            END;
            $$;
            CREATE TRIGGER trg_delay_eom_customer_handoff_insert
            BEFORE INSERT ON eom_customer_handoffs
            FOR EACH ROW EXECUTE FUNCTION delay_eom_customer_handoff_insert();
            """)
        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=2,
            server_settings={"search_path": f"{schema}, public"},
        )

        provider = DatabaseCRMProvider(pool=pool)
        first_contact_id = uuid.uuid4()
        second_contact_id = uuid.uuid4()
        third_contact_id = uuid.uuid4()
        fourth_contact_id = uuid.uuid4()
        fifth_contact_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=first_contact_id)
        await _insert_contact(conn, contact_id=second_contact_id)
        await _insert_contact(conn, contact_id=third_contact_id)
        await _insert_contact(conn, contact_id=fourth_contact_id)
        await _insert_contact(conn, contact_id=fifth_contact_id)

        same_key = _approval_key()

        async def finalize(
            contact_id: uuid.UUID,
            key: str,
            tracker_customer_id: int = 101,
            tracker_site_id: int = 202,
        ):
            return await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=tracker_customer_id,
                tracker_site_id=tracker_site_id,
                approval_key=key,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        first, retry = await asyncio.gather(
            finalize(first_contact_id, same_key),
            finalize(first_contact_id, same_key),
        )
        assert {first["idempotent"], retry["idempotent"]} == {False, True}
        assert first["handoff_id"] == retry["handoff_id"]

        cross_contact_key = _approval_key()
        cross_results = await asyncio.gather(
            finalize(second_contact_id, cross_contact_key, 303, 404),
            finalize(third_contact_id, cross_contact_key, 303, 404),
            return_exceptions=True,
        )
        successes = [result for result in cross_results if isinstance(result, dict)]
        rejections = [
            result
            for result in cross_results
            if isinstance(result, EOMLeadConversionError)
        ]
        assert len(successes) == 1
        assert len(rejections) == 1
        assert rejections[0].status_code == 409
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM eom_customer_handoffs WHERE approval_key = $1",
                cross_contact_key,
            )
            == 1
        )

        tracker_collision_results = await asyncio.gather(
            finalize(fourth_contact_id, _approval_key(), 505, 606),
            finalize(fifth_contact_id, _approval_key(), 505, 606),
            return_exceptions=True,
        )
        tracker_successes = [
            result for result in tracker_collision_results if isinstance(result, dict)
        ]
        tracker_rejections = [
            result
            for result in tracker_collision_results
            if isinstance(result, EOMLeadConversionError)
        ]
        assert len(tracker_successes) == 1
        assert len(tracker_rejections) == 1
        assert tracker_rejections[0].status_code == 409
        assert await conn.fetchval("""
            SELECT COUNT(*) FROM eom_customer_handoffs
            WHERE tracker_customer_id = 505 AND tracker_site_id = 606
            """) == 1
        assert (
            await conn.fetchval(
                """
            SELECT COUNT(*) FROM contacts
            WHERE id = ANY($1::uuid[]) AND contact_type = 'customer'
            """,
                [second_contact_id, third_contact_id],
            )
            == 1
        )
    finally:
        if pool is not None:
            await pool.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()

@pytest.mark.asyncio
async def test_first_clean_booking_lifecycle_promotes_to_won_with_pending_draft():
    """The normal path: estimate booked, then the first cleaning is booked.

    Completion must move the lead to won and enqueue exactly one pending
    onboarding draft in the same transaction; prepare/complete replays must
    stay idempotent and report the same draft; the won lead must remain in
    the review queue and hand off with from_stage='won'.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        estimate_key = f"office-booking-{uuid.uuid4().hex}"
        first_clean_key = f"office-first-clean-{uuid.uuid4().hex}"
        estimate_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=estimate_key,
        )
        first_clean_event_id = deterministic_eom_first_clean_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=first_clean_key,
        )
        estimate_start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        estimate_end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        clean_start = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
        clean_end = datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc)
        await _insert_contact(
            conn,
            contact_id=contact_id,
            full_name="Won Lead",
            email="won-lead@example.com",
            address="100 Main St",
        )

        await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=estimate_start,
            scheduled_end=estimate_end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=estimate_key,
            expected_calendar_event_id=estimate_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await provider.complete_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=estimate_start,
            scheduled_end=estimate_end,
            calendar_id="estimate-calendar",
            notes="Bring estimate worksheet",
            booking_key=estimate_key,
            expected_calendar_event_id=estimate_event_id,
            calendar_event_id=estimate_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        prepared = await provider.prepare_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=clean_start,
            scheduled_end=clean_end,
            calendar_id="estimate-calendar",
            notes="First clean crew notes",
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert prepared["idempotent"] is False
        assert prepared["status"] == "calendar_pending"
        requested_metadata = _metadata_dict(
            await conn.fetchval(
                """
                SELECT metadata
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = 'first_clean_booking_requested'
                  AND operation_key = $2
                """,
                contact_id,
                first_clean_key,
            )
        )
        assert requested_metadata["calendar_event"]["summary"] == (
            "First clean: Won Lead"
        )
        assert requested_metadata["calendar_event"]["event_id"] == (
            first_clean_event_id
        )

        completed = await provider.complete_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=clean_start,
            scheduled_end=clean_end,
            calendar_id="estimate-calendar",
            notes="First clean crew notes",
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert completed["idempotent"] is False
        assert completed["status"] == "first_clean_booked"
        assert completed["lead_stage"] == "won"
        draft_id = completed["onboarding_draft_id"]
        assert draft_id

        draft = await conn.fetchrow(
            "SELECT * FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            draft_id,
        )
        assert draft["contact_id"] == contact_id
        assert draft["operation_key"] == first_clean_key
        assert draft["status"] == "pending"
        assert draft["recipient_email"] == "won-lead@example.com"
        assert draft["blocker"] is None
        assert draft["subject"].strip()
        assert "Won Lead" in draft["body"]
        assert draft["sent_at"] is None

        booked_event = await conn.fetchrow(
            """
            SELECT from_stage, to_stage
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'first_clean_booked'
              AND operation_key = $2
            """,
            contact_id,
            first_clean_key,
        )
        assert dict(booked_event) == {
            "from_stage": "estimate_booked",
            "to_stage": "won",
        }
        contact = await conn.fetchrow(
            "SELECT contact_type, lead_stage FROM contacts WHERE id = $1",
            contact_id,
        )
        assert dict(contact) == {"contact_type": "lead", "lead_stage": "won"}

        prepare_replay = await provider.prepare_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=clean_start,
            scheduled_end=clean_end,
            calendar_id="estimate-calendar",
            notes="First clean crew notes",
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert prepare_replay["idempotent"] is True
        assert prepare_replay["status"] == "first_clean_booked"
        assert prepare_replay["onboarding_draft_id"] == draft_id

        complete_replay = await provider.complete_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=clean_start,
            scheduled_end=clean_end,
            calendar_id="estimate-calendar",
            notes="First clean crew notes",
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert complete_replay["idempotent"] is True
        assert complete_replay["onboarding_draft_id"] == draft_id
        assert (
            await conn.fetchval(
                """
                SELECT COUNT(*) FROM eom_onboarding_email_drafts
                WHERE contact_id = $1
                """,
                contact_id,
            )
            == 1
        )

        review_rows = await provider.list_eom_new_lead_review_items(limit=10)
        assert [(row["contact_id"], row["lead_stage"]) for row in review_rows] == [
            (contact_id, "won")
        ]

        handoff = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=_approval_key(),
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert handoff["idempotent"] is False
        approval_from_stage = await conn.fetchval(
            """
            SELECT from_stage
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'customer_approved'
            """,
            contact_id,
        )
        assert approval_from_stage == "won"
        assert await _contact_state(conn, contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
                "lead_stage": None,
                "status": "active",
            },
            1,
            1,
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_first_clean_draft_enqueue_records_no_email_blocker_and_one_pending_row():
    """A contact without an email still gets a draft (blocker='no_email'),
    and the partial unique index refuses a second pending draft per contact."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_draft_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        first_clean_key = f"office-first-clean-{uuid.uuid4().hex}"
        first_clean_event_id = deterministic_eom_first_clean_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=first_clean_key,
        )
        start = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc)
        await _insert_contact(
            conn,
            contact_id=contact_id,
            full_name="No Email Lead",
            email=None,
        )

        await provider.prepare_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        completed = await provider.complete_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        draft = await conn.fetchrow(
            "SELECT * FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            completed["onboarding_draft_id"],
        )
        assert draft["recipient_email"] is None
        assert draft["blocker"] == "no_email"
        assert draft["status"] == "pending"

        # The migration-360 claim predicate refuses blocked rows outright: a
        # draft with no usable recipient can never be claimed into sending.
        blocked_claim = await conn.fetchrow(
            """
            UPDATE eom_onboarding_email_drafts
               SET status = 'sending', claimed_at = NOW(),
                   approved_by_employee_id = $2, approved_by_name = $3
             WHERE id = $1::uuid
               AND status = 'pending'
               AND blocker IS NULL
               AND recipient_email IS NOT NULL
             RETURNING id
            """,
            draft["id"],
            1,
            "Juan Canfield",
        )
        assert blocked_claim is None

        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """
                INSERT INTO eom_onboarding_email_drafts (
                    contact_id, operation_key, subject, body
                )
                VALUES ($1, $2, 'dup subject', 'dup body')
                """,
                contact_id,
                f"office-first-clean-{uuid.uuid4().hex}",
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_first_clean_draft_recipient_follows_latest_intake_projection():
    """Ingress leaves contacts.email unchanged when an existing contact
    re-submits with a new address (the new address lives in the web_form
    interaction metadata), so the draft recipient must resolve through the
    same latest-intake projection the office review queue shows -- not the
    stale contact column."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_recipient_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        first_clean_key = f"office-first-clean-{uuid.uuid4().hex}"
        first_clean_event_id = deterministic_eom_first_clean_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=first_clean_key,
        )
        start = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc)
        await _insert_contact(
            conn,
            contact_id=contact_id,
            full_name="Re-Submitting Lead",
            email="stale-original@example.com",
        )
        # The same ingress-shaped interaction rows the review projection
        # reads: an older re-submission and the latest one.
        await conn.execute(
            """
            INSERT INTO contact_interactions (
                id, contact_id, interaction_type, summary, intent, occurred_at, metadata
            ) VALUES (
                $1, $2, 'web_form', 'older re-submission', 'estimate_request', $3,
                '{"submitted_email":"older-address@example.com","submitted_phone":"2175550102"}'::jsonb
            )
            """,
            uuid.uuid4(),
            contact_id,
            datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
        )
        await conn.execute(
            """
            INSERT INTO contact_interactions (
                id, contact_id, interaction_type, summary, intent, occurred_at, metadata
            ) VALUES (
                $1, $2, 'web_form', 'latest re-submission', 'estimate_request', $3,
                '{"submitted_email":"latest-address@example.com","submitted_phone":"2175550199"}'::jsonb
            )
            """,
            uuid.uuid4(),
            contact_id,
            datetime(2026, 8, 10, 13, 0, tzinfo=timezone.utc),
        )

        review_rows = await provider.list_eom_new_lead_review_items(limit=10)
        assert review_rows[0]["email"] == "latest-address@example.com"

        await provider.prepare_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        completed = await provider.complete_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        draft = await conn.fetchrow(
            "SELECT recipient_email, blocker FROM eom_onboarding_email_drafts "
            "WHERE id = $1::uuid",
            completed["onboarding_draft_id"],
        )
        assert draft["recipient_email"] == "latest-address@example.com"
        assert draft["blocker"] is None
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_onboarding_draft_claim_contract_wins_exactly_once_under_two_sessions():
    """The A3 single-send contract documented in migration 360: two sessions
    racing UPDATE ... WHERE status = 'pending' RETURNING settle to exactly
    one winner; the loser gets zero rows, never a second send."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_draft_claim_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    claimer_conn = None
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        first_clean_key = f"office-first-clean-{uuid.uuid4().hex}"
        first_clean_event_id = deterministic_eom_first_clean_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=first_clean_key,
        )
        start = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc)
        await _insert_contact(
            conn,
            contact_id=contact_id,
            email="claim-race@example.com",
        )
        await provider.prepare_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        completed = await provider.complete_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=first_clean_key,
            expected_calendar_event_id=first_clean_event_id,
            calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        draft_id = completed["onboarding_draft_id"]

        claimer_conn = await asyncpg.connect(database_url)
        await claimer_conn.execute(f'SET search_path TO "{schema}", public')
        claim_sql = """
            UPDATE eom_onboarding_email_drafts
               SET status = 'sending', claimed_at = NOW(),
                   approved_by_employee_id = $2, approved_by_name = $3
             WHERE id = $1::uuid
               AND status = 'pending'
               AND blocker IS NULL
               AND recipient_email IS NOT NULL
             RETURNING id
        """
        first_claim, second_claim = await asyncio.gather(
            conn.fetchrow(claim_sql, draft_id, 1, "Juan Canfield"),
            claimer_conn.fetchrow(claim_sql, draft_id, 2, "Mayra Canfield"),
        )
        winners = [row for row in (first_claim, second_claim) if row is not None]
        assert len(winners) == 1

        claimed = await conn.fetchrow(
            "SELECT status, claimed_at, sent_at FROM eom_onboarding_email_drafts "
            "WHERE id = $1::uuid",
            draft_id,
        )
        assert claimed["status"] == "sending"
        assert claimed["claimed_at"] is not None
        assert claimed["sent_at"] is None
        late_claim = await conn.fetchrow(claim_sql, draft_id, 3, "Tina Gomez")
        assert late_claim is None

        # Delivery is confirmed separately, only after transport acceptance.
        confirmed = await conn.fetchrow(
            """
            UPDATE eom_onboarding_email_drafts
               SET status = 'sent', sent_at = NOW()
             WHERE id = $1::uuid AND status = 'sending'
             RETURNING status, sent_at
            """,
            draft_id,
        )
        assert confirmed["status"] == "sent"
        assert confirmed["sent_at"] is not None
        reconfirm = await conn.fetchrow(
            """
            UPDATE eom_onboarding_email_drafts
               SET status = 'sent', sent_at = NOW()
             WHERE id = $1::uuid AND status = 'sending'
             RETURNING id
            """,
            draft_id,
        )
        assert reconfirm is None
    finally:
        if claimer_conn is not None:
            await claimer_conn.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_handoff_stays_fenced_while_a_first_clean_booking_is_executing():
    """The handoff fence covers the first-clean family through the shared
    eom-estimate-booking:execution:<key> namespace: while a first-clean
    executor holds the lock, a terminal failed marker alone must not admit
    handoff."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    executor_conn = None
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        booking_key = f"office-first-clean-{uuid.uuid4().hex}"
        execution_lock_key = f"eom-estimate-booking:execution:{booking_key}"
        first_clean_event_id = deterministic_eom_first_clean_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=booking_key,
        )
        start = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc)
        await _insert_contact(conn, contact_id=contact_id)

        await provider.prepare_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=booking_key,
            expected_calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await provider.mark_eom_first_clean_booking_calendar_failed(
            contact_id=str(contact_id),
            booking_key=booking_key,
            expected_calendar_event_id=first_clean_event_id,
            calendar_error="API_ERROR",
            calendar_message="Calendar API error: 404",
            actor_id=1,
            actor_name="Juan Canfield",
        )

        executor_conn = await asyncpg.connect(database_url)
        assert await executor_conn.fetchval(
            "SELECT pg_try_advisory_lock(hashtextextended($1, 0))",
            execution_lock_key,
        )

        with pytest.raises(EOMLeadConversionError, match="still executing"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            )

        with pytest.raises(EOMLeadConversionError, match="already executing"):
            async with provider.eom_estimate_booking_execution_lock(
                booking_key=booking_key
            ):
                pass  # pragma: no cover - the lock must refuse entry

        assert await executor_conn.fetchval(
            "SELECT pg_advisory_unlock(hashtextextended($1, 0))",
            execution_lock_key,
        )

        settled = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=_approval_key(),
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert settled["idempotent"] is False
        assert await _contact_state(conn, contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
                "lead_stage": None,
                "status": "active",
            },
            1,
            1,
        )
    finally:
        if executor_conn is not None:
            await executor_conn.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_first_clean_completion_settles_after_mid_execution_status_flip():
    """An operator status flip mid-execution must not orphan the Calendar
    event: completion still records won and the pending draft; the inactive
    contact stays rejected downstream by review and handoff."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_status_flip_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        booking_key = f"office-first-clean-{uuid.uuid4().hex}"
        first_clean_event_id = deterministic_eom_first_clean_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=booking_key,
        )
        start = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc)
        await _insert_contact(
            conn, contact_id=contact_id, email="flip@example.com"
        )

        await provider.prepare_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=booking_key,
            expected_calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await conn.execute(
            "UPDATE contacts SET status = 'inactive' WHERE id = $1",
            contact_id,
        )

        completed = await provider.complete_eom_first_clean_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=booking_key,
            expected_calendar_event_id=first_clean_event_id,
            calendar_event_id=first_clean_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        assert completed["idempotent"] is False
        assert completed["status"] == "first_clean_booked"
        assert completed["onboarding_draft_id"]
        contact = await conn.fetchrow(
            "SELECT lead_stage, status FROM contacts WHERE id = $1", contact_id
        )
        assert dict(contact) == {"lead_stage": "won", "status": "inactive"}
        assert (
            await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND operation_key = $2
                  AND event_type = 'first_clean_booked'
                """,
                contact_id,
                booking_key,
            )
            == 1
        )
        assert (
            await conn.fetchval(
                "SELECT status FROM eom_onboarding_email_drafts "
                "WHERE operation_key = $1",
                booking_key,
            )
            == "pending"
        )

        with pytest.raises(EOMLeadConversionError, match="must be active"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            )
        assert await provider.list_eom_new_lead_review_items(limit=10) == []
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_booking_key_ownership_and_blocking_across_families():
    """A booking key belongs to one family forever; an unsettled operation of
    either family blocks the other; a COMPLETED estimate booking never blocks
    the first clean (that is the normal funnel path)."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_cross_family_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        estimate_key = f"office-booking-{uuid.uuid4().hex}"
        first_clean_key = f"office-first-clean-{uuid.uuid4().hex}"
        estimate_event_id = deterministic_eom_estimate_calendar_event_id(
            contact_id=str(contact_id),
            booking_key=estimate_key,
        )
        start = datetime(2026, 8, 4, 19, 0, tzinfo=timezone.utc)
        end = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
        await _insert_contact(
            conn, contact_id=contact_id, email="cross@example.com"
        )

        await provider.prepare_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=estimate_key,
            expected_calendar_event_id=estimate_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        def _first_clean_kwargs(key: str) -> dict:
            return dict(
                contact_id=str(contact_id),
                scheduled_start=start,
                scheduled_end=end,
                calendar_id="estimate-calendar",
                notes=None,
                booking_key=key,
                expected_calendar_event_id=(
                    deterministic_eom_first_clean_calendar_event_id(
                        contact_id=str(contact_id),
                        booking_key=key,
                    )
                ),
                actor_id=1,
                actor_name="Juan Canfield",
            )

        # The estimate family owns this key even before it settles.
        with pytest.raises(
            EOMLeadConversionError, match="different EOM booking"
        ):
            await provider.prepare_eom_first_clean_booking(
                **_first_clean_kwargs(estimate_key)
            )

        # An unsettled estimate operation blocks a first-clean request under
        # a fresh key.
        with pytest.raises(
            EOMLeadConversionError, match="another booking operation"
        ):
            await provider.prepare_eom_first_clean_booking(
                **_first_clean_kwargs(first_clean_key)
            )

        await provider.complete_eom_estimate_booking(
            contact_id=str(contact_id),
            scheduled_start=start,
            scheduled_end=end,
            calendar_id="estimate-calendar",
            notes=None,
            booking_key=estimate_key,
            expected_calendar_event_id=estimate_event_id,
            calendar_event_id=estimate_event_id,
            actor_id=1,
            actor_name="Juan Canfield",
        )

        # Key ownership is permanent after settlement too.
        with pytest.raises(
            EOMLeadConversionError, match="different EOM booking"
        ):
            await provider.prepare_eom_first_clean_booking(
                **_first_clean_kwargs(estimate_key)
            )

        # A reconciled estimate operation can legitimately carry BOTH a
        # historical ambiguous marker and the booked outcome (the A1
        # precedence ladder); the booked outcome must dominate that stale
        # ambiguity when the first-clean prepare scans other operations,
        # or the normal estimate -> first clean path wedges permanently.
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor,
                source, operation_key, metadata
            )
            VALUES ($1::uuid, 'estimate_booking_calendar_ambiguous',
                    'estimate_booked', 'estimate_booked',
                    'employee:1:Juan Canfield', 'eom_office', $2::varchar,
                    jsonb_build_object(
                        'expected_calendar_event_id', $3::text,
                        'observed_calendar_event_id', ''
                    ))
            """,
            contact_id,
            estimate_key,
            estimate_event_id,
        )

        # The settled estimate booking is the normal path into first clean.
        prepared = await provider.prepare_eom_first_clean_booking(
            **_first_clean_kwargs(first_clean_key)
        )
        assert prepared["idempotent"] is False
        assert prepared["status"] == "calendar_pending"

        # And the first-clean family owns its key against estimate reuse.
        with pytest.raises(
            EOMLeadConversionError, match="different EOM booking"
        ):
            await provider.prepare_eom_estimate_booking(
                contact_id=str(contact_id),
                scheduled_start=start,
                scheduled_end=end,
                calendar_id="estimate-calendar",
                notes=None,
                booking_key=first_clean_key,
                expected_calendar_event_id=(
                    deterministic_eom_estimate_calendar_event_id(
                        contact_id=str(contact_id),
                        booking_key=first_clean_key,
                    )
                ),
                actor_id=1,
                actor_name="Juan Canfield",
            )

        assert (
            await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = 'first_clean_booking_requested'
                """,
                contact_id,
            )
            == 1
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()

from atlas_brain.services.eom_onboarding_drafts import (  # noqa: E402
    EOMOnboardingDraftApproval,
    approve_and_send_eom_onboarding_draft,
)


class _RecordingDraftSender:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    async def __call__(self, *, to, subject, body, idempotency_key):
        self.calls.append(
            {
                "to": to,
                "subject": subject,
                "body": body,
                "idempotency_key": idempotency_key,
            }
        )
        if self.fail:
            raise RuntimeError("transport unavailable")
        return {"message_id": "resend-msg-1", "idempotent_replay": False}


async def _book_first_clean_draft(
    conn,
    provider,
    *,
    email: str | None = "won-lead@example.com",
    full_name: str = "Won Lead",
) -> tuple[uuid.UUID, str]:
    """Insert a lead and complete a first-clean booking; return its draft."""
    contact_id = uuid.uuid4()
    booking_key = f"office-first-clean-{uuid.uuid4().hex}"
    event_id = deterministic_eom_first_clean_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )
    start = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc)
    await _insert_contact(
        conn, contact_id=contact_id, email=email, full_name=full_name
    )
    await provider.prepare_eom_first_clean_booking(
        contact_id=str(contact_id),
        scheduled_start=start,
        scheduled_end=end,
        calendar_id="estimate-calendar",
        notes=None,
        booking_key=booking_key,
        expected_calendar_event_id=event_id,
        actor_id=1,
        actor_name="Juan Canfield",
    )
    completed = await provider.complete_eom_first_clean_booking(
        contact_id=str(contact_id),
        scheduled_start=start,
        scheduled_end=end,
        calendar_id="estimate-calendar",
        notes=None,
        booking_key=booking_key,
        expected_calendar_event_id=event_id,
        calendar_event_id=event_id,
        actor_id=1,
        actor_name="Juan Canfield",
    )
    return contact_id, str(completed["onboarding_draft_id"])


@pytest.mark.asyncio
async def test_onboarding_draft_approval_pipeline_edits_sends_and_replays():
    """Edit -> approve -> sent against real Postgres, then idempotent replay.

    The global pool is deliberately uninitialized here: the default
    evidence writers must bind to the CRM provider's own pool (the store
    that owns the draft), so the sent_emails history row and the CRM
    interaction land in this schema, not wherever the global pool points.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_draft_approval_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)

        edited = await provider.update_eom_onboarding_draft(
            draft_id=draft_id,
            subject="Welcome aboard, from the whole crew",
        )
        assert edited["subject"] == "Welcome aboard, from the whole crew"
        assert edited["status"] == "pending"

        sender = _RecordingDraftSender()
        result = await approve_and_send_eom_onboarding_draft(
            provider,
            EOMOnboardingDraftApproval(
                draft_id=draft_id, actor_id=7, actor_name="Mayra Canfield"
            ),
            sender=sender,
        )
        assert result["status"] == "sent"
        assert result["idempotent"] is False
        assert result["resend_message_id"] == "resend-msg-1"
        assert sender.calls == [
            {
                "to": "won-lead@example.com",
                "subject": "Welcome aboard, from the whole crew",
                "body": result["body"],
                "idempotency_key": f"eom-onboarding-draft:{draft_id}",
            }
        ]
        row = await conn.fetchrow(
            "SELECT status, sent_at, claimed_at, approved_by_employee_id, "
            "approved_by_name FROM eom_onboarding_email_drafts "
            "WHERE id = $1::uuid",
            draft_id,
        )
        assert row["status"] == "sent"
        assert row["sent_at"] is not None
        assert row["claimed_at"] is not None
        assert row["approved_by_employee_id"] == 7
        assert row["approved_by_name"] == "Mayra Canfield"

        evidence = await conn.fetchrow(
            "SELECT to_addresses, template_type, resend_message_id, "
            "business_context_id FROM sent_emails"
        )
        assert evidence is not None
        assert evidence["to_addresses"] == ["won-lead@example.com"]
        assert evidence["template_type"] == "onboarding_welcome"
        assert evidence["resend_message_id"] == "resend-msg-1"
        assert evidence["business_context_id"] == "effingham_maids"
        interaction = await conn.fetchrow(
            "SELECT interaction_type FROM contact_interactions "
            "WHERE contact_id = $1 AND interaction_type = 'email'",
            contact_id,
        )
        assert interaction is not None

        replay = await approve_and_send_eom_onboarding_draft(
            provider,
            EOMOnboardingDraftApproval(
                draft_id=draft_id, actor_id=7, actor_name="Mayra Canfield"
            ),
            sender=sender,
        )
        assert replay["idempotent"] is True
        assert replay["status"] == "sent"
        assert len(sender.calls) == 1  # no second transport call

        with pytest.raises(EOMLeadConversionError, match="only pending"):
            await provider.update_eom_onboarding_draft(
                draft_id=draft_id, subject="Too late"
            )
        with pytest.raises(EOMLeadConversionError, match="cannot be revoked"):
            await provider.revoke_eom_onboarding_draft(draft_id=draft_id)
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_onboarding_draft_claim_refuses_archived_contact():
    """Archiving the lead after its draft was enqueued blocks approval.

    The claim carries the same contact-activity admission as the booking
    family, so a pending draft for an archived contact answers 409 with
    zero state change, and restoring the contact makes the same draft
    claimable again.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_draft_archived_contact_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)

        await conn.execute(
            "UPDATE contacts SET status = 'archived' WHERE id = $1", contact_id
        )
        with pytest.raises(EOMLeadConversionError, match="not an active"):
            await provider.claim_eom_onboarding_draft(
                draft_id=draft_id, actor_id=7, actor_name="Mayra Canfield"
            )
        row = await conn.fetchrow(
            "SELECT status, claimed_at FROM eom_onboarding_email_drafts "
            "WHERE id = $1::uuid",
            draft_id,
        )
        assert row["status"] == "pending"
        assert row["claimed_at"] is None

        await conn.execute(
            "UPDATE contacts SET status = 'active' WHERE id = $1", contact_id
        )
        claim = await provider.claim_eom_onboarding_draft(
            draft_id=draft_id, actor_id=7, actor_name="Mayra Canfield"
        )
        assert claim["claimed"] is True
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_onboarding_draft_provider_claim_wins_exactly_once_under_two_sessions():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_draft_provider_claim_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    second_conn = None
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)

        second_conn = await asyncpg.connect(database_url)
        await second_conn.execute(f'SET search_path TO "{schema}", public')
        second_provider = DatabaseCRMProvider(pool=second_conn)

        first, second = await asyncio.gather(
            provider.claim_eom_onboarding_draft(
                draft_id=draft_id, actor_id=1, actor_name="Juan Canfield"
            ),
            second_provider.claim_eom_onboarding_draft(
                draft_id=draft_id, actor_id=2, actor_name="Mayra Canfield"
            ),
            return_exceptions=True,
        )
        outcomes = [first, second]
        winners = [
            result
            for result in outcomes
            if isinstance(result, dict) and result.get("claimed")
        ]
        losers = [
            result
            for result in outcomes
            if isinstance(result, EOMLeadConversionError)
        ]
        assert len(winners) == 1
        assert len(losers) == 1
        assert losers[0].status_code == 409
        assert (
            await conn.fetchval(
                "SELECT status FROM eom_onboarding_email_drafts "
                "WHERE id = $1::uuid",
                draft_id,
            )
            == "sending"
        )
    finally:
        if second_conn is not None:
            await second_conn.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_onboarding_draft_no_email_blocker_resolves_through_edit():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_draft_blocker_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(
            conn, provider, email=None, full_name="No Email Lead"
        )

        with pytest.raises(EOMLeadConversionError, match="blocked: no_email"):
            await provider.claim_eom_onboarding_draft(
                draft_id=draft_id, actor_id=1, actor_name="Juan Canfield"
            )

        fixed = await provider.update_eom_onboarding_draft(
            draft_id=draft_id, recipient_email="found-address@example.com"
        )
        assert fixed["recipient_email"] == "found-address@example.com"
        assert fixed["blocker"] is None

        sender = _RecordingDraftSender()
        result = await approve_and_send_eom_onboarding_draft(
            provider,
            EOMOnboardingDraftApproval(
                draft_id=draft_id, actor_id=1, actor_name="Juan Canfield"
            ),
            sender=sender,
        )
        assert result["status"] == "sent"
        assert sender.calls[0]["to"] == "found-address@example.com"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_onboarding_draft_stuck_sending_reconciles_by_confirm_or_revoke():
    """Migration 360 step 4 against real Postgres: a transport failure
    leaves 'sending'; the operator either confirms sent or revokes, and a
    revoke that raced the send makes the later confirm fail loudly."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_draft_reconcile_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)

        # Draft A: transport fails mid-approve -> stuck sending -> confirm.
        _, draft_a = await _book_first_clean_draft(conn, provider)
        failing = _RecordingDraftSender(fail=True)
        from atlas_brain.services.eom_onboarding_drafts import (
            EOMOnboardingDraftError,
        )

        with pytest.raises(EOMOnboardingDraftError) as excinfo:
            await approve_and_send_eom_onboarding_draft(
                provider,
                EOMOnboardingDraftApproval(
                    draft_id=draft_a, actor_id=1, actor_name="Juan Canfield"
                ),
                sender=failing,
            )
        assert excinfo.value.status_code == 502
        assert (
            await conn.fetchval(
                "SELECT status FROM eom_onboarding_email_drafts "
                "WHERE id = $1::uuid",
                draft_a,
            )
            == "sending"
        )
        # While the claim is FRESH the send may still be mid-flight:
        # operator reconciliation (confirm-sent and revoke alike) refuses.
        with pytest.raises(EOMLeadConversionError, match="still in flight"):
            await provider.confirm_eom_onboarding_draft_sent(
                draft_id=draft_a, require_stale=True
            )
        with pytest.raises(EOMLeadConversionError, match="still in flight"):
            await provider.revoke_eom_onboarding_draft(draft_id=draft_a)
        await conn.execute(
            "UPDATE eom_onboarding_email_drafts "
            "SET claimed_at = NOW() - INTERVAL '20 minutes' "
            "WHERE id = $1::uuid",
            draft_a,
        )
        confirmed = await provider.confirm_eom_onboarding_draft_sent(
            draft_id=draft_a, require_stale=True
        )
        assert confirmed["status"] == "sent"
        assert confirmed["idempotent"] is False
        # Crash-recovery deliveries record sent-email history in the
        # provider's own store, with a null transport id (never observed).
        from atlas_brain.services.eom_onboarding_drafts import (
            record_operator_confirmed_send_evidence,
        )

        await record_operator_confirmed_send_evidence(provider, confirmed)
        recovery_evidence = await conn.fetchrow(
            "SELECT to_addresses, resend_message_id, template_type "
            "FROM sent_emails"
        )
        assert recovery_evidence is not None
        assert recovery_evidence["to_addresses"] == ["won-lead@example.com"]
        assert recovery_evidence["resend_message_id"] is None
        assert recovery_evidence["template_type"] == "onboarding_welcome"
        replay = await provider.confirm_eom_onboarding_draft_sent(
            draft_id=draft_a, require_stale=True
        )
        assert replay["idempotent"] is True

        # Draft B: STALE sending -> revoked -> a late zombie confirm from
        # the in-flow path (require_stale=False) fails loudly instead of
        # re-recording delivery, and further claims stay refused.
        _, draft_b = await _book_first_clean_draft(conn, provider)
        await provider.claim_eom_onboarding_draft(
            draft_id=draft_b, actor_id=1, actor_name="Juan Canfield"
        )
        await conn.execute(
            "UPDATE eom_onboarding_email_drafts "
            "SET claimed_at = NOW() - INTERVAL '20 minutes' "
            "WHERE id = $1::uuid",
            draft_b,
        )
        revoked = await provider.revoke_eom_onboarding_draft(draft_id=draft_b)
        assert revoked["status"] == "revoked"
        with pytest.raises(EOMLeadConversionError, match="revoked while sending"):
            await provider.confirm_eom_onboarding_draft_sent(draft_id=draft_b)
        with pytest.raises(EOMLeadConversionError, match="revoked"):
            await provider.claim_eom_onboarding_draft(
                draft_id=draft_b, actor_id=1, actor_name="Juan Canfield"
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_onboarding_draft_list_projection_filters_and_paginates():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_draft_list_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)
        contact_a, draft_a = await _book_first_clean_draft(
            conn, provider, full_name="Lead Alpha"
        )
        contact_b, draft_b = await _book_first_clean_draft(
            conn, provider, full_name="Lead Beta"
        )

        pending = await provider.list_eom_onboarding_drafts(status="pending")
        assert {str(row["draft_id"]) for row in pending} == {draft_a, draft_b}
        assert {row["full_name"] for row in pending} == {
            "Lead Alpha",
            "Lead Beta",
        }

        first_page = await provider.list_eom_onboarding_drafts(
            status="pending", limit=1
        )
        assert len(first_page) == 1
        second_page = await provider.list_eom_onboarding_drafts(
            status="pending",
            limit=2,
            cursor_created_at=first_page[0]["created_at"],
            cursor_draft_id=first_page[0]["draft_id"],
        )
        assert len(second_page) == 1
        assert second_page[0]["draft_id"] != first_page[0]["draft_id"]

        await provider.claim_eom_onboarding_draft(
            draft_id=draft_a, actor_id=1, actor_name="Juan Canfield"
        )
        assert {
            str(row["draft_id"])
            for row in await provider.list_eom_onboarding_drafts(status="pending")
        } == {draft_b}
        sending = await provider.list_eom_onboarding_drafts(status="sending")
        assert [str(row["draft_id"]) for row in sending] == [draft_a]
        assert sending[0]["approved_by_name"] == "Juan Canfield"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_mark_lead_lost_records_reason_is_idempotent_and_reopens():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_lead_lost_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        # lost/reopen need only the lifecycle + lead_stage schema, not the
        # privilege-migration role bootstrap (354), so it runs without a
        # disposable-role-admin session.
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn, contact_id=contact_id, lead_stage="estimate_booked"
        )

        lost_key = f"office-lost-{uuid.uuid4().hex}"
        result = await provider.mark_eom_lead_lost(
            contact_id=str(contact_id),
            reason_code="declined_after_estimate",
            note="Too expensive",
            operation_key=lost_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert result["idempotent"] is False
        assert result["lead_stage"] == "lost"
        assert result["from_stage"] == "estimate_booked"

        contact, _, _ = await _contact_state(conn, contact_id)
        assert contact["lead_stage"] == "lost"
        # a lost lead is no longer in the office review-queue predicate
        reviewable = await conn.fetchval(
            """
            SELECT COUNT(*) FROM contacts
            WHERE id = $1 AND lead_stage IN ('new', 'estimate_booked', 'won')
            """,
            contact_id,
        )
        assert int(reviewable) == 0

        row = await conn.fetchrow(
            """
            SELECT from_stage, to_stage, reason, actor, source, operation_key, metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'lead_lost'
            """,
            contact_id,
        )
        assert row["from_stage"] == "estimate_booked"
        assert row["to_stage"] == "lost"
        assert row["reason"] == "Too expensive"
        assert row["actor"] == "employee:1:Juan Canfield"
        assert row["source"] == "eom_office"
        assert row["operation_key"] == lost_key
        metadata = _metadata_dict(row["metadata"])
        assert metadata["lost_reason_code"] == "declined_after_estimate"
        assert metadata["lost_by_employee_id"] == 1

        # replay under the same key: idempotent, no second lifecycle row
        replay = await provider.mark_eom_lead_lost(
            contact_id=str(contact_id),
            reason_code="declined_after_estimate",
            note="Too expensive",
            operation_key=lost_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert replay["idempotent"] is True
        lost_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'lead_lost'
            """,
            contact_id,
        )
        assert int(lost_count) == 1

        # reopen returns the lead to the active queue at the stage the loss
        # displaced. For estimate_booked, that keeps the existing estimate
        # booking evidence consistent with the active pipeline stage.
        reopen_key = f"office-reopen-{uuid.uuid4().hex}"
        reopened = await provider.reopen_eom_lead(
            contact_id=str(contact_id),
            operation_key=reopen_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert reopened["idempotent"] is False
        assert reopened["lead_stage"] == "estimate_booked"
        contact, _, _ = await _contact_state(conn, contact_id)
        assert contact["lead_stage"] == "estimate_booked"
        reopened_row = await conn.fetchrow(
            """
            SELECT from_stage, to_stage
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'lead_reopened'
            """,
            contact_id,
        )
        assert reopened_row["from_stage"] == "lost"
        assert reopened_row["to_stage"] == "estimate_booked"

        # A lead that was lost from new still restores to new.
        new_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=new_id, lead_stage="new")
        await provider.mark_eom_lead_lost(
            contact_id=str(new_id),
            reason_code="spam",
            note=None,
            operation_key=f"office-lost-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        reopened_new = await provider.reopen_eom_lead(
            contact_id=str(new_id),
            operation_key=f"office-reopen-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert reopened_new["lead_stage"] == "new"
        new_contact, _, _ = await _contact_state(conn, new_id)
        assert new_contact["lead_stage"] == "new"

        # The database-owned lifecycle sequence, not transaction-start timestamp
        # ordering, owns the latest loss. Exercise the real mark-lost writer for
        # both loss cycles, but install an insert-time test trigger that gives
        # the stale loss the newer timestamps. A timestamp-ordered reopen would
        # restore this lead to new instead of estimate_booked.
        chronological_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=chronological_id, lead_stage="new")
        stale_loss_key = f"lost-old-{uuid.uuid4().hex}"
        current_loss_key = f"lost-current-{uuid.uuid4().hex}"
        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION set_test_loss_clock()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.operation_key = current_setting(
                    'atlas.test.stale_loss_key', true
                ) THEN
                    NEW.occurred_at = '2026-01-04T00:00:00Z'::timestamptz;
                    NEW.created_at = '2026-01-04T00:00:00Z'::timestamptz;
                ELSIF NEW.operation_key = current_setting(
                    'atlas.test.current_loss_key', true
                ) THEN
                    NEW.occurred_at = '2026-01-01T00:00:00Z'::timestamptz;
                    NEW.created_at = '2026-01-01T00:00:00Z'::timestamptz;
                END IF;
                RETURN NEW;
            END;
            $$;
            """
        )
        await conn.execute(
            """
            CREATE TRIGGER trg_set_test_loss_clock
                BEFORE INSERT ON eom_lead_lifecycle_events
                FOR EACH ROW
                WHEN (NEW.event_type = 'lead_lost')
                EXECUTE FUNCTION set_test_loss_clock()
            """
        )
        await conn.execute(
            "SELECT set_config('atlas.test.stale_loss_key', $1, false)",
            stale_loss_key,
        )
        await conn.execute(
            "SELECT set_config('atlas.test.current_loss_key', $1, false)",
            current_loss_key,
        )
        await provider.mark_eom_lead_lost(
            contact_id=str(chronological_id),
            reason_code="spam",
            note=None,
            operation_key=stale_loss_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await provider.reopen_eom_lead(
            contact_id=str(chronological_id),
            operation_key=f"reopen-old-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key
            ) VALUES ($1, 'estimate_booked', 'new', 'estimate_booked',
                      'employee:1:Juan Canfield', 'eom_office', $2)
            """,
            chronological_id,
            f"estimate-{uuid.uuid4().hex}",
        )
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'estimate_booked' WHERE id = $1",
            chronological_id,
        )
        await provider.mark_eom_lead_lost(
            contact_id=str(chronological_id),
            reason_code="declined_after_estimate",
            note=None,
            operation_key=current_loss_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        sequence_rows = await conn.fetch(
            """
            SELECT from_stage, lifecycle_sequence, occurred_at
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'lead_lost'
            ORDER BY lifecycle_sequence
            """,
            chronological_id,
        )
        assert [row["from_stage"] for row in sequence_rows] == [
            "new",
            "estimate_booked",
        ]
        assert (
            sequence_rows[0]["lifecycle_sequence"]
            < sequence_rows[1]["lifecycle_sequence"]
        )
        assert sequence_rows[0]["occurred_at"] > sequence_rows[1]["occurred_at"]
        reopened_chronological = await provider.reopen_eom_lead(
            contact_id=str(chronological_id),
            operation_key=f"office-reopen-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert reopened_chronological["lead_stage"] == "estimate_booked"

        legacy_ambiguous_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=legacy_ambiguous_id, lead_stage="new")
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'lost' WHERE id = $1",
            legacy_ambiguous_id,
        )
        await conn.executemany(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, occurred_at, created_at, lifecycle_sequence
            )
            VALUES ($1, 'lead_lost', $2, 'lost',
                    'employee:1:Juan Canfield', 'eom_office', $3,
                    $4::timestamptz, $4::timestamptz, NULL)
            """,
            [
                (
                    legacy_ambiguous_id,
                    "new",
                    f"legacy-lost-new-{uuid.uuid4().hex}",
                    datetime(2026, 1, 4, tzinfo=timezone.utc),
                ),
                (
                    legacy_ambiguous_id,
                    "estimate_booked",
                    f"legacy-lost-estimate-{uuid.uuid4().hex}",
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                ),
            ],
        )
        with pytest.raises(
            EOMLeadConversionError, match="requires chronology reconciliation"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(legacy_ambiguous_id),
                operation_key=f"office-reopen-{uuid.uuid4().hex}",
                actor_id=1,
                actor_name="Juan Canfield",
            )
        legacy_ambiguous_contact, _, _ = await _contact_state(
            conn, legacy_ambiguous_id
        )
        assert legacy_ambiguous_contact["lead_stage"] == "lost"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_mark_lead_lost_guards_admission_fence_reuse_and_reopen():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_lead_lost_guard_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)

    async def _lose(contact_id, key, *, reason_code="spam"):
        return await provider.mark_eom_lead_lost(
            contact_id=str(contact_id),
            reason_code=reason_code,
            note=None,
            operation_key=key,
            actor_id=1,
            actor_name="Juan Canfield",
        )

    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)

        # (1) 'won' is out of the mark-lost admission set: it already booked a
        # first clean and enqueued an onboarding welcome draft, so losing it is
        # a separate slice.
        won_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=won_id, lead_stage="won")
        with pytest.raises(
            EOMLeadConversionError, match="stage that can be marked lost"
        ):
            await _lose(won_id, f"k-{uuid.uuid4().hex}", reason_code="other")

        # (2) an unreconciled requested booking (execution lock free, but no
        # booked/terminal marker) still blocks the loss -- the complete handoff
        # fence, not only the execution-lock probe.
        booking_id = uuid.uuid4()
        await _insert_contact(
            conn, contact_id=booking_id, lead_stage="estimate_booked"
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key
            ) VALUES ($1, 'estimate_booking_requested', 'new', 'estimate_booked',
                      'employee:1:Juan Canfield', 'eom_office', $2)
            """,
            booking_id,
            f"book-{uuid.uuid4().hex}",
        )
        with pytest.raises(
            EOMLeadConversionError, match="pending calendar completion"
        ):
            await _lose(booking_id, f"k-{uuid.uuid4().hex}", reason_code="other")

        # (3) an Idempotency-Key is bound to one lead: reusing it on another
        # contact is a client error, not a second loss.
        a_id, b_id = uuid.uuid4(), uuid.uuid4()
        await _insert_contact(conn, contact_id=a_id, lead_stage="new")
        await _insert_contact(conn, contact_id=b_id, lead_stage="new")
        shared_key = f"shared-{uuid.uuid4().hex}"
        await _lose(a_id, shared_key)
        with pytest.raises(EOMLeadConversionError, match="another EOM lead"):
            await _lose(b_id, shared_key)

        # (4) replaying the original lost key AFTER a reopen is a conflict, not a
        # false "still lost" success reporting the stale stage.
        await provider.reopen_eom_lead(
            contact_id=str(a_id),
            operation_key=f"re-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        with pytest.raises(
            EOMLeadConversionError, match="reopened after this operation"
        ):
            await _lose(a_id, shared_key)
        current_loss_key = f"lost-current-{uuid.uuid4().hex}"
        current_loss = await _lose(a_id, current_loss_key, reason_code="no_response")
        assert current_loss["idempotent"] is False
        current_replay = await _lose(
            a_id, current_loss_key, reason_code="no_response"
        )
        assert current_replay["idempotent"] is True
        with pytest.raises(EOMLeadConversionError, match="lost operation was superseded"):
            await _lose(a_id, shared_key)

        # (5) reopen requires an active contact: an archived lost lead cannot be
        # reported back as active while it stays out of the review queue.
        c_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=c_id, lead_stage="new")
        await _lose(c_id, f"lost-{uuid.uuid4().hex}", reason_code="no_response")
        await conn.execute(
            "UPDATE contacts SET status = 'inactive' WHERE id = $1", c_id
        )
        with pytest.raises(
            EOMLeadConversionError, match="must be active to reopen"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(c_id),
                operation_key=f"re-{uuid.uuid4().hex}",
                actor_id=1,
                actor_name="Juan Canfield",
            )

        # (6) already lost under a *different* key is a conflict, not a keyless
        # 200 — so no operation_key is reported successful without a durable
        # replay row behind it.
        d_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=d_id, lead_stage="new")
        await _lose(d_id, f"lost-{uuid.uuid4().hex}")
        with pytest.raises(EOMLeadConversionError, match="already lost"):
            await _lose(d_id, f"lost-{uuid.uuid4().hex}")

        # (7) reopen under a different key when the lead is already active is a
        # conflict, not a no-op.
        await provider.reopen_eom_lead(
            contact_id=str(d_id),
            operation_key=f"re-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        with pytest.raises(
            EOMLeadConversionError, match="not lost and cannot be reopened"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(d_id),
                operation_key=f"re-{uuid.uuid4().hex}",
                actor_id=1,
                actor_name="Juan Canfield",
            )

        # (8) replaying a reopen key after the lead was lost again is a 409, not
        # a stale "new/active" success.
        e_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=e_id, lead_stage="new")
        await _lose(e_id, f"lost-{uuid.uuid4().hex}")
        reopen_e = f"re-{uuid.uuid4().hex}"
        await provider.reopen_eom_lead(
            contact_id=str(e_id),
            operation_key=reopen_e,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        await _lose(e_id, f"lost-{uuid.uuid4().hex}")
        with pytest.raises(
            EOMLeadConversionError, match="changed after this reopen"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(e_id),
                operation_key=reopen_e,
                actor_id=1,
                actor_name="Juan Canfield",
            )
        later_reopen_e = f"re-{uuid.uuid4().hex}"
        later_reopen = await provider.reopen_eom_lead(
            contact_id=str(e_id),
            operation_key=later_reopen_e,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert later_reopen["idempotent"] is False
        later_replay = await provider.reopen_eom_lead(
            contact_id=str(e_id),
            operation_key=later_reopen_e,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert later_replay["idempotent"] is True
        with pytest.raises(
            EOMLeadConversionError, match="reopen operation was superseded"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(e_id),
                operation_key=reopen_e,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        # (9) legacy, pre-sequence replay rows are accepted only while they are
        # the sole lost/reopen disposition for the contact. If another
        # disposition row exists, replay ownership is ambiguous and fails closed.
        legacy_loss_id = uuid.uuid4()
        legacy_loss_key = f"legacy-lost-{uuid.uuid4().hex}"
        await _insert_contact(conn, contact_id=legacy_loss_id, lead_stage="new")
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'lost' WHERE id = $1",
            legacy_loss_id,
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, metadata, lifecycle_sequence
            )
            VALUES (
                $1, 'lead_lost', 'new', 'lost', 'employee:1:Juan Canfield',
                'eom_office', $2, jsonb_build_object('lost_reason_code', 'spam'), NULL
            )
            """,
            legacy_loss_id,
            legacy_loss_key,
        )
        legacy_loss_replay = await provider.mark_eom_lead_lost(
            contact_id=str(legacy_loss_id),
            reason_code="spam",
            note=None,
            operation_key=legacy_loss_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert legacy_loss_replay["idempotent"] is True

        malformed_loss_id = uuid.uuid4()
        malformed_loss_key = f"legacy-malformed-lost-{uuid.uuid4().hex}"
        await _insert_contact(conn, contact_id=malformed_loss_id, lead_stage="new")
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'lost' WHERE id = $1",
            malformed_loss_id,
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, metadata, lifecycle_sequence
            )
            VALUES (
                $1, 'lead_lost', 'won', 'lost', 'employee:1:Juan Canfield',
                'eom_office', $2, jsonb_build_object('lost_reason_code', 'spam'), NULL
            )
            """,
            malformed_loss_id,
            malformed_loss_key,
        )
        with pytest.raises(EOMLeadConversionError, match="lost operation was superseded"):
            await provider.mark_eom_lead_lost(
                contact_id=str(malformed_loss_id),
                reason_code="spam",
                note=None,
                operation_key=malformed_loss_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            )
            VALUES ($1, 'lead_lost', 'estimate_booked', 'lost',
                    'employee:1:Juan Canfield', 'eom_office', $2, NULL)
            """,
            legacy_loss_id,
            f"legacy-lost-later-{uuid.uuid4().hex}",
        )
        with pytest.raises(EOMLeadConversionError, match="lost operation was superseded"):
            await provider.mark_eom_lead_lost(
                contact_id=str(legacy_loss_id),
                reason_code="spam",
                note=None,
                operation_key=legacy_loss_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        legacy_reopen_id = uuid.uuid4()
        legacy_reopen_loss_key = f"legacy-reopen-loss-{uuid.uuid4().hex}"
        legacy_reopen_key = f"legacy-reopen-{uuid.uuid4().hex}"
        await _insert_contact(conn, contact_id=legacy_reopen_id, lead_stage="new")
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            )
            VALUES ($1, 'lead_lost', 'new', 'lost',
                    'employee:1:Juan Canfield', 'eom_office', $2, NULL)
            """,
            legacy_reopen_id,
            legacy_reopen_loss_key,
        )
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'lost' WHERE id = $1",
            legacy_reopen_id,
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            )
            VALUES ($1, 'lead_reopened', 'lost', 'new',
                    'employee:1:Juan Canfield', 'eom_office', $2, NULL)
            """,
            legacy_reopen_id,
            legacy_reopen_key,
        )
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'new' WHERE id = $1",
            legacy_reopen_id,
        )
        legacy_reopen_replay = await provider.reopen_eom_lead(
            contact_id=str(legacy_reopen_id),
            operation_key=legacy_reopen_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert legacy_reopen_replay["idempotent"] is True
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            )
            VALUES ($1, 'lead_lost', 'new', 'lost',
                    'employee:1:Juan Canfield', 'eom_office', $2, NULL)
            """,
            legacy_reopen_id,
            f"legacy-reopen-later-loss-{uuid.uuid4().hex}",
        )
        with pytest.raises(
            EOMLeadConversionError, match="reopen operation was superseded"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(legacy_reopen_id),
                operation_key=legacy_reopen_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        # (10) a lost contact without a lead_lost row cannot be reopened: the
        # transaction returns 409, leaves the contact lost, and appends no
        # lead_reopened lifecycle event.
        missing_loss_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=missing_loss_id, lead_stage="new")
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'lost' WHERE id = $1",
            missing_loss_id,
        )
        with pytest.raises(
            EOMLeadConversionError, match="no lost-stage evidence"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(missing_loss_id),
                operation_key=f"re-{uuid.uuid4().hex}",
                actor_id=1,
                actor_name="Juan Canfield",
            )
        missing_loss_contact, _, _ = await _contact_state(conn, missing_loss_id)
        assert missing_loss_contact["lead_stage"] == "lost"
        missing_reopen_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'lead_reopened'
            """,
            missing_loss_id,
        )
        assert int(missing_reopen_count) == 0

        # (11) unsafe lead_lost.from_stage evidence is also rejected without
        # changing the contact or appending a lead_reopened event.
        unsafe_loss_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=unsafe_loss_id, lead_stage="new")
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'lost' WHERE id = $1",
            unsafe_loss_id,
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key
            )
            VALUES ($1, 'lead_lost', 'won', 'lost',
                    'employee:1:Juan Canfield', 'eom_office', $2)
            """,
            unsafe_loss_id,
            f"lost-unsafe-{uuid.uuid4().hex}",
        )
        with pytest.raises(
            EOMLeadConversionError, match="cannot be safely restored"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(unsafe_loss_id),
                operation_key=f"re-{uuid.uuid4().hex}",
                actor_id=1,
                actor_name="Juan Canfield",
            )
        unsafe_loss_contact, _, _ = await _contact_state(conn, unsafe_loss_id)
        assert unsafe_loss_contact["lead_stage"] == "lost"
        unsafe_reopen_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'lead_reopened'
            """,
            unsafe_loss_id,
        )
        assert int(unsafe_reopen_count) == 0
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_legacy_reopen_replay_accepts_pre_sequence_loss_reopen_pair():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_legacy_reopen_pair_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(
            conn,
            schema,
            apply_privilege_migration=False,
            apply_lifecycle_sequence_migration=False,
        )
        # Current provider code reads the column on replay, but this fixture must
        # still prove the legacy/unsequenced producer shape. Add only the
        # nullable column here, without migration 363's sequence/default, so the
        # real provider can create pre-sequence lost/reopen rows with NULL
        # lifecycle_sequence before the database-owned ordering migration lands.
        await conn.execute(
            "ALTER TABLE eom_lead_lifecycle_events ADD COLUMN lifecycle_sequence BIGINT"
        )
        legacy_provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        lost_key = f"legacy-lost-{uuid.uuid4().hex}"
        reopen_key = f"legacy-reopen-{uuid.uuid4().hex}"
        await _insert_contact(conn, contact_id=contact_id, lead_stage="new")
        produced_loss = await legacy_provider.mark_eom_lead_lost(
            contact_id=str(contact_id),
            reason_code="spam",
            note=None,
            operation_key=lost_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert produced_loss["idempotent"] is False
        produced_reopen = await legacy_provider.reopen_eom_lead(
            contact_id=str(contact_id),
            operation_key=reopen_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert produced_reopen["idempotent"] is False
        assert produced_reopen["lead_stage"] == "new"
        await conn.execute(
            (MIGRATIONS / "363_eom_lead_lifecycle_sequence.sql").read_text()
        )
        provider = DatabaseCRMProvider(pool=conn)

        sequence_rows = await conn.fetch(
            """
            SELECT event_type, lifecycle_sequence
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type IN ('lead_lost', 'lead_reopened')
            ORDER BY event_type
            """,
            contact_id,
        )
        assert [row["event_type"] for row in sequence_rows] == [
            "lead_lost",
            "lead_reopened",
        ]
        assert all(row["lifecycle_sequence"] is None for row in sequence_rows)

        replay = await provider.reopen_eom_lead(
            contact_id=str(contact_id),
            operation_key=reopen_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert replay["idempotent"] is True
        assert replay["lead_stage"] == "new"

        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            ) VALUES (
                $1, 'lead_lost', 'new', 'lost',
                'employee:1:Juan Canfield', 'eom_office', $2, NULL
            )
            """,
            contact_id,
            f"legacy-extra-lost-{uuid.uuid4().hex}",
        )
        with pytest.raises(
            EOMLeadConversionError, match="reopen operation was superseded"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(contact_id),
                operation_key=reopen_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        mismatched_contact_id = uuid.uuid4()
        mismatched_loss_key = f"legacy-mismatched-lost-{uuid.uuid4().hex}"
        mismatched_reopen_key = f"legacy-mismatched-reopen-{uuid.uuid4().hex}"
        await _insert_contact(
            conn, contact_id=mismatched_contact_id, lead_stage="new"
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            ) VALUES (
                $1, 'lead_lost', 'estimate_booked', 'lost',
                'employee:1:Juan Canfield', 'eom_office', $2, NULL
            )
            """,
            mismatched_contact_id,
            mismatched_loss_key,
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            ) VALUES (
                $1, 'lead_reopened', 'lost', 'new',
                'employee:1:Juan Canfield', 'eom_office', $2, NULL
            )
            """,
            mismatched_contact_id,
            mismatched_reopen_key,
        )
        with pytest.raises(
            EOMLeadConversionError, match="reopen operation was superseded"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(mismatched_contact_id),
                operation_key=mismatched_reopen_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        malformed_reopen_contact_id = uuid.uuid4()
        malformed_reopen_loss_key = f"legacy-malformed-reopen-lost-{uuid.uuid4().hex}"
        malformed_reopen_key = f"legacy-malformed-reopen-{uuid.uuid4().hex}"
        await _insert_contact(
            conn, contact_id=malformed_reopen_contact_id, lead_stage="new"
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            ) VALUES (
                $1, 'lead_lost', 'new', 'lost',
                'employee:1:Juan Canfield', 'eom_office', $2, NULL
            )
            """,
            malformed_reopen_contact_id,
            malformed_reopen_loss_key,
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, from_stage, to_stage, actor, source,
                operation_key, lifecycle_sequence
            ) VALUES (
                $1, 'lead_reopened', 'estimate_booked', 'new',
                'employee:1:Juan Canfield', 'eom_office', $2, NULL
            )
            """,
            malformed_reopen_contact_id,
            malformed_reopen_key,
        )
        with pytest.raises(
            EOMLeadConversionError, match="reopen operation was superseded"
        ):
            await provider.reopen_eom_lead(
                contact_id=str(malformed_reopen_contact_id),
                operation_key=malformed_reopen_key,
                actor_id=1,
                actor_name="Juan Canfield",
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_legacy_disposition_replay_rejects_held_out_transition_shapes():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_legacy_disposition_shapes_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(
            conn,
            schema,
            apply_privilege_migration=False,
            apply_lifecycle_sequence_migration=False,
        )
        await conn.execute(
            (MIGRATIONS / "363_eom_lead_lifecycle_sequence.sql").read_text()
        )
        provider = DatabaseCRMProvider(pool=conn)

        async def insert_legacy_lost(
            *,
            contact_id: uuid.UUID,
            from_stage: str,
            to_stage: str,
            operation_key: str,
        ) -> None:
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor, source,
                    operation_key, metadata, lifecycle_sequence
                ) VALUES (
                    $1, 'lead_lost', $2, $3, 'employee:1:Juan Canfield',
                    'eom_office', $4, jsonb_build_object('lost_reason_code', 'spam'), NULL
                )
                """,
                contact_id,
                from_stage,
                to_stage,
                operation_key,
            )

        async def insert_legacy_reopened(
            *,
            contact_id: uuid.UUID,
            from_stage: str,
            to_stage: str,
            operation_key: str,
        ) -> None:
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor, source,
                    operation_key, lifecycle_sequence
                ) VALUES (
                    $1, 'lead_reopened', $2, $3,
                    'employee:1:Juan Canfield', 'eom_office', $4, NULL
                )
                """,
                contact_id,
                from_stage,
                to_stage,
                operation_key,
            )

        async def assert_lost_replay_rejected(from_stage: str, to_stage: str) -> None:
            contact_id = uuid.uuid4()
            operation_key = f"legacy-lost-shape-{uuid.uuid4().hex}"
            await _insert_contact(conn, contact_id=contact_id, lead_stage="new")
            await conn.execute(
                "UPDATE contacts SET lead_stage = 'lost' WHERE id = $1",
                contact_id,
            )
            await insert_legacy_lost(
                contact_id=contact_id,
                from_stage=from_stage,
                to_stage=to_stage,
                operation_key=operation_key,
            )

            with pytest.raises(
                EOMLeadConversionError, match="lost operation was superseded"
            ):
                await provider.mark_eom_lead_lost(
                    contact_id=str(contact_id),
                    reason_code="spam",
                    note=None,
                    operation_key=operation_key,
                    actor_id=1,
                    actor_name="Juan Canfield",
                )

        async def assert_reopen_replay_rejected(
            *,
            loss_from_stage: str,
            loss_to_stage: str,
            reopen_from_stage: str,
            reopen_to_stage: str,
            current_stage: str,
        ) -> None:
            contact_id = uuid.uuid4()
            lost_key = f"legacy-lost-shape-{uuid.uuid4().hex}"
            reopen_key = f"legacy-reopen-shape-{uuid.uuid4().hex}"
            await _insert_contact(conn, contact_id=contact_id, lead_stage="new")
            await insert_legacy_lost(
                contact_id=contact_id,
                from_stage=loss_from_stage,
                to_stage=loss_to_stage,
                operation_key=lost_key,
            )
            await insert_legacy_reopened(
                contact_id=contact_id,
                from_stage=reopen_from_stage,
                to_stage=reopen_to_stage,
                operation_key=reopen_key,
            )
            await conn.execute(
                "UPDATE contacts SET lead_stage = $2 WHERE id = $1",
                contact_id,
                current_stage,
            )

            with pytest.raises(
                EOMLeadConversionError, match="reopen operation was superseded"
            ):
                await provider.reopen_eom_lead(
                    contact_id=str(contact_id),
                    operation_key=reopen_key,
                    actor_id=1,
                    actor_name="Juan Canfield",
                )

        # Held-out valid legacy loss shape: the earlier tests use `new -> lost`.
        valid_loss_id = uuid.uuid4()
        valid_loss_key = f"legacy-valid-lost-shape-{uuid.uuid4().hex}"
        await _insert_contact(conn, contact_id=valid_loss_id, lead_stage="new")
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'lost' WHERE id = $1",
            valid_loss_id,
        )
        await insert_legacy_lost(
            contact_id=valid_loss_id,
            from_stage="estimate_booked",
            to_stage="lost",
            operation_key=valid_loss_key,
        )
        valid_loss = await provider.mark_eom_lead_lost(
            contact_id=str(valid_loss_id),
            reason_code="spam",
            note=None,
            operation_key=valid_loss_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert valid_loss["idempotent"] is True

        # Held-out valid legacy reopen shape: the earlier tests use `lost -> new`.
        valid_reopen_id = uuid.uuid4()
        valid_reopen_lost_key = f"legacy-valid-reopen-lost-{uuid.uuid4().hex}"
        valid_reopen_key = f"legacy-valid-reopen-shape-{uuid.uuid4().hex}"
        await _insert_contact(conn, contact_id=valid_reopen_id, lead_stage="new")
        await insert_legacy_lost(
            contact_id=valid_reopen_id,
            from_stage="estimate_booked",
            to_stage="lost",
            operation_key=valid_reopen_lost_key,
        )
        await insert_legacy_reopened(
            contact_id=valid_reopen_id,
            from_stage="lost",
            to_stage="estimate_booked",
            operation_key=valid_reopen_key,
        )
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'estimate_booked' WHERE id = $1",
            valid_reopen_id,
        )
        valid_reopen = await provider.reopen_eom_lead(
            contact_id=str(valid_reopen_id),
            operation_key=valid_reopen_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert valid_reopen["idempotent"] is True
        assert valid_reopen["lead_stage"] == "estimate_booked"

        for from_stage, to_stage in (
            ("won", "lost"),
            ("lost", "lost"),
            ("new", "estimate_booked"),
            ("estimate_booked", "new"),
        ):
            await assert_lost_replay_rejected(from_stage, to_stage)

        for replay_shape in (
            {
                "loss_from_stage": "new",
                "loss_to_stage": "lost",
                "reopen_from_stage": "estimate_booked",
                "reopen_to_stage": "new",
                "current_stage": "new",
            },
            {
                "loss_from_stage": "estimate_booked",
                "loss_to_stage": "lost",
                "reopen_from_stage": "lost",
                "reopen_to_stage": "new",
                "current_stage": "new",
            },
            {
                "loss_from_stage": "new",
                "loss_to_stage": "estimate_booked",
                "reopen_from_stage": "lost",
                "reopen_to_stage": "new",
                "current_stage": "new",
            },
            {
                "loss_from_stage": "new",
                "loss_to_stage": "lost",
                "reopen_from_stage": "lost",
                "reopen_to_stage": "won",
                "current_stage": "won",
            },
            {
                "loss_from_stage": "estimate_booked",
                "loss_to_stage": "lost",
                "reopen_from_stage": "new",
                "reopen_to_stage": "estimate_booked",
                "current_stage": "estimate_booked",
            },
        ):
            await assert_reopen_replay_rejected(**replay_shape)
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
