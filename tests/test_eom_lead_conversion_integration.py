"""Real-Postgres proof for the EOM customer handoff transaction."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager, suppress
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
    EOMFirstCleanBooking,
    deterministic_eom_estimate_calendar_event_id,
    deterministic_eom_first_clean_calendar_event_id,
    schedule_eom_first_clean_booking,
)
from atlas_brain.services.eom_lead_conversion import (  # noqa: E402
    EOMLeadConversionError,
    EOMLeadLost,
)
from atlas_brain.services.eom_won_lead_loss import (  # noqa: E402
    mark_eom_lead_lost_with_won_teardown,
)
from atlas_brain.services.eom_lead_ingress import (  # noqa: E402
    resolve_or_create_eom_inbound_lead,
)
from atlas_brain.services.eom_public_onboarding_tokens import (  # noqa: E402
    eom_public_onboarding_hmac_key_fingerprint,
    parse_eom_public_onboarding_token,
)
from atlas_brain.tools.base import ToolResult  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
_NOCODB_TEST_PASSWORD = "test-only-nocodb-password"
_NON_SUPERUSER_TEST_PASSWORD = "test-only-migrator-password"
_PUBLIC_ONBOARDING_URL = "https://effinghamofficemaids.com/onboarding"
_PUBLIC_ONBOARDING_SECRET = "test-only-public-onboarding-secret-value-123456"
_PUBLIC_ONBOARDING_KEY_FINGERPRINT = eom_public_onboarding_hmac_key_fingerprint(
    secret=_PUBLIC_ONBOARDING_SECRET
)


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
    apply_customer_type_revision_migration: bool = True,
    apply_public_onboarding_migration: bool = False,
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
        # The provider's contact INSERT names customer_type explicitly, so the
        # column has to exist for any operator write in these tests.
        "366_contacts_customer_type.sql",
    ]
    if apply_customer_type_revision_migration:
        migration_names.append("367_contacts_customer_type_revision.sql")
    if apply_lifecycle_sequence_migration:
        migration_names.append("363_eom_lead_lifecycle_sequence.sql")
    if apply_privilege_migration:
        await _provision_nocodb_login(conn)
        migration_names.append("354_eom_customer_handoff_privileges.sql")
    if apply_public_onboarding_migration:
        migration_names.append("383_eom_public_onboarding_tokens.sql")
    migration_names.append("386_eom_won_loss_nocodb_fence.sql")
    for name in migration_names:
        if name == "367_contacts_customer_type_revision.sql":
            async with conn.transaction():
                await conn.execute((MIGRATIONS / name).read_text())
        else:
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
        # A create overwrote nothing, so it must carry neither key. Asserted
        # directly rather than left to the other assertions staying green: they
        # would all still pass if a regression started attaching overwritten
        # values to creates.
        assert "previous_values" not in metadata
        assert "changed_fields" not in metadata
        # The #254 tri-state key is update-only by the same rule: a create
        # overwrote (and cleared) nothing, and consumers distinguish
        # pre-slice events from new-format ones by key presence, so a create
        # emitting the key would corrupt that signal.
        assert "cleared_fields" not in metadata

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

        crossed_actor = EOMOperatorContactMutation.from_raw(
            operation_key=operation_key,
            actor_id=9,
            actor_name="Other Operator",
            source_channel="time_tracker",
            source_ref="customer:88",
            contact_type="customer",
            fields={
                "full_name": "Operator Created",
                "email": "operator@example.com",
                "phone": "2175550100",
                "address": "",
            },
        )
        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, crossed_actor)
        assert exc_info.value.status_code == 409

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
        # What was overwritten, not merely which fields moved. There is no
        # contact history table, so the event is the only place the prior value
        # survives the UPDATE.
        assert metadata["previous_values"] == {
            "email": None,
            "phone": "1 (217) 555-0100",
        }
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
async def test_operator_contact_mutation_rejects_non_object_contact_metadata():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_non_object_metadata_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=contact_id,
            full_name="Legacy Metadata Shape",
            phone="2175550100",
            contact_type="customer",
            lead_stage=None,
            source="legacy",
        )
        await conn.execute(
            "UPDATE contacts SET metadata = '[\"legacy\"]'::jsonb WHERE id = $1",
            contact_id,
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-non-object-metadata-{uuid.uuid4().hex}",
            actor_id=8,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:non-object-metadata",
            fields={"phone": "217-555-0100", "notes": "must not overwrite metadata"},
        )

        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, command)

        assert exc_info.value.status_code == 409
        assert await conn.fetchval(
            "SELECT metadata::text FROM contacts WHERE id = $1", contact_id
        ) == '["legacy"]'
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
@pytest.mark.parametrize(
    "source_metadata",
    (
        None,
        ["time_tracker:customer:malformed-provenance"],
        {"time_tracker:customer:malformed-provenance": "not-a-record"},
    ),
)
async def test_operator_contact_mutation_rejects_malformed_source_provenance(
    source_metadata: object,
):
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_bad_source_metadata_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        metadata = {"eom_operator_contact_sources": source_metadata}
        await _insert_contact(
            conn,
            contact_id=contact_id,
            full_name="Malformed Source Metadata",
            phone="2175550100",
            contact_type="customer",
            lead_stage=None,
            source="legacy",
        )
        await conn.execute(
            "UPDATE contacts SET metadata = $2::jsonb WHERE id = $1",
            contact_id,
            json.dumps(metadata, sort_keys=True),
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-bad-source-metadata-{uuid.uuid4().hex}",
            actor_id=8,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:malformed-provenance",
            fields={"full_name": "Must Not Mutate"},
        )

        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, command)

        assert exc_info.value.status_code == 409
        row = await conn.fetchrow(
            "SELECT full_name, metadata::text FROM contacts WHERE id = $1", contact_id
        )
        assert row["full_name"] == "Malformed Source Metadata"
        assert json.loads(row["metadata"]) == metadata
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
async def test_operator_contact_mutation_crossed_identities_fail_without_deadlock():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_stable_row_locks_{uuid.uuid4().hex}"
    setup = await asyncpg.connect(database_url)
    pool = None
    try:
        await _prepare_schema(setup, schema, apply_privilege_migration=False)
        first_id = uuid.uuid4()
        second_id = uuid.uuid4()
        await _insert_contact(
            setup,
            contact_id=first_id,
            full_name="First Identity",
            phone="2175550101",
            email="first@example.com",
            contact_type="customer",
            lead_stage=None,
        )
        await _insert_contact(
            setup,
            contact_id=second_id,
            full_name="Second Identity",
            phone="2175550102",
            email="second@example.com",
            contact_type="customer",
            lead_stage=None,
        )
        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=2,
            server_settings={"search_path": f'"{schema}", public'},
        )
        provider = DatabaseCRMProvider(pool=pool)

        first_command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-crossed-a-{uuid.uuid4().hex}",
            actor_id=8,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:crossed-a",
            fields={"phone": "217-555-0101", "email": "second@example.com"},
        )
        second_command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-crossed-b-{uuid.uuid4().hex}",
            actor_id=8,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:crossed-b",
            fields={"phone": "217-555-0102", "email": "first@example.com"},
        )

        results = await asyncio.wait_for(
            asyncio.gather(
                mutate_eom_operator_contact(provider, first_command),
                mutate_eom_operator_contact(provider, second_command),
                return_exceptions=True,
            ),
            timeout=5,
        )

        assert all(
            isinstance(result, EOMOperatorContactMutationError)
            and result.status_code == 409
            for result in results
        )
        assert await setup.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE event_type IN ('contact_created', 'contact_updated')
            """
        ) == 0
    finally:
        if pool is not None:
            await pool.close()
        await setup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await setup.close()


@pytest.mark.asyncio
async def test_operator_contact_mutation_claims_legacy_contact_by_padded_email():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_email_claim_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=contact_id,
            business_context_id=None,
            full_name="Padded Email Legacy",
            contact_type="customer",
            lead_stage=None,
            email="\tAda.Example@Example.COM\n",
            phone=None,
            source="legacy",
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-padded-email-{uuid.uuid4().hex}",
            actor_id=8,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:padded-email",
            fields={"email": "ada.example@example.com", "notes": "claimed"},
        )

        updated = await mutate_eom_operator_contact(provider, command)

        assert updated["operation"] == "contact_updated"
        assert updated["contact_id"] == str(contact_id)
        row = await conn.fetchrow("SELECT * FROM contacts WHERE id = $1", contact_id)
        assert row["business_context_id"] == "effingham_maids"
        assert row["email"] == "ada.example@example.com"
        assert row["notes"] == "claimed"
        assert await conn.fetchval("SELECT COUNT(*) FROM contacts") == 1
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'contact_updated'
            """,
            contact_id,
        ) == 1
    finally:
        await conn.close()


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
            email=(
                f"\t{claimed_value.upper()}\n"
                if field == "email"
                else "owned@example.com"
            ),
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
        target = await conn.fetchrow(
            "SELECT phone, email FROM contacts WHERE id = $1", target_id
        )
        owner = await conn.fetchrow(
            "SELECT phone, email FROM contacts WHERE id = $1", owner_id
        )
        assert target[field] == original_value
        assert owner[field].strip().lower() == claimed_value
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
async def test_operator_contact_mutation_rejects_ambiguous_direct_provenance():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_ambiguous_source_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        first_id = uuid.uuid4()
        second_id = uuid.uuid4()
        for contact_id, full_name in (
            (first_id, "First Manual Customer"),
            (second_id, "Second Manual Customer"),
        ):
            await _insert_contact(
                conn,
                contact_id=contact_id,
                full_name=full_name,
                contact_type="customer",
                lead_stage=None,
                source="manual",
            )
            await conn.execute(
                """
                UPDATE contacts
                SET source_ref = 'time_tracker:customer:duplicate'
                WHERE id = $1
                """,
                contact_id,
            )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-ambiguous-source-{uuid.uuid4().hex}",
            actor_id=9,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:duplicate",
            fields={"full_name": "Wrong Target"},
        )

        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, command)

        assert exc_info.value.status_code == 409
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM contacts WHERE full_name = 'Wrong Target'"
        ) == 0
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
@pytest.mark.parametrize(
    "stored_phone",
    ("2175550100 ext 123", "2175550100x123", "fax2175550100"),
)
async def test_operator_contact_mutation_matches_stored_phone_with_extension(
    stored_phone: str,
):
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_extension_match_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=contact_id,
            business_context_id=None,
            full_name="Legacy Extension Match",
            phone=stored_phone,
            contact_type="customer",
            lead_stage=None,
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-extension-match-{uuid.uuid4().hex}",
            actor_id=9,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:extension-match",
            fields={"phone": "217-555-0100", "email": "extension@example.com"},
        )

        result = await mutate_eom_operator_contact(provider, command)

        assert result["contact_id"] == str(contact_id)
        rows = await conn.fetch("SELECT id, phone, email FROM contacts ORDER BY id")
        assert len(rows) == 1
        assert rows[0]["id"] == contact_id
        assert rows[0]["phone"] == "2175550100"
        assert rows[0]["email"] == "extension@example.com"
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_phone",
    ("2175550100 ext 123", "2175550100x123"),
)
async def test_operator_contact_mutation_does_not_match_extension_suffix(
    stored_phone: str,
):
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_extension_suffix_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        existing_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=existing_id,
            business_context_id=None,
            full_name="Legacy Extension Suffix",
            phone=stored_phone,
            email="legacy-extension@example.com",
            contact_type="customer",
            lead_stage=None,
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-extension-suffix-{uuid.uuid4().hex}",
            actor_id=9,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="customer:extension-suffix",
            fields={
                "full_name": "Extension Suffix New Contact",
                "phone": "555-010-0123",
                "email": "suffix@example.com",
            },
        )

        result = await mutate_eom_operator_contact(provider, command)

        assert result["contact_id"] != str(existing_id)
        rows = await conn.fetch(
            "SELECT id, phone, email FROM contacts ORDER BY email"
        )
        assert len(rows) == 2
        legacy = next(row for row in rows if row["id"] == existing_id)
        assert legacy["phone"] == stored_phone
        assert legacy["email"] == "legacy-extension@example.com"
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


@pytest.mark.parametrize(
    ("lead_stage", "full_name", "source_ref", "email"),
    [
        (None, "Legacy Unstaged Lead", "lead:unstaged", "unstaged@example.com"),
        ("lost", "Lost Lead", "lead:lost", "lost@example.com"),
    ],
)
@pytest.mark.asyncio
async def test_operator_contact_mutation_rejects_unsupported_lead_stages(
    lead_stage: str | None,
    full_name: str,
    source_ref: str,
    email: str,
):
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_bad_stage_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=contact_id,
            business_context_id=None,
            full_name=full_name,
            contact_type="lead",
            lead_stage=lead_stage,
            phone="2175550100",
            email=email,
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-bad-stage-{uuid.uuid4().hex}",
            actor_id=10,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref=source_ref,
            contact_id=str(contact_id),
            fields={"phone": "217-555-0100", "notes": "needs funnel stage"},
        )

        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, command)

        assert exc_info.value.status_code == 409
        assert str(exc_info.value) == (
            "EOM operator lead updates require a supported lead stage"
        )
        row = await conn.fetchrow("SELECT * FROM contacts WHERE id = $1", contact_id)
        assert row["business_context_id"] is None
        assert row["contact_type"] == "lead"
        assert row["lead_stage"] == lead_stage
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
async def test_operator_contact_mutation_rejects_inactive_legacy_lead():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_inactive_lead_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=contact_id,
            business_context_id=None,
            full_name="Inactive Legacy Lead",
            contact_type="lead",
            lead_stage="new",
            status="inactive",
            phone="2175550100",
            email="inactive-lead@example.com",
        )
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-inactive-lead-{uuid.uuid4().hex}",
            actor_id=10,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref="lead:inactive",
            contact_id=str(contact_id),
            fields={"phone": "217-555-0100", "notes": "needs active status"},
        )

        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, command)

        assert exc_info.value.status_code == 409
        row = await conn.fetchrow("SELECT * FROM contacts WHERE id = $1", contact_id)
        assert row["business_context_id"] is None
        assert row["contact_type"] == "lead"
        assert row["lead_stage"] == "new"
        assert row["status"] == "inactive"
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
            # The security-definer trigger may consume the sequence for an
            # ordinary NocoDB insert, but that does not grant the NocoDB login
            # direct access to either the sequence or the protected evidence
            # column. Pin both catalog capabilities and their executable
            # denial paths so a future ACL widening cannot pass on the happy
            # path alone.
            assert not await nocodb_conn.fetchval(
                "SELECT has_sequence_privilege("
                "'contacts_customer_type_revision_seq'::regclass, 'USAGE')"
            )
            assert not await nocodb_conn.fetchval(
                "SELECT has_column_privilege("
                "'contacts'::regclass, 'customer_type_revision', 'INSERT')"
            )
            assert not await nocodb_conn.fetchval(
                "SELECT has_column_privilege("
                "'contacts'::regclass, 'customer_type_revision', 'UPDATE')"
            )
            with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
                await nocodb_conn.fetchval(
                    "SELECT nextval('contacts_customer_type_revision_seq'::regclass)"
                )
            with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
                await nocodb_conn.execute(
                    "INSERT INTO contacts (id, full_name, customer_type_revision) "
                    "VALUES ($1, 'Forged revision', 1)",
                    contact_id,
                )
            await nocodb_conn.execute(
                "INSERT INTO contacts (id, full_name, notes) VALUES ($1, 'NocoDB CRM', 'before')",
                contact_id,
            )
            revision_after_insert = await nocodb_conn.fetchval(
                "SELECT customer_type_revision FROM contacts WHERE id = $1",
                contact_id,
            )
            assert revision_after_insert > 0
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
            assert (
                await nocodb_conn.fetchval(
                    "SELECT customer_type_revision FROM contacts WHERE id = $1",
                    contact_id,
                )
                == revision_after_insert
            )
            with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
                await nocodb_conn.execute(
                    "UPDATE contacts SET customer_type_revision = 1 WHERE id = $1",
                    contact_id,
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
        if missing_column == "contact_type":
            # Migration 386 correctly declares this trigger dependency. This
            # fixture intentionally tears down an already-migrated schema to
            # exercise readiness validation for a historical partial schema.
            await conn.execute(
                "DROP TRIGGER IF EXISTS trg_reject_nocodb_eom_won_loss_mutation ON contacts"
            )
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
    interaction metadata), so the draft recipient and issued token prefill
    must resolve through the same latest-intake projection the office review
    queue shows -- not the stale contact column."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_recipient_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
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

        _, token_id = await _claim_public_onboarding_token(
            provider, draft_id=str(completed["onboarding_draft_id"])
        )
        assert await conn.fetchval(
            "SELECT prefill_email FROM eom_public_onboarding_tokens WHERE id = $1",
            token_id,
        ) == "latest-address@example.com"
        assert (
            await provider.get_eom_public_onboarding_session(
                token_id=str(token_id),
                signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            )
        )["email"] == "latest-address@example.com"
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


class _WonLossCalendar:
    """A narrow Calendar boundary fake: Postgres remains the system under test."""

    def __init__(self, *, results: list[ToolResult] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict[str, str]] = []

    async def delete_event(self, *, calendar_id: str, event_id: str) -> ToolResult:
        self.calls.append({"calendar_id": calendar_id, "event_id": event_id})
        if self.results:
            return self.results.pop(0)
        return ToolResult(
            success=True,
            data={
                "calendar_id": calendar_id,
                "event_id": event_id,
                "already_absent": False,
            },
            message="Calendar event deleted",
        )


class _FirstCleanIdentityCalendar:
    """Record identity resolution and creation while Postgres owns lifecycle proof."""

    configured_calendar_id = "primary"

    def __init__(self, *, resolved_calendar_id: str) -> None:
        self.resolved_calendar_id = resolved_calendar_id
        self.resolve_calls: list[dict[str, str]] = []
        self.create_calls: list[dict[str, object]] = []

    async def resolve_calendar_id(self, *, calendar_id: str) -> ToolResult:
        self.resolve_calls.append({"calendar_id": calendar_id})
        return ToolResult(
            success=True,
            data={"calendar_id": self.resolved_calendar_id},
            message="Calendar identity resolved",
        )

    async def create_event(self, **kwargs: object) -> ToolResult:
        self.create_calls.append(kwargs)
        return ToolResult(
            success=True,
            data={"event_id": kwargs["event_id"]},
            message="Calendar event created",
        )


class _BlockingWonLossCalendar(_WonLossCalendar):
    def __init__(self, *, results: list[ToolResult] | None = None) -> None:
        super().__init__(results=results)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def delete_event(self, *, calendar_id: str, event_id: str) -> ToolResult:
        self.calls.append({"calendar_id": calendar_id, "event_id": event_id})
        self.started.set()
        await self.release.wait()
        if self.results:
            return self.results.pop(0)
        return ToolResult(
            success=True,
            data={
                "calendar_id": calendar_id,
                "event_id": event_id,
                "already_absent": False,
            },
            message="Calendar event deleted",
        )


async def _book_first_clean_draft(
    conn,
    provider,
    *,
    email: str | None = "won-lead@example.com",
    full_name: str = "Won Lead",
    contact_id: uuid.UUID | None = None,
    calendar_id: str = "estimate-calendar",
) -> tuple[uuid.UUID, str]:
    """Insert a lead and complete a first-clean booking; return its draft."""
    if contact_id is None:
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn, contact_id=contact_id, email=email, full_name=full_name
        )
    booking_key = f"office-first-clean-{uuid.uuid4().hex}"
    event_id = deterministic_eom_first_clean_calendar_event_id(
        contact_id=str(contact_id),
        booking_key=booking_key,
    )
    start = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc)
    await provider.prepare_eom_first_clean_booking(
        contact_id=str(contact_id),
        scheduled_start=start,
        scheduled_end=end,
        calendar_id=calendar_id,
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
        calendar_id=calendar_id,
        notes=None,
        booking_key=booking_key,
        expected_calendar_event_id=event_id,
        calendar_event_id=event_id,
        actor_id=1,
        actor_name="Juan Canfield",
    )
    return contact_id, str(completed["onboarding_draft_id"])


@pytest.mark.asyncio
async def test_first_clean_booking_persists_resolved_calendar_identity_for_won_loss():
    """A default alias binds before persistence and teardown uses that same ID."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_first_clean_identity_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=contact_id)
        calendar = _FirstCleanIdentityCalendar(
            resolved_calendar_id="office-owner@example.com"
        )
        booking_key = f"office-first-clean-{uuid.uuid4().hex}"
        result = await schedule_eom_first_clean_booking(
            provider,
            calendar,
            EOMFirstCleanBooking(
                contact_id=str(contact_id),
                scheduled_start=datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc),
                scheduled_end=datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc),
                calendar_id=None,
                notes=None,
                booking_key=booking_key,
                actor_id=1,
                actor_name="Juan Canfield",
            ),
        )

        assert result["lead_stage"] == "won"
        assert calendar.resolve_calls == [{"calendar_id": "primary"}]
        assert calendar.create_calls[0]["calendar_id"] == "office-owner@example.com"
        booking = await conn.fetchrow(
            """
            SELECT metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'first_clean_booked'
            """,
            contact_id,
        )
        assert booking is not None
        booking_metadata = _metadata_dict(booking["metadata"])
        assert booking_metadata["calendar_id"] == "office-owner@example.com"

        loss_calendar = _WonLossCalendar()
        await mark_eom_lead_lost_with_won_teardown(
            provider,
            loss_calendar,
            EOMLeadLost(
                contact_id=str(contact_id),
                reason_code="no_response",
                note=None,
                operation_key=f"office-won-loss-{uuid.uuid4().hex}",
                actor_id=1,
                actor_name="Juan Canfield",
            ),
        )
        assert loss_calendar.calls == [
            {
                "calendar_id": "office-owner@example.com",
                "event_id": str(booking_metadata["calendar_event_id"]),
            }
        ]
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_completed_explicit_primary_first_clean_replay_skips_identity_lookup():
    """A completed operation replays its persisted target through an outage."""

    class _UnavailableIdentityCalendar(_FirstCleanIdentityCalendar):
        async def resolve_calendar_id(self, *, calendar_id: str) -> ToolResult:
            self.resolve_calls.append({"calendar_id": calendar_id})
            return ToolResult(
                success=False,
                error="API_ERROR",
                data={"request_phase": "calendar_identity", "status_code": 503},
                message="Calendar identity unavailable",
            )

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_completed_primary_replay_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=contact_id)
        booking_key = f"office-first-clean-{uuid.uuid4().hex}"
        command = EOMFirstCleanBooking(
            contact_id=str(contact_id),
            scheduled_start=datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc),
            calendar_id="primary",
            notes="Bring first-clean checklist",
            booking_key=booking_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        first_calendar = _FirstCleanIdentityCalendar(
            resolved_calendar_id="office-owner@example.com"
        )
        first = await schedule_eom_first_clean_booking(provider, first_calendar, command)
        assert first["idempotent"] is False
        assert first_calendar.resolve_calls == [{"calendar_id": "primary"}]
        requested = await conn.fetchrow(
            """
            SELECT metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND operation_key = $2
              AND event_type = 'first_clean_booking_requested'
            """,
            contact_id,
            booking_key,
        )
        assert requested is not None
        assert _metadata_dict(requested["metadata"])["requested_calendar_id"] == "primary"

        replay_calendar = _UnavailableIdentityCalendar(
            resolved_calendar_id="different-owner@example.com"
        )
        replay = await schedule_eom_first_clean_booking(provider, replay_calendar, command)
        assert replay["idempotent"] is True
        assert replay["calendar_event_id"] == first["calendar_event_id"]
        assert replay["onboarding_draft_id"] == first["onboarding_draft_id"]
        assert replay_calendar.resolve_calls == []
        assert replay_calendar.create_calls == []

        with pytest.raises(EOMLeadConversionError, match="different first clean booking"):
            await schedule_eom_first_clean_booking(
                provider,
                replay_calendar,
                EOMFirstCleanBooking(
                    contact_id=command.contact_id,
                    scheduled_start=command.scheduled_start,
                    scheduled_end=command.scheduled_end,
                    calendar_id="primary",
                    notes="Different notes",
                    booking_key=command.booking_key,
                    actor_id=command.actor_id,
                    actor_name=command.actor_name,
                ),
            )
        assert replay_calendar.resolve_calls == []
        assert replay_calendar.create_calls == []
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_completed_concrete_first_clean_rejects_primary_replay():
    """A changed primary alias cannot replace a concrete original request."""

    class _UnavailableIdentityCalendar(_FirstCleanIdentityCalendar):
        async def resolve_calendar_id(self, *, calendar_id: str) -> ToolResult:
            self.resolve_calls.append({"calendar_id": calendar_id})
            return ToolResult(
                success=False,
                error="API_ERROR",
                data={"request_phase": "calendar_identity", "status_code": 503},
                message="Calendar identity unavailable",
            )

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_concrete_primary_replay_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=contact_id)
        booking_key = f"office-first-clean-{uuid.uuid4().hex}"
        command = EOMFirstCleanBooking(
            contact_id=str(contact_id),
            scheduled_start=datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc),
            calendar_id="team@example.com",
            notes="Bring first-clean checklist",
            booking_key=booking_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        first_calendar = _FirstCleanIdentityCalendar(
            resolved_calendar_id="office-owner@example.com"
        )
        first = await schedule_eom_first_clean_booking(provider, first_calendar, command)
        assert first["idempotent"] is False
        assert first_calendar.resolve_calls == []

        requested = await conn.fetchrow(
            """
            SELECT metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND operation_key = $2
              AND event_type = 'first_clean_booking_requested'
            """,
            contact_id,
            booking_key,
        )
        assert requested is not None
        assert _metadata_dict(requested["metadata"])["requested_calendar_id"] == (
            "team@example.com"
        )

        replay_calendar = _UnavailableIdentityCalendar(
            resolved_calendar_id="team@example.com"
        )
        with pytest.raises(EOMLeadConversionError, match="different first clean booking"):
            await schedule_eom_first_clean_booking(
                provider,
                replay_calendar,
                EOMFirstCleanBooking(
                    contact_id=command.contact_id,
                    scheduled_start=command.scheduled_start,
                    scheduled_end=command.scheduled_end,
                    calendar_id="primary",
                    notes=command.notes,
                    booking_key=command.booking_key,
                    actor_id=command.actor_id,
                    actor_name=command.actor_name,
                ),
            )
        assert replay_calendar.resolve_calls == []
        assert replay_calendar.create_calls == []
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


async def _claim_public_onboarding_token(
    provider,
    *,
    draft_id: str,
    actor_id: int = 7,
    actor_name: str = "Mayra Canfield",
    hmac_secret: str = _PUBLIC_ONBOARDING_SECRET,
) -> tuple[dict[str, object], uuid.UUID]:
    """Claim a won lead's draft and recover only the test bearer UUID."""

    claim = await provider.claim_eom_onboarding_draft(
        draft_id=draft_id,
        actor_id=actor_id,
        actor_name=actor_name,
        public_onboarding_base_url=_PUBLIC_ONBOARDING_URL,
        public_onboarding_hmac_secret=hmac_secret,
    )
    assert claim["claimed"] is True
    link = str(claim["public_onboarding_link"])
    assert link.startswith(f"{_PUBLIC_ONBOARDING_URL}#token=eomob1.")
    bearer = link.partition("#token=")[2]
    return claim, parse_eom_public_onboarding_token(
        token=bearer,
        secret=hmac_secret,
    )


@pytest.mark.asyncio
async def test_public_onboarding_claim_redeems_once_through_the_existing_handoff():
    """One token owns one won lead, then atomically becomes its handoff evidence."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)

        claim, token_id = await _claim_public_onboarding_token(
            provider, draft_id=draft_id
        )
        token_row = await conn.fetchrow(
            """
            SELECT draft_id, contact_id, approval_key, status,
                   approved_by_employee_id, approved_by_name, handoff_id,
                   signing_key_fingerprint, prefill_full_name, prefill_email,
                   prefill_phone, prefill_address, prefill_city, prefill_state,
                   prefill_zip, prefill_customer_type
            FROM eom_public_onboarding_tokens
            WHERE id = $1
            """,
            token_id,
        )
        assert token_row is not None
        assert dict(token_row) == {
            "draft_id": uuid.UUID(draft_id),
            "contact_id": contact_id,
            "approval_key": token_row["approval_key"],
            "status": "issued",
            "approved_by_employee_id": 7,
            "approved_by_name": "Mayra Canfield",
            "handoff_id": None,
            "signing_key_fingerprint": _PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            "prefill_full_name": "Won Lead",
            "prefill_email": "won-lead@example.com",
            "prefill_phone": None,
            "prefill_address": None,
            "prefill_city": None,
            "prefill_state": None,
            "prefill_zip": None,
            "prefill_customer_type": "unknown",
        }
        assert await conn.fetchval(
            "SELECT body FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            draft_id,
        ) == claim["draft"]["body"]
        assert "#token=" not in str(claim["draft"]["body"])

        await conn.execute(
            """
            UPDATE contacts
               SET full_name = 'Corrected Lead', email = 'corrected@example.com',
                   phone = '2175550199', address = '200 Updated St',
                   city = 'Chicago', state = 'IL', zip = '60601',
                   customer_type = 'commercial'
             WHERE id = $1
            """,
            contact_id,
        )
        with pytest.raises(EOMLeadConversionError) as mismatched_verifier:
            await provider.get_eom_public_onboarding_session(
                token_id=str(token_id),
                signing_key_fingerprint=eom_public_onboarding_hmac_key_fingerprint(
                    secret="different-test-only-public-onboarding-secret-value-987654"
                ),
            )
        assert mismatched_verifier.value.status_code == 404

        tracker_context = await provider.get_eom_public_onboarding_tracker_context(
            token_id=str(token_id),
            signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
        )
        assert tracker_context == {
            "status": "ready",
            "token_id": str(token_id),
            "draft_id": draft_id,
            "contact_id": str(contact_id),
            "full_name": "Won Lead",
            "email": "won-lead@example.com",
            "phone": None,
            "address": None,
            "city": None,
            "state": None,
            "zip": None,
            "customer_type": "unknown",
        }
        ready = await provider.get_eom_public_onboarding_session(
            token_id=str(token_id),
            signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
        )
        assert ready == {
            "status": "ready",
            "contact_id": str(contact_id),
            "full_name": "Won Lead",
            "email": "won-lead@example.com",
            "phone": None,
            "address": None,
            "city": None,
            "state": None,
            "zip": None,
            "customer_type": "unknown",
        }

        completed = await provider.complete_eom_public_onboarding(
            token_id=str(token_id),
            signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            tracker_customer_id=101,
            tracker_site_id=202,
        )
        assert completed["status"] == "completed"
        assert completed["idempotent"] is False
        assert completed["contact_id"] == str(contact_id)
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
        lifecycle_metadata = _metadata_dict(
            await conn.fetchval(
                """
                SELECT metadata
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = 'customer_approved'
                  AND operation_key = $2
                """,
                contact_id,
                token_row["approval_key"],
            )
        )
        assert lifecycle_metadata["completion_channel"] == "public_onboarding"
        assert lifecycle_metadata["approved_by_employee_id"] == 7
        redeemed_row = await conn.fetchrow(
            """
            SELECT status, handoff_id IS NOT NULL AS has_handoff,
                   redeemed_at IS NOT NULL AS is_redeemed
            FROM eom_public_onboarding_tokens WHERE id = $1
            """,
            token_id,
        )
        assert dict(redeemed_row) == {
            "status": "redeemed",
            "has_handoff": True,
            "is_redeemed": True,
        }

        replay = await provider.complete_eom_public_onboarding(
            token_id=str(token_id),
            signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            tracker_customer_id=101,
            tracker_site_id=202,
        )
        assert replay == {**completed, "idempotent": True}
        assert await provider.get_eom_public_onboarding_session(
            token_id=str(token_id),
            signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
        ) == {
            "status": "completed",
            "contact_id": str(contact_id),
            "tracker_customer_id": 101,
            "tracker_site_id": 202,
            "handoff_id": completed["handoff_id"],
            "idempotent": True,
        }
        with pytest.raises(EOMLeadConversionError, match="different tracker records"):
            await provider.complete_eom_public_onboarding(
                token_id=str(token_id),
                signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
                tracker_customer_id=101,
                tracker_site_id=203,
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_recovery_finishes_each_durable_token_state_once():
    """Staff recovery needs stored IDs, not the raw browser bearer."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_recovery_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        provider = DatabaseCRMProvider(pool=conn)

        issued_contact_id, issued_draft_id = await _book_first_clean_draft(
            conn, provider
        )
        _, issued_token_id = await _claim_public_onboarding_token(
            provider, draft_id=issued_draft_id
        )
        recovered = await provider.recover_eom_public_onboarding(
            token_id=str(issued_token_id),
            contact_id=str(issued_contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert recovered["status"] == "completed"
        assert recovered["contact_id"] == str(issued_contact_id)
        assert recovered["tracker_customer_id"] == 101
        assert recovered["tracker_site_id"] == 202
        assert recovered["idempotent"] is False
        assert await _contact_state(conn, issued_contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
                "lead_stage": None,
                "status": "active",
            },
            1,
            1,
        )
        assert dict(
            await conn.fetchrow(
                """
                SELECT status, handoff_id, revoked_at IS NOT NULL AS is_revoked
                FROM eom_public_onboarding_tokens WHERE id = $1
                """,
                issued_token_id,
            )
        ) == {
            "status": "revoked",
            "handoff_id": None,
            "is_revoked": True,
        }
        recovery_event = await conn.fetchrow(
            """
            SELECT actor, operation_key, metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'customer_approved'
            """,
            issued_contact_id,
        )
        assert recovery_event is not None
        assert recovery_event["actor"] == "employee:1:Juan Canfield"
        assert recovery_event["operation_key"] == (
            f"eom-public-onboarding-recovery:{issued_token_id}"
        )
        assert _metadata_dict(recovery_event["metadata"])["completion_channel"] == "office"
        replay = await provider.recover_eom_public_onboarding(
            token_id=str(issued_token_id),
            contact_id=str(issued_contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert replay == {**recovered, "idempotent": True}
        with pytest.raises(EOMLeadConversionError, match="different customer handoff"):
            await provider.recover_eom_public_onboarding(
                token_id=str(issued_token_id),
                contact_id=str(issued_contact_id),
                tracker_customer_id=101,
                tracker_site_id=203,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        redeemed_contact_id, redeemed_draft_id = await _book_first_clean_draft(
            conn, provider
        )
        _, redeemed_token_id = await _claim_public_onboarding_token(
            provider, draft_id=redeemed_draft_id
        )
        await provider.complete_eom_public_onboarding(
            token_id=str(redeemed_token_id),
            signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            tracker_customer_id=303,
            tracker_site_id=404,
        )
        redeemed_replay = await provider.recover_eom_public_onboarding(
            token_id=str(redeemed_token_id),
            contact_id=str(redeemed_contact_id),
            tracker_customer_id=303,
            tracker_site_id=404,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert redeemed_replay["status"] == "completed"
        assert redeemed_replay["idempotent"] is True
        assert (
            await conn.fetchval(
                "SELECT status FROM eom_public_onboarding_tokens WHERE id = $1",
                redeemed_token_id,
            )
            == "redeemed"
        )

        revoked_contact_id, revoked_draft_id = await _book_first_clean_draft(
            conn, provider
        )
        _, revoked_token_id = await _claim_public_onboarding_token(
            provider, draft_id=revoked_draft_id
        )
        await provider.revoke_eom_public_onboarding_token(draft_id=revoked_draft_id)
        revoked_recovery = await provider.recover_eom_public_onboarding(
            token_id=str(revoked_token_id),
            contact_id=str(revoked_contact_id),
            tracker_customer_id=505,
            tracker_site_id=606,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert revoked_recovery["status"] == "completed"
        assert revoked_recovery["idempotent"] is False
        assert await _contact_state(conn, revoked_contact_id) == (
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
async def test_public_onboarding_recovery_rolls_back_token_revocation_on_handoff_conflict():
    """A failed recovery leaves its issued token retryable rather than stranded."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_recovery_rollback_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        _, token_id = await _claim_public_onboarding_token(provider, draft_id=draft_id)

        conflicting_contact_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=conflicting_contact_id)
        await provider.finalize_eom_customer_handoff(
            contact_id=str(conflicting_contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=_approval_key(),
            actor_id=1,
            actor_name="Juan Canfield",
        )

        with pytest.raises(EOMLeadConversionError, match="Tracker Customer or Site"):
            await provider.recover_eom_public_onboarding(
                token_id=str(token_id),
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                actor_id=1,
                actor_name="Juan Canfield",
            )

        assert (
            await conn.fetchval(
                "SELECT status FROM eom_public_onboarding_tokens WHERE id = $1",
                token_id,
            )
            == "issued"
        )
        assert await _contact_state(conn, contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "lead",
                "lead_stage": "won",
                "status": "active",
            },
            0,
            0,
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_recovery_and_bearer_completion_serialize_one_handoff():
    """The shared locks admit either winner, never two handoffs or a deadlock."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_recovery_race_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    pool = None
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=2,
            server_settings={"search_path": f"{schema}, public"},
        )
        provider = DatabaseCRMProvider(pool=pool)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        _, token_id = await _claim_public_onboarding_token(provider, draft_id=draft_id)

        public_result, recovery_result = await asyncio.gather(
            provider.complete_eom_public_onboarding(
                token_id=str(token_id),
                signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
                tracker_customer_id=101,
                tracker_site_id=202,
            ),
            provider.recover_eom_public_onboarding(
                token_id=str(token_id),
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                actor_id=1,
                actor_name="Juan Canfield",
            ),
            return_exceptions=True,
        )

        token_status = await conn.fetchval(
            "SELECT status FROM eom_public_onboarding_tokens WHERE id = $1",
            token_id,
        )
        assert token_status in ("redeemed", "revoked")
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
        if token_status == "redeemed":
            assert isinstance(public_result, dict)
            assert public_result["status"] == "completed"
            assert isinstance(recovery_result, dict)
            assert recovery_result["idempotent"] is True
        else:
            assert isinstance(recovery_result, dict)
            assert recovery_result["status"] == "completed"
            assert isinstance(public_result, EOMLeadConversionError)
            assert public_result.status_code == 409
    finally:
        if pool is not None:
            await pool.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_session_accepts_only_the_key_that_minted_the_row():
    """A controlled rotation keeps an old link valid without cross-key replay."""

    previous_secret = "previous-test-only-public-onboarding-secret-value-654321"
    previous_fingerprint = eom_public_onboarding_hmac_key_fingerprint(
        secret=previous_secret
    )
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_key_rotation_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        provider = DatabaseCRMProvider(pool=conn)
        _, draft_id = await _book_first_clean_draft(conn, provider)
        _, token_id = await _claim_public_onboarding_token(
            provider,
            draft_id=draft_id,
            hmac_secret=previous_secret,
        )

        ready = await provider.get_eom_public_onboarding_session(
            token_id=str(token_id),
            signing_key_fingerprint=previous_fingerprint,
        )
        assert ready["status"] == "ready"
        with pytest.raises(EOMLeadConversionError) as wrong_key:
            await provider.get_eom_public_onboarding_session(
                token_id=str(token_id),
                signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            )
        assert wrong_key.value.status_code == 404
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_claim_waits_for_the_office_contact_lock_before_minting():
    """Issuance cannot mint a link after an office handoff has started."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_claim_lock_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    lock_conn = None
    pool = None
    claim_task = None
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        setup_provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, setup_provider)
        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=1,
            server_settings={"search_path": f"{schema}, public"},
        )
        issuer = DatabaseCRMProvider(pool=pool)
        lock_conn = await asyncpg.connect(database_url)

        async with lock_conn.transaction():
            await lock_conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"eom-customer-handoff:contact:{contact_id}",
            )
            claim_task = asyncio.create_task(
                _claim_public_onboarding_token(issuer, draft_id=draft_id)
            )
            waiting_for_contact_lock = False
            for _ in range(100):
                waiting_for_contact_lock = bool(
                    await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_stat_activity
                            WHERE datname = current_database()
                              AND pid <> pg_backend_pid()
                              AND query LIKE 'SELECT pg_advisory_xact_lock%'
                              AND wait_event_type = 'Lock'
                        )
                        """
                    )
                )
                if waiting_for_contact_lock:
                    break
                await asyncio.sleep(0.01)
            assert waiting_for_contact_lock
            assert not claim_task.done()
            assert (
                await conn.fetchval(
                    "SELECT status FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
                    draft_id,
                )
                == "pending"
            )
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM eom_public_onboarding_tokens"
                )
                == 0
            )

        claim, token_id = await claim_task
        assert claim["claimed"] is True
        assert await conn.fetchval(
            "SELECT status FROM eom_public_onboarding_tokens WHERE id = $1",
            token_id,
        ) == "issued"
        with pytest.raises(EOMLeadConversionError, match="active public onboarding"):
            await issuer.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            )
    finally:
        if claim_task is not None and not claim_task.done():
            claim_task.cancel()
            with suppress(asyncio.CancelledError):
                await claim_task
        if lock_conn is not None:
            await lock_conn.close()
        if pool is not None:
            await pool.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_revocation_releases_office_handoff_and_stale_draft_revoke():
    """Either explicit link revocation or stale-draft reconciliation fences safely."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_revoke_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        _, token_id = await _claim_public_onboarding_token(provider, draft_id=draft_id)

        with pytest.raises(EOMLeadConversionError, match="active public onboarding"):
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=101,
                tracker_site_id=202,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            )
        revoked = await provider.revoke_eom_public_onboarding_token(draft_id=draft_id)
        assert revoked["status"] == "revoked"
        assert revoked["idempotent"] is False
        assert (
            await conn.fetchval(
                "SELECT status FROM eom_public_onboarding_tokens WHERE id = $1",
                token_id,
            )
            == "revoked"
        )
        with pytest.raises(EOMLeadConversionError) as unavailable:
            await provider.get_eom_public_onboarding_session(
                token_id=str(token_id),
                signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            )
        assert unavailable.value.status_code == 404

        office = await provider.finalize_eom_customer_handoff(
            contact_id=str(contact_id),
            tracker_customer_id=101,
            tracker_site_id=202,
            approval_key=_approval_key(),
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert office["idempotent"] is False

        stale_contact_id, stale_draft_id = await _book_first_clean_draft(conn, provider)
        _, stale_token_id = await _claim_public_onboarding_token(
            provider, draft_id=stale_draft_id
        )
        await conn.execute(
            """
            UPDATE eom_onboarding_email_drafts
               SET claimed_at = NOW() - INTERVAL '20 minutes'
             WHERE id = $1::uuid
            """,
            stale_draft_id,
        )
        stale_revoke = await provider.revoke_eom_onboarding_draft(
            draft_id=stale_draft_id
        )
        assert stale_revoke["status"] == "revoked"
        assert (
            await conn.fetchval(
                "SELECT status FROM eom_public_onboarding_tokens WHERE id = $1",
                stale_token_id,
            )
            == "revoked"
        )
        assert await _contact_state(conn, stale_contact_id) == (
            {
                "business_context_id": "effingham_maids",
                "contact_type": "lead",
                "lead_stage": "won",
                "status": "active",
            },
            0,
            0,
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_readiness_requires_its_migration_only_when_enabled():
    """A dormant deploy may lack the table or issuance-only fields; enabled
    issuance fails closed until the full token shape exists."""
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store
    from atlas_brain.storage.migrations import run_migrations

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_readiness_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema)
        provider = DatabaseCRMProvider(pool=conn)

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await conn.fetchval(query))

        class _MigrationPool:
            async def acquire(self):
                return conn

            async def release(self, released) -> None:
                assert released is conn

        disabled = type(
            "Config", (), {"api_enabled": True, "public_onboarding_enabled": False}
        )()
        enabled = type(
            "Config", (), {"api_enabled": True, "public_onboarding_enabled": True}
        )()
        await require_eom_funnel_data_store(
            disabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                enabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
            )
        with pytest.raises(EOMLeadConversionError, match="token storage") as exc_info:
            await provider.revoke_eom_public_onboarding_token(draft_id=str(uuid.uuid4()))
        assert exc_info.value.status_code == 503

        await run_migrations(
            _MigrationPool(),
            migrations_dir=MIGRATIONS,
            only={"383_eom_public_onboarding_tokens"},
        )
        assert await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM schema_migrations
                WHERE name = '383_eom_public_onboarding_tokens'
            )
            """
        )
        await require_eom_funnel_data_store(
            disabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )
        await require_eom_funnel_data_store(
            enabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )
        await conn.execute(
            "ALTER TABLE eom_public_onboarding_tokens DROP COLUMN prefill_email"
        )
        await require_eom_funnel_data_store(
            disabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                enabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_disabled_public_onboarding_readiness_requires_present_recovery_columns():
    """Once migration 383 exists, dormant startup still protects the columns
    that the office fence and private revoke-link recovery command use."""
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_recovery_columns_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await conn.fetchval(query))

        disabled = type(
            "Config", (), {"api_enabled": True, "public_onboarding_enabled": False}
        )()
        await require_eom_funnel_data_store(
            disabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )
        await conn.execute(
            "ALTER TABLE eom_public_onboarding_tokens "
            "RENAME COLUMN revoked_at TO missing_revoked_at"
        )
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                disabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_disabled_public_onboarding_readiness_requires_present_recovery_privileges():
    """A deployed token relation must grant the runtime its fence/revoke
    SELECT and UPDATE surface even when issuance is disabled."""
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_recovery_privileges_{uuid.uuid4().hex}"
    runtime_role = f"atlas_eom_public_token_runtime_{uuid.uuid4().hex}"
    schema_ident = _quote_ident(schema)
    runtime_ident = _quote_ident(runtime_role)
    database_name: str | None = None
    role_created = False
    runtime_conn = None
    conn = await asyncpg.connect(database_url)
    try:
        await _require_disposable_role_administration(conn)
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        database_name = await conn.fetchval("SELECT current_database()")
        await conn.execute(
            f"CREATE ROLE {runtime_ident} LOGIN NOINHERIT "
            f"PASSWORD '{_NON_SUPERUSER_TEST_PASSWORD}'"
        )
        role_created = True
        await conn.execute(
            f"GRANT CONNECT ON DATABASE {_quote_ident(database_name)} TO {runtime_ident}"
        )
        await conn.execute(f"GRANT USAGE ON SCHEMA {schema_ident} TO {runtime_ident}")

        runtime_conn = await asyncpg.connect(
            database_url,
            user=runtime_role,
            password=_NON_SUPERUSER_TEST_PASSWORD,
        )
        await runtime_conn.execute(f"SET search_path TO {schema_ident}, public")

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await runtime_conn.fetchval(query))

        disabled = type(
            "Config", (), {"api_enabled": True, "public_onboarding_enabled": False}
        )()
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                disabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
            )

        await conn.execute(
            "GRANT SELECT ON TABLE "
            f"{schema_ident}.eom_public_onboarding_tokens TO {runtime_ident}"
        )
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                disabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
            )

        await conn.execute(
            "GRANT UPDATE ON TABLE "
            f"{schema_ident}.eom_public_onboarding_tokens TO {runtime_ident}"
        )
        assert not await runtime_conn.fetchval(
            "SELECT has_table_privilege("
            "current_user, 'eom_public_onboarding_tokens', 'INSERT')"
        )
        await require_eom_funnel_data_store(
            disabled, database_enabled=True, get_db_pool_fn=lambda: _Pool()
        )
    finally:
        if runtime_conn is not None:
            await runtime_conn.close()
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema_ident} CASCADE")
        if role_created:
            assert database_name is not None
            await conn.execute(
                f"REVOKE CONNECT ON DATABASE {_quote_ident(database_name)} "
                f"FROM {runtime_ident}"
            )
            await conn.execute(f"DROP ROLE IF EXISTS {runtime_ident}")
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_and_office_finalizers_serialize_on_one_contact():
    """An issued link wins the shared contact decision; office cannot make a second handoff."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_race_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    pool = None
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=2,
            server_settings={"search_path": f"{schema}, public"},
        )
        provider = DatabaseCRMProvider(pool=pool)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        _, token_id = await _claim_public_onboarding_token(provider, draft_id=draft_id)

        public_result, office_result = await asyncio.gather(
            provider.complete_eom_public_onboarding(
                token_id=str(token_id),
                signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
                tracker_customer_id=101,
                tracker_site_id=202,
            ),
            provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=303,
                tracker_site_id=404,
                approval_key=_approval_key(),
                actor_id=1,
                actor_name="Juan Canfield",
            ),
            return_exceptions=True,
        )
        assert isinstance(public_result, dict)
        assert public_result["status"] == "completed"
        assert isinstance(office_result, EOMLeadConversionError)
        assert office_result.status_code == 409
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
        assert (
            await conn.fetchval(
                "SELECT status FROM eom_public_onboarding_tokens WHERE id = $1",
                token_id,
            )
            == "redeemed"
        )
    finally:
        if pool is not None:
            await pool.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


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
async def test_public_onboarding_issued_link_projection_excludes_terminal_tokens():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_issued_links_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        provider = DatabaseCRMProvider(pool=conn)
        issued_contact_id, issued_draft_id = await _book_first_clean_draft(
            conn, provider, full_name="Issued Link"
        )
        _, issued_token_id = await _claim_public_onboarding_token(
            provider, draft_id=issued_draft_id
        )
        redeemed_contact_id, redeemed_draft_id = await _book_first_clean_draft(
            conn, provider, full_name="Redeemed Link"
        )
        _, redeemed_token_id = await _claim_public_onboarding_token(
            provider, draft_id=redeemed_draft_id
        )
        await conn.execute(
            """
            UPDATE eom_public_onboarding_tokens
               SET status = 'redeemed', redeemed_at = NOW(), handoff_id = $2::uuid
             WHERE id = $1::uuid
            """,
            redeemed_token_id,
            uuid.uuid4(),
        )
        revoked_contact_id, revoked_draft_id = await _book_first_clean_draft(
            conn, provider, full_name="Revoked Link"
        )
        _, revoked_token_id = await _claim_public_onboarding_token(
            provider, draft_id=revoked_draft_id
        )
        revoked = await provider.revoke_eom_public_onboarding_token(
            draft_id=revoked_draft_id
        )

        links = await provider.list_eom_public_onboarding_issued_links(
            accepted_signing_key_fingerprints=(_PUBLIC_ONBOARDING_KEY_FINGERPRINT,),
            limit=100,
        )

        assert [{str(row["draft_id"]), str(row["contact_id"])} for row in links] == [
            {issued_draft_id, str(issued_contact_id)}
        ]
        assert links[0]["full_name"] == "Issued Link"
        assert links[0]["recipient_email"] == "won-lead@example.com"
        assert links[0]["status"] == "issued"
        assert "id" not in links[0]
        assert str(issued_token_id) not in str(links[0])
        assert revoked["status"] == "revoked"
        assert str(redeemed_contact_id) not in str(links)
        assert str(revoked_contact_id) not in str(links)
        assert str(redeemed_token_id) not in str(links)
        assert str(revoked_token_id) not in str(links)
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_issued_link_projection_matches_session_readiness():
    """The office queue includes exactly the rows a public session can open."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_issued_link_readiness_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        provider = DatabaseCRMProvider(pool=conn)

        async def issued(full_name: str) -> tuple[uuid.UUID, str, uuid.UUID]:
            contact_id, draft_id = await _book_first_clean_draft(
                conn, provider, full_name=full_name
            )
            _, token_id = await _claim_public_onboarding_token(
                provider, draft_id=draft_id
            )
            return contact_id, draft_id, token_id

        sending_contact_id, sending_draft_id, sending_token_id = await issued(
            "Sending Link"
        )
        sent_contact_id, sent_draft_id, sent_token_id = await issued("Sent Link")
        inactive_contact_id, _, inactive_token_id = await issued("Inactive Link")
        archived_contact_id, _, archived_token_id = await issued("Archived Link")
        other_context_contact_id, _, other_context_token_id = await issued(
            "Other Context Link"
        )
        non_lead_contact_id, _, non_lead_token_id = await issued("Customer Link")
        non_won_contact_id, _, non_won_token_id = await issued("Non-won Link")
        _, pending_draft_id, pending_token_id = await issued("Pending Draft Link")
        _, revoked_draft_id, revoked_draft_token_id = await issued("Revoked Draft Link")

        await conn.execute(
            "UPDATE eom_onboarding_email_drafts SET status = 'sent', sent_at = NOW() "
            "WHERE id = $1::uuid",
            sent_draft_id,
        )
        await conn.execute(
            "UPDATE contacts SET status = 'inactive' WHERE id = $1", inactive_contact_id
        )
        await conn.execute(
            "UPDATE contacts SET status = 'archived' WHERE id = $1", archived_contact_id
        )
        await conn.execute(
            "UPDATE contacts SET business_context_id = 'other_business' WHERE id = $1",
            other_context_contact_id,
        )
        await conn.execute(
            "UPDATE contacts SET contact_type = 'customer' WHERE id = $1",
            non_lead_contact_id,
        )
        await conn.execute(
            "UPDATE contacts SET lead_stage = 'qualified' WHERE id = $1",
            non_won_contact_id,
        )
        await conn.execute(
            "UPDATE eom_onboarding_email_drafts SET status = 'pending' WHERE id = $1::uuid",
            pending_draft_id,
        )
        await conn.execute(
            "UPDATE eom_onboarding_email_drafts SET status = 'revoked', revoked_at = NOW() "
            "WHERE id = $1::uuid",
            revoked_draft_id,
        )

        for token_id in (
            inactive_token_id,
            archived_token_id,
            other_context_token_id,
            non_lead_token_id,
            non_won_token_id,
            pending_token_id,
            revoked_draft_token_id,
        ):
            with pytest.raises(EOMLeadConversionError) as exc_info:
                await provider.get_eom_public_onboarding_session(
                    token_id=str(token_id),
                    signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
                )
            assert exc_info.value.status_code == 404

        for token_id in (sending_token_id, sent_token_id):
            session = await provider.get_eom_public_onboarding_session(
                token_id=str(token_id),
                signing_key_fingerprint=_PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            )
            assert session["status"] == "ready"

        links = await provider.list_eom_public_onboarding_issued_links(
            accepted_signing_key_fingerprints=(_PUBLIC_ONBOARDING_KEY_FINGERPRINT,),
            limit=100,
        )

        assert {str(row["draft_id"]) for row in links} == {
            sending_draft_id,
            sent_draft_id,
        }
        assert {str(row["contact_id"]) for row in links} == {
            str(sending_contact_id),
            str(sent_contact_id),
        }
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_public_onboarding_issued_link_projection_pages_current_and_previous_keys():
    """Only keys that can authenticate now enter a complete newest-first queue."""

    previous_secret = "previous-test-only-public-onboarding-secret-value-654321"
    retired_secret = "retired-test-only-public-onboarding-secret-value-987654"
    previous_fingerprint = eom_public_onboarding_hmac_key_fingerprint(
        secret=previous_secret
    )
    retired_fingerprint = eom_public_onboarding_hmac_key_fingerprint(
        secret=retired_secret
    )
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_public_onboarding_issued_link_page_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_public_onboarding_migration=True)
        provider = DatabaseCRMProvider(pool=conn)

        newest_contact_id, newest_draft_id = await _book_first_clean_draft(
            conn, provider, full_name="Current Newest"
        )
        _, newest_token_id = await _claim_public_onboarding_token(
            provider, draft_id=newest_draft_id
        )
        previous_contact_id, previous_draft_id = await _book_first_clean_draft(
            conn, provider, full_name="Previous Key"
        )
        _, previous_token_id = await _claim_public_onboarding_token(
            provider,
            draft_id=previous_draft_id,
            hmac_secret=previous_secret,
        )
        older_contact_id, older_draft_id = await _book_first_clean_draft(
            conn, provider, full_name="Current Older"
        )
        _, older_token_id = await _claim_public_onboarding_token(
            provider, draft_id=older_draft_id
        )
        retired_contact_id, retired_draft_id = await _book_first_clean_draft(
            conn, provider, full_name="Retired Key"
        )
        _, retired_token_id = await _claim_public_onboarding_token(
            provider,
            draft_id=retired_draft_id,
            hmac_secret=retired_secret,
        )
        assert retired_fingerprint not in (
            _PUBLIC_ONBOARDING_KEY_FINGERPRINT,
            previous_fingerprint,
        )

        for token_id, issued_at in (
            (newest_token_id, datetime(2026, 8, 19, 12, tzinfo=timezone.utc)),
            (previous_token_id, datetime(2026, 8, 18, 12, tzinfo=timezone.utc)),
            (older_token_id, datetime(2026, 8, 17, 12, tzinfo=timezone.utc)),
            (retired_token_id, datetime(2026, 8, 20, 12, tzinfo=timezone.utc)),
        ):
            await conn.execute(
                "UPDATE eom_public_onboarding_tokens SET issued_at = $2 WHERE id = $1",
                token_id,
                issued_at,
            )

        first_page = await provider.list_eom_public_onboarding_issued_links(
            accepted_signing_key_fingerprints=(
                _PUBLIC_ONBOARDING_KEY_FINGERPRINT,
                previous_fingerprint,
            ),
            limit=2,
        )
        second_page = await provider.list_eom_public_onboarding_issued_links(
            accepted_signing_key_fingerprints=(
                _PUBLIC_ONBOARDING_KEY_FINGERPRINT,
                previous_fingerprint,
            ),
            limit=2,
            cursor_issued_at=first_page[-1]["issued_at"],
            cursor_draft_id=first_page[-1]["draft_id"],
        )

        assert [str(row["draft_id"]) for row in first_page] == [
            newest_draft_id,
            previous_draft_id,
        ]
        assert [str(row["draft_id"]) for row in second_page] == [older_draft_id]
        assert {str(row["contact_id"]) for row in first_page + second_page} == {
            str(newest_contact_id),
            str(previous_contact_id),
            str(older_contact_id),
        }
        assert str(retired_contact_id) not in str(first_page + second_page)
        assert str(retired_token_id) not in str(first_page + second_page)
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_won_lead_loss_cancels_first_clean_and_revokes_pending_draft():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        booking = await conn.fetchrow(
            """
            SELECT metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'first_clean_booked'
            """,
            contact_id,
        )
        assert booking is not None
        booking_metadata = _metadata_dict(booking["metadata"])
        calendar_event_id = str(booking_metadata["calendar_event_id"])
        calendar = _WonLossCalendar()
        command = EOMLeadLost(
            contact_id=str(contact_id),
            reason_code="no_response",
            note="Could not reach customer",
            operation_key=f"office-won-loss-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )

        completed = await mark_eom_lead_lost_with_won_teardown(
            provider, calendar, command
        )

        assert completed == {
            "contact_id": str(contact_id),
            "lead_stage": "lost",
            "status": "lost",
            "reason_code": "no_response",
            "from_stage": "won",
            "idempotent": False,
        }
        assert calendar.calls == [
            {
                "calendar_id": "estimate-calendar",
                "event_id": calendar_event_id,
            }
        ]
        assert await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1", contact_id
        ) == "lost"
        assert await conn.fetchval(
            "SELECT status FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            draft_id,
        ) == "revoked"
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'first_clean_cancelled'
            """,
            contact_id,
        ) == 1
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'lead_lost'
            """,
            contact_id,
        ) == 1

        replay = await mark_eom_lead_lost_with_won_teardown(
            provider, calendar, command
        )
        assert replay["idempotent"] is True
        assert replay["reason_code"] == "no_response"
        assert len(calendar.calls) == 1
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_won_lead_loss_blocks_sending_or_sent_draft_before_calendar_delete():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_delivery_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        for draft_status in ("sending", "sent"):
            contact_id, draft_id = await _book_first_clean_draft(conn, provider)
            await conn.execute(
                """
                UPDATE eom_onboarding_email_drafts
                SET status = $2::varchar,
                    claimed_at = CASE
                        WHEN $2::varchar = 'sending' THEN NOW()
                        ELSE claimed_at
                    END,
                    sent_at = CASE WHEN $2::varchar = 'sent' THEN NOW() ELSE sent_at END
                WHERE id = $1::uuid
                """,
                draft_id,
                draft_status,
            )
            calendar = _WonLossCalendar()
            with pytest.raises(
                EOMLeadConversionError, match="delivery must be reconciled"
            ) as exc:
                await mark_eom_lead_lost_with_won_teardown(
                    provider,
                    calendar,
                    EOMLeadLost(
                        contact_id=str(contact_id),
                        reason_code="no_response",
                        note=None,
                        operation_key=f"office-won-loss-{uuid.uuid4().hex}",
                        actor_id=1,
                        actor_name="Juan Canfield",
                    ),
                )
            assert exc.value.status_code == 409
            assert calendar.calls == []
            assert await conn.fetchval(
                "SELECT lead_stage FROM contacts WHERE id = $1", contact_id
            ) == "won"
            assert await conn.fetchval(
                "SELECT status FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
                draft_id,
            ) == draft_status
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pre_won_stage", "reason_code"),
    (("new", "spam"), ("estimate_booked", "declined_after_estimate")),
)
async def test_won_lead_loss_rejects_reused_prewon_lost_key_before_calendar_delete(
    pre_won_stage: str, reason_code: str
):
    """A pre-won loss key remains permanently incompatible with won teardown."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_reused_key_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(
            conn,
            contact_id=contact_id,
            lead_stage=pre_won_stage,
            email="reused-key@example.com",
        )
        legacy_loss_key = f"office-legacy-loss-{uuid.uuid4().hex}"
        lost = await provider.mark_eom_lead_lost(
            contact_id=str(contact_id),
            reason_code=reason_code,
            note="Original pre-won disposition",
            operation_key=legacy_loss_key,
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert lost["from_stage"] == pre_won_stage
        reopened = await provider.reopen_eom_lead(
            contact_id=str(contact_id),
            operation_key=f"office-reopen-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        assert reopened["lead_stage"] == pre_won_stage
        _, draft_id = await _book_first_clean_draft(
            conn, provider, contact_id=contact_id
        )
        calendar = _WonLossCalendar()

        with pytest.raises(
            EOMLeadConversionError,
            match="Idempotency-Key already belongs to another EOM operation",
        ) as exc:
            await mark_eom_lead_lost_with_won_teardown(
                provider,
                calendar,
                EOMLeadLost(
                    contact_id=str(contact_id),
                    reason_code="no_response",
                    note=None,
                    operation_key=legacy_loss_key,
                    actor_id=1,
                    actor_name="Juan Canfield",
                ),
            )

        assert exc.value.status_code == 409
        assert calendar.calls == []
        assert await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1", contact_id
        ) == "won"
        assert await conn.fetchval(
            "SELECT status FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            draft_id,
        ) == "pending"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_won_lead_loss_rejects_second_key_while_cancellation_is_unsettled():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_retry_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        command = EOMLeadLost(
            contact_id=str(contact_id),
            reason_code="no_response",
            note=None,
            operation_key=f"office-won-loss-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        calendar = _WonLossCalendar(
            results=[
                ToolResult(
                    success=False,
                    error="API_ERROR",
                    data={"request_phase": "delete", "status_code": 503},
                    message="Calendar API error: 503",
                ),
                ToolResult(
                    success=True,
                    data={"already_absent": True},
                    message="Calendar event was already absent",
                ),
            ]
        )

        with pytest.raises(
            EOMLeadConversionError, match="Calendar API error: 503"
        ) as exc:
            await mark_eom_lead_lost_with_won_teardown(provider, calendar, command)
        assert exc.value.status_code == 502
        assert await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1", contact_id
        ) == "won"
        assert await conn.fetchval(
            "SELECT status FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            draft_id,
        ) == "pending"
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'first_clean_cancellation_calendar_unsettled'
            """,
            contact_id,
        ) == 1

        with pytest.raises(
            EOMLeadConversionError, match="cancellation requires reconciliation"
        ) as second_key:
            await mark_eom_lead_lost_with_won_teardown(
                provider,
                calendar,
                EOMLeadLost(
                    contact_id=str(contact_id),
                    reason_code="no_response",
                    note=None,
                    operation_key=f"office-won-loss-{uuid.uuid4().hex}",
                    actor_id=1,
                    actor_name="Juan Canfield",
                ),
            )
        assert second_key.value.status_code == 409
        assert len(calendar.calls) == 1
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'first_clean_cancellation_requested'
            """,
            contact_id,
        ) == 1

        completed = await mark_eom_lead_lost_with_won_teardown(
            provider, calendar, command
        )
        assert completed["idempotent"] is False
        assert await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1", contact_id
        ) == "lost"
        assert await conn.fetchval(
            "SELECT status FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            draft_id,
        ) == "revoked"
        assert len(calendar.calls) == 2
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_won_lead_loss_fences_durable_unsettled_cancellation_from_claim_and_handoff():
    """A released executor lock does not erase unfinished cancellation work."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_durable_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(
            conn,
            schema,
            apply_privilege_migration=False,
            apply_public_onboarding_migration=True,
        )
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        command = EOMLeadLost(
            contact_id=str(contact_id),
            reason_code="no_response",
            note=None,
            operation_key=f"office-won-loss-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        calendar = _WonLossCalendar(
            results=[
                ToolResult(
                    success=False,
                    error="API_ERROR",
                    data={"request_phase": "delete", "status_code": 503},
                    message="Calendar API error: 503",
                )
            ]
        )

        with pytest.raises(EOMLeadConversionError, match="Calendar API error: 503"):
            await mark_eom_lead_lost_with_won_teardown(provider, calendar, command)

        with pytest.raises(
            EOMLeadConversionError, match="cancellation requires reconciliation"
        ) as standard_claim:
            await provider.claim_eom_onboarding_draft(
                draft_id=draft_id,
                actor_id=2,
                actor_name="Mayra Canfield",
            )
        assert standard_claim.value.status_code == 409

        with pytest.raises(
            EOMLeadConversionError, match="cancellation requires reconciliation"
        ) as public_claim:
            await provider.claim_eom_onboarding_draft(
                draft_id=draft_id,
                actor_id=2,
                actor_name="Mayra Canfield",
                public_onboarding_base_url=_PUBLIC_ONBOARDING_URL,
                public_onboarding_hmac_secret=_PUBLIC_ONBOARDING_SECRET,
            )
        assert public_claim.value.status_code == 409

        with pytest.raises(
            EOMLeadConversionError, match="cancellation requires reconciliation"
        ) as handoff:
            await provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=981,
                tracker_site_id=1981,
                approval_key=f"office-handoff-{uuid.uuid4().hex}",
                actor_id=2,
                actor_name="Mayra Canfield",
            )
        assert handoff.value.status_code == 409
        assert len(calendar.calls) == 1
        assert await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1", contact_id
        ) == "won"
        assert await conn.fetchval(
            "SELECT status FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            draft_id,
        ) == "pending"
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM eom_public_onboarding_tokens WHERE draft_id = $1::uuid",
            draft_id,
        ) == 0
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1::uuid",
            contact_id,
        ) == 0
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_won_lead_loss_fences_durable_cancellation_from_generic_contact_status_writes():
    """Archive and generic status writes cannot strand Calendar teardown."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_generic_status_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, _ = await _book_first_clean_draft(conn, provider)
        command = EOMLeadLost(
            contact_id=str(contact_id),
            reason_code="no_response",
            note=None,
            operation_key=f"office-won-loss-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        calendar = _WonLossCalendar(
            results=[
                ToolResult(
                    success=False,
                    error="API_ERROR",
                    data={"request_phase": "delete", "status_code": 503},
                    message="Calendar API error: 503",
                )
            ]
        )

        with pytest.raises(EOMLeadConversionError, match="Calendar API error: 503"):
            await mark_eom_lead_lost_with_won_teardown(provider, calendar, command)

        with pytest.raises(
            EOMLeadConversionError, match="cancellation requires reconciliation"
        ) as archive:
            await provider.delete_contact(str(contact_id))
        assert archive.value.status_code == 409

        with pytest.raises(
            EOMLeadConversionError, match="cancellation requires reconciliation"
        ) as status_update:
            await provider.update_contact(str(contact_id), {"status": "inactive"})
        assert status_update.value.status_code == 409
        assert await conn.fetchval(
            "SELECT status FROM contacts WHERE id = $1", contact_id
        ) == "active"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def _legacy_386_function_body() -> str:
    source = (MIGRATIONS / "386_eom_won_loss_nocodb_fence.sql").read_text()
    marker = "AS $function$"
    body_start = source.index(marker) + len(marker)
    body_end = source.index("$function$;", body_start)
    return (
        source[body_start:body_end]
        .replace(
            "IF OLD.business_context_id = 'effingham_maids'",
            "IF session_user = 'atlas_nocodb'\n"
            "               AND OLD.business_context_id = 'effingham_maids'",
        )
        .replace(
            "before direct contact mutation",
            "before NocoDB can change the contact",
        )
    )


async def _install_legacy_386_nocodb_fence(conn, schema: str) -> None:
    schema_ident = _quote_ident(schema)
    function_ident = f"{schema_ident}.reject_nocodb_eom_won_loss_mutation"
    await conn.execute(
        "CREATE OR REPLACE FUNCTION "
        f"{function_ident}() RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER "
        f"SET search_path = pg_catalog, {schema_ident} AS $function$"
        f"{_legacy_386_function_body()}$function$;"
    )
    await conn.execute(
        "DROP TRIGGER IF EXISTS trg_reject_nocodb_eom_won_loss_mutation "
        f"ON {schema_ident}.contacts"
    )
    await conn.execute(
        "CREATE TRIGGER trg_reject_nocodb_eom_won_loss_mutation "
        "BEFORE UPDATE OF status OR DELETE ON "
        f"{schema_ident}.contacts FOR EACH ROW EXECUTE FUNCTION {function_ident}()"
    )
    await conn.execute(f"REVOKE ALL ON FUNCTION {function_ident}() FROM PUBLIC")


@pytest.mark.asyncio
async def test_nocodb_cannot_mutate_won_lead_with_unsettled_cancellation():
    """The direct CRM login cannot bypass a prepared won-loss cancellation."""

    from atlas_brain.storage.migrations import run_migrations
    from atlas_brain.storage.migrations.reconciliation import (
        MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY,
    )

    record = MIGRATION_386_WON_LOSS_FENCE_FORWARD_RECOVERY
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_nocodb_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    nocodb_conn = None
    direct_conn = None
    try:
        await _prepare_schema(conn, schema)
        await _install_legacy_386_nocodb_fence(conn, schema)
        await conn.execute(
            "ALTER TABLE schema_migrations "
            "ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64)"
        )
        await conn.execute(
            """
            INSERT INTO schema_migrations (version, name, content_sha256, applied_at)
            VALUES ($1, $2, $3, $4)
            """,
            386,
            record.migration_name,
            record.historical_ledger_sha256,
            record.observed_applied_at,
        )

        class _MigrationPool:
            async def acquire(self):
                return conn

            async def release(self, released) -> None:
                assert released is conn

        await run_migrations(
            _MigrationPool(),
            migrations_dir=MIGRATIONS,
            only={record.recovery_migration_name},
        )
        assert await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE name = $1)",
            record.recovery_migration_name,
        )

        provider = DatabaseCRMProvider(pool=conn)
        contact_id, _ = await _book_first_clean_draft(conn, provider)
        command = EOMLeadLost(
            contact_id=str(contact_id),
            reason_code="no_response",
            note=None,
            operation_key=f"office-won-loss-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        calendar = _BlockingWonLossCalendar(
            results=[
                ToolResult(
                    success=False,
                    error="API_ERROR",
                    data={"request_phase": "delete", "status_code": 503},
                    message="Calendar API error: 503",
                ),
                ToolResult(
                    success=True,
                    data={"already_absent": True},
                    message="Calendar event was already absent",
                ),
            ]
        )

        nocodb_conn = await asyncpg.connect(
            database_url,
            user="atlas_nocodb",
            password=_NOCODB_TEST_PASSWORD,
        )
        await nocodb_conn.execute(
            f"SET search_path TO {_quote_ident(schema)}, public"
        )
        assert await nocodb_conn.fetchval("SELECT session_user") == "atlas_nocodb"
        direct_conn = await asyncpg.connect(database_url)
        await direct_conn.execute(f"SET search_path TO {_quote_ident(schema)}, public")
        assert await direct_conn.fetchval("SELECT session_user") != "atlas_nocodb"

        loss_task = asyncio.create_task(
            mark_eom_lead_lost_with_won_teardown(provider, calendar, command)
        )
        await asyncio.wait_for(calendar.started.wait(), timeout=3)

        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="cancellation requires reconciliation",
        ):
            await nocodb_conn.execute(
                "UPDATE contacts SET status = 'inactive' WHERE id = $1",
                contact_id,
            )
        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="cancellation requires reconciliation",
        ):
            await nocodb_conn.execute("DELETE FROM contacts WHERE id = $1", contact_id)
        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="cancellation requires reconciliation",
        ):
            await direct_conn.execute(
                "UPDATE contacts SET status = 'inactive' WHERE id = $1", contact_id
            )
        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="cancellation requires reconciliation",
        ):
            await direct_conn.execute(
                "UPDATE contacts SET contact_type = 'customer' WHERE id = $1",
                contact_id,
            )
        assert not loss_task.done()

        calendar.release.set()
        with pytest.raises(EOMLeadConversionError, match="Calendar API error: 503"):
            await loss_task

        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="cancellation requires reconciliation",
        ):
            await nocodb_conn.execute(
                "UPDATE contacts SET status = 'inactive' WHERE id = $1",
                contact_id,
            )
        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="cancellation requires reconciliation",
        ):
            await nocodb_conn.execute("DELETE FROM contacts WHERE id = $1", contact_id)
        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="cancellation requires reconciliation",
        ):
            await direct_conn.execute(
                "UPDATE contacts SET status = 'inactive' WHERE id = $1", contact_id
            )
        with pytest.raises(
            asyncpg.exceptions.RaiseError,
            match="cancellation requires reconciliation",
        ):
            await direct_conn.execute(
                "UPDATE contacts SET contact_type = 'customer' WHERE id = $1",
                contact_id,
            )

        await nocodb_conn.execute(
            "UPDATE contacts SET notes = 'ordinary NocoDB edit' WHERE id = $1",
            contact_id,
        )
        assert await conn.fetchval(
            "SELECT status FROM contacts WHERE id = $1", contact_id
        ) == "active"
        assert await conn.fetchval(
            "SELECT notes FROM contacts WHERE id = $1", contact_id
        ) == "ordinary NocoDB edit"

        completed = await mark_eom_lead_lost_with_won_teardown(
            provider, calendar, command
        )
        assert completed["lead_stage"] == "lost"
        assert await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1", contact_id
        ) == "lost"
        await direct_conn.execute(
            "UPDATE contacts SET status = 'inactive' WHERE id = $1", contact_id
        )
        assert await conn.fetchval(
            "SELECT status FROM contacts WHERE id = $1", contact_id
        ) == "inactive"
        await nocodb_conn.execute(
            "UPDATE contacts SET status = 'inactive' WHERE id = $1", contact_id
        )
        assert await conn.fetchval(
            "SELECT status FROM contacts WHERE id = $1", contact_id
        ) == "inactive"
    finally:
        if direct_conn is not None:
            await direct_conn.close()
        if nocodb_conn is not None:
            await nocodb_conn.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("relative_calendar_id", ("primary", "PRIMARY"))
async def test_won_lead_loss_rejects_relative_calendar_identifier_before_delete(
    relative_calendar_id: str,
):
    """A credential-relative booking alias cannot turn another account's 404 into loss."""

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_relative_calendar_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        # Model rows booked before the concrete-identity contract. Both the
        # prepare and completed evidence carried the relative alias then; a
        # current service must reject the historical state rather than creating
        # another event or treating a different principal's 404 as success.
        await conn.execute(
            "ALTER TABLE eom_lead_lifecycle_events DISABLE TRIGGER USER"
        )
        try:
            await conn.execute(
                """
                UPDATE eom_lead_lifecycle_events
                SET metadata = jsonb_set(metadata, '{calendar_id}', to_jsonb($2::text))
                WHERE contact_id = $1
                  AND event_type = ANY($3::varchar[])
                """,
                contact_id,
                relative_calendar_id,
                ["first_clean_booking_requested", "first_clean_booked"],
            )
        finally:
            await conn.execute(
                "ALTER TABLE eom_lead_lifecycle_events ENABLE TRIGGER USER"
            )
        calendar = _WonLossCalendar()

        with pytest.raises(
            EOMLeadConversionError, match="relative Calendar identifier"
        ) as exc:
            await mark_eom_lead_lost_with_won_teardown(
                provider,
                calendar,
                EOMLeadLost(
                    contact_id=str(contact_id),
                    reason_code="no_response",
                    note=None,
                    operation_key=f"office-won-loss-{uuid.uuid4().hex}",
                    actor_id=1,
                    actor_name="Juan Canfield",
                ),
            )

        assert exc.value.status_code == 409
        assert calendar.calls == []
        assert await conn.fetchval(
            "SELECT lead_stage FROM contacts WHERE id = $1", contact_id
        ) == "won"
        assert await conn.fetchval(
            "SELECT status FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            draft_id,
        ) == "pending"
        assert await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1
              AND event_type = 'first_clean_cancellation_requested'
            """,
            contact_id,
        ) == 0
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_won_lead_loss_execution_fences_claim_handoff_and_status_writers():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_won_loss_fence_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    pool = None
    loss_task = None
    claim_task = None
    handoff_task = None
    status_task = None
    archive_task = None
    calendar = _BlockingWonLossCalendar()
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        pool = await asyncpg.create_pool(
            database_url,
            min_size=3,
            max_size=6,
            server_settings={"search_path": f'"{schema}", public'},
        )
        provider = DatabaseCRMProvider(pool=pool)
        contact_id, draft_id = await _book_first_clean_draft(conn, provider)
        command = EOMLeadLost(
            contact_id=str(contact_id),
            reason_code="no_response",
            note=None,
            operation_key=f"office-won-loss-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )
        loss_task = asyncio.create_task(
            mark_eom_lead_lost_with_won_teardown(provider, calendar, command)
        )
        await asyncio.wait_for(calendar.started.wait(), timeout=5)

        claim_task = asyncio.create_task(
            provider.claim_eom_onboarding_draft(
                draft_id=draft_id,
                actor_id=2,
                actor_name="Mayra Canfield",
            )
        )
        handoff_task = asyncio.create_task(
            provider.finalize_eom_customer_handoff(
                contact_id=str(contact_id),
                tracker_customer_id=981,
                tracker_site_id=1981,
                approval_key=f"office-handoff-{uuid.uuid4().hex}",
                actor_id=2,
                actor_name="Mayra Canfield",
            )
        )
        status_task = asyncio.create_task(
            provider.update_contact(str(contact_id), {"status": "inactive"})
        )
        archive_task = asyncio.create_task(provider.delete_contact(str(contact_id)))
        await asyncio.sleep(0.1)
        assert not claim_task.done()
        assert not handoff_task.done()
        assert not status_task.done()
        assert not archive_task.done()

        calendar.release.set()
        completed = await asyncio.wait_for(loss_task, timeout=5)
        assert completed["lead_stage"] == "lost"
        with pytest.raises(EOMLeadConversionError, match="revoked") as claim_exc:
            await asyncio.wait_for(claim_task, timeout=5)
        assert claim_exc.value.status_code == 409
        with pytest.raises(EOMLeadConversionError) as handoff_exc:
            await asyncio.wait_for(handoff_task, timeout=5)
        assert handoff_exc.value.status_code == 409
        status_result, archived = await asyncio.gather(
            asyncio.wait_for(status_task, timeout=5),
            asyncio.wait_for(archive_task, timeout=5),
        )
        assert status_result is not None
        assert archived is True
    finally:
        calendar.release.set()
        for task in (loss_task, claim_task, handoff_task, status_task, archive_task):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if pool is not None:
            await pool.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_pre_won_lead_loss_retains_the_direct_replay_without_calendar():
    """Routing through the safety service cannot alter pre-won loss semantics."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_pre_won_loss_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await _insert_contact(conn, contact_id=contact_id, lead_stage="estimate_booked")
        calendar = _WonLossCalendar()
        command = EOMLeadLost(
            contact_id=str(contact_id),
            reason_code="declined_after_estimate",
            note="Price did not fit",
            operation_key=f"office-pre-won-loss-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
        )

        first = await mark_eom_lead_lost_with_won_teardown(provider, calendar, command)
        replay = await mark_eom_lead_lost_with_won_teardown(
            provider, calendar, command
        )

        assert first["from_stage"] == "estimate_booked"
        assert first["idempotent"] is False
        assert replay["from_stage"] == "estimate_booked"
        assert replay["idempotent"] is True
        assert calendar.calls == []
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


@pytest.mark.asyncio
async def test_operator_contact_records_the_identity_it_overwrote_on_a_phone_match():
    """An operator CREATE can silently rewrite an existing contact's identity.

    Reproduces a live 2026-08-08 case: an office customer create carrying no
    contact id matched a calendar_import contact on phone and overwrote its
    full_name. The prior name could not be recovered from anything afterwards,
    which is what this event has to prevent.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_overwrite_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        contact_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO contacts (
                id, full_name, phone, business_context_id, contact_type,
                status, source
            ) VALUES (
                $1, 'Cal Import Label', '217-555-0142', 'effingham_maids',
                'customer', 'active', 'calendar_import'
            )
            """,
            contact_id,
        )

        # No contact_id: a create that resolves to the existing row by phone.
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"operator-overwrite-{uuid.uuid4().hex}",
            actor_id=1,
            actor_name="Juan Canfield",
            source_channel="time_tracker",
            source_ref=str(uuid.uuid4()),
            fields={"full_name": "Canonical Person", "phone": "217-555-0142"},
        )
        result = await mutate_eom_operator_contact(provider, command)

        assert result["operation"] == "contact_updated"
        assert result["contact_id"] == str(contact_id)

        event = await conn.fetchrow(
            """
            SELECT metadata
            FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'contact_updated'
            """,
            contact_id,
        )
        metadata = _metadata_dict(event["metadata"])
        assert "full_name" in metadata["changed_fields"]
        assert metadata["previous_values"]["full_name"] == "Cal Import Label"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_operator_create_persists_customer_type_rather_than_defaulting():
    """A create that states a type must store it, not silently fall back.

    The provider's contact INSERT names its columns explicitly, so a column the
    statement omits is written from its DEFAULT and the caller's value is lost
    with no error. 'commercial' landing as 'unknown' is exactly the silent
    downgrade this slice exists to make impossible.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_ctype_create_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"ctype-create-{uuid.uuid4().hex}",
            actor_id=7,
            actor_name="Mayra Canfield",
            source_channel="time_tracker",
            source_ref="customer:401",
            contact_type="customer",
            fields={"full_name": "Commercial Create", "customer_type": "commercial"},
        )

        created = await mutate_eom_operator_contact(provider, command)

        contact = await conn.fetchrow(
            "SELECT * FROM contacts WHERE id = $1", uuid.UUID(created["contact_id"])
        )
        assert contact["customer_type"] == "commercial"
        assert contact["customer_type_revision"] > 0
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_operator_create_without_a_type_is_unknown_not_guessed():
    """Silence means unknown. The boundary must not infer a type."""
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_ctype_absent_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        command = EOMOperatorContactMutation.from_raw(
            operation_key=f"ctype-absent-{uuid.uuid4().hex}",
            actor_id=7,
            actor_name="Mayra Canfield",
            source_channel="time_tracker",
            source_ref="customer:402",
            contact_type="customer",
            fields={"full_name": "AKRA Builders LLC"},
        )

        created = await mutate_eom_operator_contact(provider, command)

        contact = await conn.fetchrow(
            "SELECT * FROM contacts WHERE id = $1", uuid.UUID(created["contact_id"])
        )
        assert contact["customer_type"] == "unknown", (
            "a company-shaped name must not be guessed into 'commercial'"
        )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_operator_update_changes_customer_type_and_audits_the_old_value():
    """Both directions of the change, and the prior value recorded.

    There is no contact history table, so if the lifecycle event does not carry
    previous_values the overwritten type exists nowhere once the UPDATE commits.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_ctype_update_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        created = await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=f"ctype-seed-{uuid.uuid4().hex}",
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="customer:403",
                contact_type="customer",
                fields={"full_name": "Type Flip", "customer_type": "residential"},
            ),
        )
        contact_id = created["contact_id"]
        before = await conn.fetchval(
            "SELECT customer_type_revision FROM contacts WHERE id = $1",
            uuid.UUID(contact_id),
        )

        updated = await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=f"ctype-flip-{uuid.uuid4().hex}",
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="customer:403",
                contact_id=contact_id,
                contact_type="customer",
                fields={"customer_type": "commercial"},
            ),
        )

        assert updated["operation"] == "contact_updated"
        contact = await conn.fetchrow(
            "SELECT * FROM contacts WHERE id = $1", uuid.UUID(contact_id)
        )
        assert contact["customer_type"] == "commercial"
        assert contact["customer_type_revision"] > before

        event = await conn.fetchrow(
            """
            SELECT metadata FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'contact_updated'
            ORDER BY created_at DESC LIMIT 1
            """,
            uuid.UUID(contact_id),
        )
        metadata = _metadata_dict(event["metadata"])
        assert "customer_type" in metadata["changed_fields"]
        assert metadata["previous_values"]["customer_type"] == "residential"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_the_database_refuses_a_customer_type_outside_the_set():
    """The constraint is the enforcement; the boundary is only the front door.

    Application validation can be bypassed by a future writer. This asserts
    Postgres itself rejects the value, which is why the CHECK exists.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_ctype_check_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        row = await conn.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Check Probe', 'effingham_maids') RETURNING id
            """
        )
        for accepted in ("residential", "commercial", "unknown"):
            await conn.execute(
                "UPDATE contacts SET customer_type = $2 WHERE id = $1",
                row["id"],
                accepted,
            )

        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "UPDATE contacts SET customer_type = 'bogus' WHERE id = $1", row["id"]
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def test_the_boundary_refuses_a_bad_customer_type_before_the_database_sees_it():
    """422 from the contract, not a 500 from the CHECK.

    The constraint is the durable enforcement, but a value that only the
    database rejects reaches the caller as a server error instead of a
    validation error. Both layers must agree on the same set.
    """

    def _command(value):
        return EOMOperatorContactMutation.from_raw(
            operation_key=f"ctype-guard-{uuid.uuid4().hex}",
            actor_id=7,
            actor_name="Mayra Canfield",
            source_channel="time_tracker",
            source_ref="customer:404",
            contact_type="customer",
            fields={"full_name": "Guard Probe", "customer_type": value},
        )

    # The tracker stores these capitalised; refusing them would make the
    # boundary reject the exact values the backfill reads.
    assert _command("Residential").fields["customer_type"] == "residential"
    assert _command("  COMMERCIAL  ").fields["customer_type"] == "commercial"
    assert _command("unknown").fields["customer_type"] == "unknown"

    for rejected in ("bogus", "", "   ", None, 3):
        with pytest.raises(EOMOperatorContactMutationError) as caught:
            _command(rejected)
        assert caught.value.status_code == 422


@pytest.mark.asyncio
async def test_funnel_readiness_refuses_a_contacts_table_without_customer_type_or_revision():
    """Migrations 366 and 367 are readiness preconditions, not nice-to-haves.

    Startup catches a failed migration and continues into the readiness guard.
    The provider's contact INSERT names customer_type explicitly and the
    known-contacts read names customer_type_revision. Admitting the funnel
    against a table missing either field would move failure from the gate --
    where it is one controlled error -- to a later write or tracker refresh.
    """
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    database_url = _database_url_or_skip()
    schema = f"atlas_eom_ctype_readiness_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        # Privilege migration applied (the default), because that is the
        # arrangement the readiness query accepts -- see
        # test_privilege_migration_satisfies_the_enabled_full_app_startup_guard.
        await _prepare_schema(conn, schema)

        class _Pool:
            is_initialized = True

            async def fetchval(self, query: str) -> bool:
                return bool(await conn.fetchval(query))

        # Ready while the column is present...
        await require_eom_funnel_data_store(
            type("Config", (), {"api_enabled": True})(),
            database_enabled=True,
            get_db_pool_fn=lambda: _Pool(),
        )

        # ...and refused once either field is gone, which is the state a failed
        # migration leaves.
        await conn.execute("ALTER TABLE contacts DROP COLUMN customer_type")
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                type("Config", (), {"api_enabled": True})(),
                database_enabled=True,
                get_db_pool_fn=lambda: _Pool(),
            )

        await conn.execute("ALTER TABLE contacts ADD COLUMN customer_type VARCHAR(16)")
        await conn.execute("ALTER TABLE contacts DROP COLUMN customer_type_revision")
        with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
            await require_eom_funnel_data_store(
                type("Config", (), {"api_enabled": True})(),
                database_enabled=True,
                get_db_pool_fn=lambda: _Pool(),
            )

        # A present but incompatible value would satisfy a name-only guard, yet
        # cannot support the BIGINT source-version contract. The type check is
        # therefore a real fail-closed branch, not an incidental refinement.
        await conn.execute(
            "ALTER TABLE contacts ADD COLUMN customer_type_revision INTEGER"
        )
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
async def test_a_generic_contact_insert_still_works_without_migration_366():
    """The shared insert helper must not regress callers that never asked.

    `_insert_contact_row` is used by every contact writer -- the MCP
    create_contact tool and inbound find_or_create_contact among them -- and
    startup deliberately continues after a failed migration. Naming
    customer_type unconditionally would turn a pending 366 into
    UndefinedColumnError for those callers, a regression they did not sign up
    for. Omitting the column when the caller did not supply it keeps them on
    exactly the behaviour they had before this PR.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_ctype_pre366_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(
            conn,
            schema,
            apply_privilege_migration=False,
            apply_customer_type_revision_migration=False,
        )
        # The state a pending or failed migration 366 leaves behind.
        await conn.execute("ALTER TABLE contacts DROP COLUMN customer_type")
        provider = DatabaseCRMProvider(pool=conn)

        row = await provider._insert_contact_row(
            conn,
            {
                "full_name": "Pre-366 Caller",
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
            },
        )

        assert row["full_name"] == "Pre-366 Caller"

        # A caller that DOES specify the type still fails loudly, because its
        # intent cannot be honoured against a table without the column.
        with pytest.raises(asyncpg.exceptions.UndefinedColumnError):
            await provider._insert_contact_row(
                conn,
                {
                    "full_name": "Wants A Type",
                    "business_context_id": "effingham_maids",
                    "contact_type": "customer",
                },
                customer_type="commercial",
            )
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_a_generic_create_cannot_set_customer_type():
    """Only the operator boundary may classify an account.

    customer_type drives billing shape. A generic writer dropping the key into
    the insert dict would bypass _normalize_customer_type, the authenticated
    funnel boundary, and the lifecycle event recording who changed it -- so the
    field is a keyword the operator path passes, and its presence in `data` is
    refused outright rather than ignored. Silently dropping it would be the
    same class of silent loss this slice exists to remove.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_ctype_generic_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)

        # The private insert refuses...
        with pytest.raises(ValueError, match="operator mutation"):
            await provider._insert_contact_row(
                conn,
                {
                    "full_name": "Sneaky Generic Writer",
                    "business_context_id": "effingham_maids",
                    "contact_type": "customer",
                    "customer_type": "commercial",
                },
            )

        # ...and so does the PUBLIC door, which is the one that matters:
        # create_contact returns through a dedup/merge branch that never
        # reaches the insert, so a contact matched by phone or email would
        # otherwise report success while the classification was dropped.
        await provider._insert_contact_row(
            conn,
            {
                "full_name": "Already Here",
                "phone": "2175550190",
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
            },
        )
        with pytest.raises(ValueError, match="operator mutation"):
            await provider.create_contact(
                {
                    "full_name": "Already Here",
                    "phone": "2175550190",
                    "business_context_id": "effingham_maids",
                    "contact_type": "customer",
                    "customer_type": "commercial",
                }
            )
        with pytest.raises(ValueError, match="operator mutation"):
            await provider.find_or_create_contact(
                "Already Here",
                phone="2175550190",
                customer_type="commercial",
            )
        assert await conn.fetchval(
            "SELECT customer_type FROM contacts WHERE full_name = $1", "Already Here"
        ) == "unknown", "the matched contact must not be classified"

        # Nothing was written.
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM contacts WHERE full_name = $1",
            "Sneaky Generic Writer",
        ) == 0

        # The operator path still classifies, through its keyword.
        row = await provider._insert_contact_row(
            conn,
            {
                "full_name": "Operator Classified",
                "business_context_id": "effingham_maids",
                "contact_type": "customer",
            },
            customer_type="commercial",
        )
        assert row["customer_type"] == "commercial"
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_operator_contact_mutation_clear_vs_omit_vs_change_tri_state():
    """The locked field-clearing contract (website #254), at the write authority.

    Present-null clears to SQL NULL; an absent key preserves; a value replaces.
    The lifecycle event alone distinguishes all three: OMITTED is absent from
    changed_fields, CLEARED is listed in cleared_fields, CHANGED is
    changed_fields minus cleared_fields.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_clear_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        created = await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=f"clear-seed-{uuid.uuid4().hex}",
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:seed",
                contact_type="customer",
                fields={
                    "full_name": "Clear Target",
                    "email": "clear-target@example.com",
                    "phone": "(217) 555-0181",
                },
            ),
        )
        contact_id = created["contact_id"]

        # CLEAR email (present null) while OMITTING phone (key absent).
        clear_key = f"clear-email-{uuid.uuid4().hex}"
        cleared = await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=clear_key,
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:clear",
                contact_type="customer",
                contact_id=contact_id,
                fields={"email": None},
            ),
        )
        assert cleared["operation"] == "contact_updated"
        # The mutation result itself must prove the clear persisted.
        assert cleared["contact"]["email"] is None
        row = await conn.fetchrow(
            "SELECT email, phone FROM contacts WHERE id = $1",
            uuid.UUID(contact_id),
        )
        assert row["email"] is None
        assert row["phone"] == "2175550181", "omitted phone must be preserved"
        event = await conn.fetchrow(
            """
            SELECT metadata FROM eom_lead_lifecycle_events
            WHERE operation_key = $1 AND event_type = 'contact_updated'
            """,
            clear_key,
        )
        metadata = _metadata_dict(event["metadata"])
        assert metadata["changed_fields"] == ["email"]
        assert metadata["cleared_fields"] == ["email"]
        assert metadata["previous_values"] == {"email": "clear-target@example.com"}
        # Names only in the tri-state keys: the cleared value appears in
        # previous_values (sole surviving copy) and nowhere else.
        assert "new_values" not in metadata

        # CHANGED control: a re-point lists the field but not as cleared.
        change_key = f"change-phone-{uuid.uuid4().hex}"
        await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=change_key,
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:change",
                contact_type="customer",
                contact_id=contact_id,
                fields={"phone": "(217) 555-0182"},
            ),
        )
        change_event = await conn.fetchrow(
            """
            SELECT metadata FROM eom_lead_lifecycle_events
            WHERE operation_key = $1 AND event_type = 'contact_updated'
            """,
            change_key,
        )
        change_metadata = _metadata_dict(change_event["metadata"])
        assert change_metadata["changed_fields"] == ["phone"]
        assert change_metadata["cleared_fields"] == []
        assert change_metadata["previous_values"] == {"phone": "2175550181"}

        # Sibling-path proof (both advertised fields, not just email): phone
        # rides its own normalizer and column, so its clear is proven through
        # the same DB + audit shape rather than inferred from email's.
        phone_clear_key = f"clear-phone-{uuid.uuid4().hex}"
        phone_cleared = await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=phone_clear_key,
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:clear-phone",
                contact_type="customer",
                contact_id=contact_id,
                fields={"phone": None},
            ),
        )
        assert phone_cleared["operation"] == "contact_updated"
        assert phone_cleared["contact"]["phone"] is None
        phone_row = await conn.fetchrow(
            "SELECT email, phone FROM contacts WHERE id = $1",
            uuid.UUID(contact_id),
        )
        assert phone_row["phone"] is None
        assert phone_row["email"] is None, "the earlier email clear must persist"
        phone_event = await conn.fetchrow(
            """
            SELECT metadata FROM eom_lead_lifecycle_events
            WHERE operation_key = $1 AND event_type = 'contact_updated'
            """,
            phone_clear_key,
        )
        phone_metadata = _metadata_dict(phone_event["metadata"])
        assert phone_metadata["changed_fields"] == ["phone"]
        assert phone_metadata["cleared_fields"] == ["phone"]
        assert phone_metadata["previous_values"] == {"phone": "2175550182"}
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_operator_contact_clear_replay_is_idempotent_and_omit_conflicts():
    """A retried clear replays; the same key with omit-instead-of-clear 409s.

    The second half is the fingerprint proof: fields={'email': None} and
    fields without the email key must hash differently, or a retry that
    dropped the null would silently replay the wrong intent.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_clear_replay_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        created = await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=f"clear-replay-seed-{uuid.uuid4().hex}",
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:seed",
                contact_type="customer",
                fields={
                    "full_name": "Replay Target",
                    "email": "replay-target@example.com",
                    "phone": "(217) 555-0183",
                },
            ),
        )
        contact_id = created["contact_id"]
        clear_key = f"clear-replay-{uuid.uuid4().hex}"

        def clear_command():
            return EOMOperatorContactMutation.from_raw(
                operation_key=clear_key,
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:clear",
                contact_type="customer",
                contact_id=contact_id,
                fields={"email": None, "phone": "(217) 555-0183"},
            )

        first = await mutate_eom_operator_contact(provider, clear_command())
        assert first["idempotent"] is False
        assert first["contact"]["email"] is None

        replay = await mutate_eom_operator_contact(provider, clear_command())
        assert replay["idempotent"] is True
        assert replay["contact"]["email"] is None
        assert await conn.fetchval(
            """
            SELECT COUNT(*) FROM eom_lead_lifecycle_events
            WHERE operation_key = $1
            """,
            clear_key,
        ) == 1

        omit_instead = EOMOperatorContactMutation.from_raw(
            operation_key=clear_key,
            actor_id=7,
            actor_name="Mayra Canfield",
            source_channel="time_tracker",
            source_ref="portal-contact:clear",
            contact_type="customer",
            contact_id=contact_id,
            fields={"phone": "(217) 555-0183"},
        )
        with pytest.raises(EOMOperatorContactMutationError) as exc_info:
            await mutate_eom_operator_contact(provider, omit_instead)
        assert exc_info.value.status_code == 409
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_operator_contact_clear_is_scoped_to_nullable_optional_fields():
    """full_name and customer_type refuse a clear; a repeat clear no-ops.

    The refusals are the closed edge of the clearable set (website #254 locks
    clearing to optional contact fields; identity/type never null out), and
    the repeat clear proves clearing an already-null field is a safe no-op
    update, not an error and not a duplicate event shape.
    """
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_operator_clear_scope_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        provider = DatabaseCRMProvider(pool=conn)
        created = await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=f"clear-scope-seed-{uuid.uuid4().hex}",
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:seed",
                contact_type="customer",
                fields={
                    "full_name": "Scope Target",
                    "phone": "(217) 555-0184",
                },
            ),
        )
        contact_id = created["contact_id"]

        with pytest.raises(EOMOperatorContactMutationError) as name_exc:
            EOMOperatorContactMutation.from_raw(
                operation_key=f"clear-name-{uuid.uuid4().hex}",
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:clear",
                contact_type="customer",
                contact_id=contact_id,
                fields={"full_name": None},
            )
        assert name_exc.value.status_code == 422

        with pytest.raises(EOMOperatorContactMutationError) as type_exc:
            EOMOperatorContactMutation.from_raw(
                operation_key=f"clear-type-{uuid.uuid4().hex}",
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:clear",
                contact_type="customer",
                contact_id=contact_id,
                fields={"customer_type": None},
            )
        assert type_exc.value.status_code == 422

        # email is already NULL (never set): clearing it again is a no-op
        # update with a uniform, empty tri-state event shape.
        noop_key = f"clear-noop-{uuid.uuid4().hex}"
        noop = await mutate_eom_operator_contact(
            provider,
            EOMOperatorContactMutation.from_raw(
                operation_key=noop_key,
                actor_id=7,
                actor_name="Mayra Canfield",
                source_channel="time_tracker",
                source_ref="portal-contact:clear",
                contact_type="customer",
                contact_id=contact_id,
                fields={"email": None},
            ),
        )
        assert noop["operation"] == "contact_updated"
        assert noop["contact"]["email"] is None
        noop_event = await conn.fetchrow(
            """
            SELECT metadata FROM eom_lead_lifecycle_events
            WHERE operation_key = $1 AND event_type = 'contact_updated'
            """,
            noop_key,
        )
        noop_metadata = _metadata_dict(noop_event["metadata"])
        assert noop_metadata["changed_fields"] == []
        assert noop_metadata["cleared_fields"] == []
        assert noop_metadata["previous_values"] == {}
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
