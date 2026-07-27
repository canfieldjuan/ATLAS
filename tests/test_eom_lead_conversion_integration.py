"""Real-Postgres proof for the EOM customer handoff transaction."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402
from atlas_brain.services.eom_lead_conversion import EOMLeadConversionError  # noqa: E402


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
    await conn.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'atlas_eom_handoff_owner'
            ) THEN
                CREATE ROLE atlas_eom_handoff_owner NOLOGIN NOINHERIT;
            END IF;
        END;
        $$;
        """
    )


async def _prepare_schema(
    conn,
    schema: str,
    *,
    apply_privilege_migration: bool = True,
) -> None:
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}", public')
    await conn.execute(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    await conn.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
    await conn.execute("CREATE TABLE api_keys (id UUID PRIMARY KEY)")
    await conn.execute("CREATE TABLE byok_keys (id UUID PRIMARY KEY)")
    await conn.execute("CREATE TABLE scoped_mailbox_credentials (id UUID PRIMARY KEY)")
    migration_names = (
        "035_contacts.sql",
        "256_contact_interaction_dedupe.sql",
        "346_contact_lead_pipeline.sql",
        "351_eom_lead_lifecycle_events.sql",
        "352_eom_inbound_delivery_receipts.sql",
        "353_eom_customer_handoffs.sql",
    )
    if apply_privilege_migration:
        await _provision_nocodb_login(conn)
        migration_names += ("354_eom_customer_handoff_privileges.sql",)
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


async def _contact_state(conn, contact_id: uuid.UUID) -> tuple[dict[str, object], int, int]:
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
            "events": await conn.fetchval("SELECT COUNT(*) FROM eom_lead_lifecycle_events"),
            "handoffs": await conn.fetchval("SELECT COUNT(*) FROM eom_customer_handoffs"),
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
                "email": "newer@example.com",
                "phone": "2175550101",
                "address": "101 Main St",
                "source": "web",
                "created_at": newer_created_at,
            },
            {
                "contact_id": eligible_id,
                "full_name": "Eligible Earlier",
                "email": "earlier@example.com",
                "phone": "2175550100",
                "address": "100 Main St",
                "source": "web",
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
            "created_at",
        }
        assert {
            "contacts": await conn.fetchval("SELECT COUNT(*) FROM contacts"),
            "events": await conn.fetchval("SELECT COUNT(*) FROM eom_lead_lifecycle_events"),
            "handoffs": await conn.fetchval("SELECT COUNT(*) FROM eom_customer_handoffs"),
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
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM contact_interactions WHERE contact_id = $1",
            contact["id"],
        ) == 1

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
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1", contact_id
        ) == 1
        assert await conn.fetchval(
            """
            SELECT COUNT(*) FROM eom_lead_lifecycle_events
            WHERE contact_id = $1 AND event_type = 'customer_approved'
            """,
            contact_id,
        ) == 1

        with pytest.raises(asyncpg.exceptions.RaiseError, match="immutable"):
            await conn.execute(
                """
                UPDATE eom_customer_handoffs
                SET tracker_site_id = 999
                WHERE contact_id = $1
                """,
                contact_id,
            )
        assert await conn.fetchval(
            "SELECT tracker_site_id FROM eom_customer_handoffs WHERE contact_id = $1",
            contact_id,
        ) == 202

        with pytest.raises(asyncpg.exceptions.RaiseError, match="immutable"):
            await conn.execute(
                "DELETE FROM eom_customer_handoffs WHERE contact_id = $1",
                contact_id,
            )
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM eom_customer_handoffs WHERE contact_id = $1",
            contact_id,
        ) == 1

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

        with pytest.raises(EOMLeadConversionError, match="not a completed finalization"):
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
        assert not await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS nocodb_role ON nocodb_role.oid = membership.member
                WHERE nocodb_role.rolname = 'atlas_nocodb'
            )
            """
        )

        nocodb_conn = await asyncpg.connect(
            database_url,
            user="atlas_nocodb",
            password=_NOCODB_TEST_PASSWORD,
        )
        try:
            await nocodb_conn.execute(f'SET search_path TO {_quote_ident(schema)}, public')
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
            assert await nocodb_conn.fetchval(
                "SELECT notes FROM contacts WHERE id = $1", contact_id
            ) == "ordinary edit"
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
        await conn.execute(
            """
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
            """
        )

        class _SchemaPool:
            async def acquire(self):
                return conn

            async def release(self, released):
                assert released is conn

        with pytest.raises(asyncpg.exceptions.RaiseError, match="injected privilege ledger failure"):
            await run_migrations(
                _SchemaPool(),
                migrations_dir=MIGRATIONS,
                only={"354_eom_customer_handoff_privileges"},
            )

        assert await conn.fetchval(
            """
            SELECT pg_get_userbyid(relowner) = current_user
            FROM pg_class
            WHERE oid = 'eom_customer_handoffs'::regclass
            """
        )
        assert await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = current_user
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """
        )
        assert not await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM schema_migrations
                WHERE name = '354_eom_customer_handoff_privileges'
            )
            """
        )

        await conn.execute("DROP TRIGGER trg_fail_eom_privilege_bookkeeping ON schema_migrations")
        await conn.execute("DROP FUNCTION fail_eom_privilege_bookkeeping()")
        await run_migrations(
            _SchemaPool(),
            migrations_dir=MIGRATIONS,
            only={"354_eom_customer_handoff_privileges"},
        )

        assert await conn.fetchval(
            """
            SELECT pg_get_userbyid(relowner) = 'atlas_eom_handoff_owner'
            FROM pg_class
            WHERE oid = 'eom_customer_handoffs'::regclass
            """
        )
        assert await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = current_user
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """
        )
        await conn.execute(
            f"REVOKE atlas_eom_handoff_owner FROM {runtime_ident} CASCADE"
        )
        assert not await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = current_user
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """
        )
        assert await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM schema_migrations
                WHERE name = '354_eom_customer_handoff_privileges'
            )
            """
        )
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
        assert await executor_conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_auth_members AS membership
                JOIN pg_roles AS member_role ON member_role.oid = membership.member
                JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
                WHERE member_role.rolname = current_user
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
            )
            """
        )
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
async def test_privilege_migration_satisfies_the_enabled_full_app_startup_guard(monkeypatch):
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
        await conn.execute(
            """
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
            """
        )
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
            result for result in cross_results if isinstance(result, EOMLeadConversionError)
        ]
        assert len(successes) == 1
        assert len(rejections) == 1
        assert rejections[0].status_code == 409
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM eom_customer_handoffs WHERE approval_key = $1",
            cross_contact_key,
        ) == 1

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
        assert await conn.fetchval(
            """
            SELECT COUNT(*) FROM eom_customer_handoffs
            WHERE tracker_customer_id = 505 AND tracker_site_id = 606
            """
        ) == 1
        assert await conn.fetchval(
            """
            SELECT COUNT(*) FROM contacts
            WHERE id = ANY($1::uuid[]) AND contact_type = 'customer'
            """,
            [second_contact_id, third_contact_id],
        ) == 1
    finally:
        if pool is not None:
            await pool.close()
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
