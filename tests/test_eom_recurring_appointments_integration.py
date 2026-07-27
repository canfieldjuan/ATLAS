"""Real-Postgres MCP proof for EOM appointment operating fields."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.mcp import crm_server as crm_mcp  # noqa: E402
from atlas_brain.services.crm_provider import DatabaseCRMProvider  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
TENANT = "effingham_maids"
FOREIGN_TENANT = "churnsignals"


async def _insert_appointment(
    pool,
    *,
    contact_id,
    tenant: str,
    customer_name: str,
    price: Decimal | None = None,
):
    now = datetime.now(timezone.utc)
    return await pool.fetchrow(
        """
        INSERT INTO appointments (
            start_time,
            end_time,
            duration_minutes,
            service_type,
            customer_name,
            customer_phone,
            business_context_id,
            contact_id,
            per_visit_price
        )
        VALUES ($1, $2, 120, 'Cleaning', $3, '217-555-0100', $4, $5, $6)
        RETURNING id
        """,
        now + timedelta(days=1),
        now + timedelta(days=1, hours=2),
        customer_name,
        tenant,
        contact_id,
        price,
    )


@pytest.mark.asyncio
async def test_mcp_appointment_operating_fields_are_tenant_safe(monkeypatch):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_appointment_ops_{uuid.uuid4().hex}"
    admin_conn = await asyncpg.connect(database_url)
    pool = None
    crm_mcp.set_provider_override(lambda: DatabaseCRMProvider())
    try:
        await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
        await admin_conn.execute(f'SET search_path TO "{schema}", public')
        for name in (
            "012_appointments.sql",
            "035_contacts.sql",
            "045_invoices.sql",
            "048_customer_services.sql",
            "348_appointment_operating_fields.sql",
        ):
            await admin_conn.execute((MIGRATIONS / name).read_text())

        async def set_search_path(connection):
            await connection.execute(f'SET search_path TO "{schema}", public')

        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=3,
            setup=set_search_path,
        )

        import atlas_brain.storage.database as db_mod
        from atlas_brain.config import settings

        monkeypatch.setattr(db_mod, "get_db_pool", lambda: pool)
        monkeypatch.setattr(
            settings.mcp,
            "crm_default_business_context",
            TENANT,
        )

        primary_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Mid Illinois Concrete', NULL)
            RETURNING id
            """
        )
        separate_billing_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Mid Illinois Concrete', $1)
            RETURNING id
            """,
            TENANT,
        )
        foreign_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Foreign Appointment', $1)
            RETURNING id
            """,
            FOREIGN_TENANT,
        )
        archived_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id, status)
            VALUES ('Archived Appointment', $1, 'archived')
            RETURNING id
            """,
            TENANT,
        )

        primary_appointment = await _insert_appointment(
            pool,
            contact_id=primary_contact["id"],
            tenant=TENANT,
            customer_name="Mid Illinois Concrete",
        )
        separate_billing_appointment = await _insert_appointment(
            pool,
            contact_id=separate_billing_contact["id"],
            tenant=TENANT,
            customer_name="Mid Illinois Concrete",
            price=Decimal("80.00"),
        )
        foreign_appointment = await _insert_appointment(
            pool,
            contact_id=foreign_contact["id"],
            tenant=FOREIGN_TENANT,
            customer_name="Foreign Appointment",
        )
        archived_appointment = await _insert_appointment(
            pool,
            contact_id=archived_contact["id"],
            tenant=TENANT,
            customer_name="Archived Appointment",
        )
        await pool.executemany(
            """
            INSERT INTO customer_services (
                contact_id,
                service_name,
                rate,
                calendar_keyword,
                business_context_id
            )
            VALUES ($1, 'Facility cleaning', $2, $3, $4)
            """,
            [
                (
                    primary_contact["id"],
                    Decimal("125.00"),
                    "MID-MAIN",
                    TENANT,
                ),
                (
                    separate_billing_contact["id"],
                    Decimal("80.00"),
                    "MID-BILLING",
                    TENANT,
                ),
            ],
        )
        await pool.executemany(
            """
            INSERT INTO invoices (
                invoice_number,
                contact_id,
                customer_name,
                due_date,
                total_amount,
                business_context_id
            )
            VALUES ($1, $2, 'Mid Illinois Concrete', CURRENT_DATE, $3, $4)
            """,
            [
                (
                    "INV-EOM-OPS-1",
                    primary_contact["id"],
                    Decimal("125.00"),
                    TENANT,
                ),
                (
                    "INV-EOM-OPS-2",
                    separate_billing_contact["id"],
                    Decimal("80.00"),
                    TENANT,
                ),
            ],
        )

        validation_cases = [
            (
                {},
                "No appointment operating fields provided",
            ),
            (
                {"recurrence_interval": 2},
                "recurrence_interval and recurrence_unit must be provided together",
            ),
            (
                {"recurrence_unit": "week"},
                "recurrence_interval and recurrence_unit must be provided together",
            ),
            (
                {"recurrence_interval": 0, "recurrence_unit": "week"},
                "recurrence_interval must be between 1 and 365",
            ),
            (
                {"recurrence_interval": 2, "recurrence_unit": "year"},
                "recurrence_unit must be day, week, or month",
            ),
            (
                {"assigned_cleaner": " "},
                "assigned_cleaner is required",
            ),
            (
                {"assigned_cleaner": "x" * 129},
                "assigned_cleaner must be at most 128 characters",
            ),
            (
                {"per_visit_price": "not-money"},
                "per_visit_price must be numeric",
            ),
            (
                {"per_visit_price": "NaN"},
                "per_visit_price must be finite",
            ),
            (
                {"per_visit_price": "-0.01"},
                "per_visit_price must be non-negative",
            ),
            (
                {"per_visit_price": "10.005"},
                "per_visit_price must have at most 2 decimal places",
            ),
            (
                {"per_visit_price": "1e-100000"},
                "per_visit_price must have at most 2 decimal places",
            ),
            (
                {"per_visit_price": "10000000000.00"},
                "per_visit_price exceeds the database limit",
            ),
        ]
        for kwargs, error in validation_cases:
            result = json.loads(
                await crm_mcp.update_contact_appointment_operations(
                    str(primary_contact["id"]),
                    str(primary_appointment["id"]),
                    **kwargs,
                )
            )
            assert result == {"success": False, "error": error}

        updated = json.loads(
            await crm_mcp.update_contact_appointment_operations(
                str(primary_contact["id"]),
                str(primary_appointment["id"]),
                recurrence_interval=2,
                recurrence_unit=" WEEK ",
                assigned_cleaner=" Kennedi ",
                per_visit_price="125.50",
            )
        )
        assert updated["success"] is True
        assert updated["appointment"]["recurrence_interval"] == 2
        assert updated["appointment"]["recurrence_unit"] == "week"
        assert updated["appointment"]["assigned_cleaner"] == "Kennedi"
        assert updated["appointment"]["per_visit_price"] == "125.50"
        assert await pool.fetchval(
            "SELECT business_context_id FROM contacts WHERE id = $1",
            primary_contact["id"],
        ) == TENANT

        linked = json.loads(
            await crm_mcp.get_contact_appointments(
                str(primary_contact["id"])
            )
        )
        assert linked["count"] == 1
        assert linked["appointments"][0]["id"] == str(
            primary_appointment["id"]
        )
        assert linked["appointments"][0]["per_visit_price"] == "125.50"

        untouched = await pool.fetchrow(
            """
            SELECT contact_id, recurrence_interval, assigned_cleaner,
                   per_visit_price
            FROM appointments
            WHERE id = $1
            """,
            separate_billing_appointment["id"],
        )
        assert untouched["contact_id"] == separate_billing_contact["id"]
        assert untouched["recurrence_interval"] is None
        assert untouched["assigned_cleaner"] is None
        assert untouched["per_visit_price"] == Decimal("80.00")
        assert await pool.fetchval(
            """
            SELECT COUNT(*)
            FROM contacts
            WHERE full_name = 'Mid Illinois Concrete'
            """
        ) == 2
        service_rows = await pool.fetch(
            """
            SELECT contact_id, rate
            FROM customer_services
            ORDER BY rate DESC
            """
        )
        assert [
            (row["contact_id"], row["rate"]) for row in service_rows
        ] == [
            (primary_contact["id"], Decimal("125.00")),
            (separate_billing_contact["id"], Decimal("80.00")),
        ]
        invoice_rows = await pool.fetch(
            """
            SELECT contact_id, total_amount
            FROM invoices
            ORDER BY total_amount DESC
            """
        )
        assert [
            (row["contact_id"], row["total_amount"])
            for row in invoice_rows
        ] == [
            (primary_contact["id"], Decimal("125.00")),
            (separate_billing_contact["id"], Decimal("80.00")),
        ]

        for contact_id, appointment_id in (
            (foreign_contact["id"], foreign_appointment["id"]),
            (archived_contact["id"], archived_appointment["id"]),
            (
                separate_billing_contact["id"],
                primary_appointment["id"],
            ),
        ):
            refused = json.loads(
                await crm_mcp.update_contact_appointment_operations(
                    str(contact_id),
                    str(appointment_id),
                    assigned_cleaner="Must Not Persist",
                )
            )
            assert refused == {
                "success": False,
                "error": "Appointment not found",
            }

        archived_history = json.loads(
            await crm_mcp.get_contact_appointments(
                str(archived_contact["id"])
            )
        )
        assert archived_history["count"] == 1
        assert archived_history["appointments"][0]["id"] == str(
            archived_appointment["id"]
        )

        missing_target_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Missing Target', NULL)
            RETURNING id
            """
        )
        missing_target = json.loads(
            await crm_mcp.update_contact_appointment_operations(
                str(missing_target_contact["id"]),
                str(uuid.uuid4()),
                assigned_cleaner="Must Not Claim",
            )
        )
        assert missing_target == {
            "success": False,
            "error": "Appointment not found",
        }
        assert await pool.fetchval(
            "SELECT business_context_id FROM contacts WHERE id = $1",
            missing_target_contact["id"],
        ) is None

        race_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Claim Race', NULL)
            RETURNING id
            """
        )
        race_appointment = await _insert_appointment(
            pool,
            contact_id=race_contact["id"],
            tenant=TENANT,
            customer_name="Claim Race",
        )
        async with pool.acquire() as locking_connection:
            transaction = locking_connection.transaction()
            await transaction.start()
            await locking_connection.execute(
                """
                UPDATE contacts
                SET updated_at = updated_at
                WHERE id = $1
                """,
                race_contact["id"],
            )
            update_task = asyncio.create_task(
                crm_mcp.update_contact_appointment_operations(
                    str(race_contact["id"]),
                    str(race_appointment["id"]),
                    assigned_cleaner="Losing Writer",
                )
            )
            await asyncio.sleep(0.05)
            assert not update_task.done()
            await locking_connection.execute(
                """
                UPDATE contacts
                SET business_context_id = $2
                WHERE id = $1
                """,
                race_contact["id"],
                FOREIGN_TENANT,
            )
            await transaction.commit()
        lost_claim = json.loads(await update_task)
        assert lost_claim == {
            "success": False,
            "error": "Appointment not found",
        }
        assert await pool.fetchval(
            "SELECT assigned_cleaner FROM appointments WHERE id = $1",
            race_appointment["id"],
        ) is None

        class ClaimBeforeLinkedRead(DatabaseCRMProvider):
            async def get_contact_appointments(
                self,
                contact_id: str,
                business_context_id: str | None = None,
            ):
                await pool.execute(
                    """
                    UPDATE contacts
                    SET business_context_id = $2
                    WHERE id = $1
                    """,
                    primary_contact["id"],
                    FOREIGN_TENANT,
                )
                return await super().get_contact_appointments(
                    contact_id,
                    business_context_id,
                )

        claiming_provider = ClaimBeforeLinkedRead()
        crm_mcp.set_provider_override(lambda: claiming_provider)
        raced_read = json.loads(
            await crm_mcp.get_contact_appointments(
                str(primary_contact["id"])
            )
        )
        assert raced_read == {"appointments": [], "count": 0}
    finally:
        crm_mcp.set_provider_override(None)
        if pool is not None:
            await pool.close()
        await admin_conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin_conn.close()
