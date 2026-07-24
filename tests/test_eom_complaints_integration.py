"""Real-Postgres MCP reachability proof for EOM complaint tickets."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
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


@pytest.mark.asyncio
async def test_mcp_customer_service_ticket_lifecycle_is_tenant_safe(monkeypatch):
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    schema = f"atlas_eom_complaints_{uuid.uuid4().hex}"
    admin_conn = await asyncpg.connect(database_url)
    pool = None
    crm_mcp.set_provider_override(lambda: DatabaseCRMProvider())
    try:
        await admin_conn.execute(f'CREATE SCHEMA "{schema}"')
        await admin_conn.execute(f'SET search_path TO "{schema}", public')
        await admin_conn.execute(
            "CREATE TABLE appointments (id UUID PRIMARY KEY)"
        )
        for name in ("035_contacts.sql", "347_customer_service_tickets.sql"):
            await admin_conn.execute((MIGRATIONS / name).read_text())

        async def set_search_path(connection):
            await connection.execute(
                f'SET search_path TO "{schema}", public'
            )

        pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=2,
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

        legacy_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Legacy Complaint Contact', NULL)
            RETURNING id
            """
        )
        foreign_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id)
            VALUES ('Foreign Complaint Contact', $1)
            RETURNING id
            """,
            FOREIGN_TENANT,
        )
        archived_contact = await pool.fetchrow(
            """
            INSERT INTO contacts (full_name, business_context_id, status)
            VALUES ('Archived Complaint Contact', $1, 'archived')
            RETURNING id
            """,
            TENANT,
        )

        blank = json.loads(
            await crm_mcp.open_customer_service_ticket(
                str(legacy_contact["id"]),
                "   ",
            )
        )
        assert blank == {"success": False, "error": "summary is required"}
        too_long = json.loads(
            await crm_mcp.open_customer_service_ticket(
                str(legacy_contact["id"]),
                "x" * 501,
            )
        )
        assert too_long == {
            "success": False,
            "error": "summary must be at most 500 characters",
        }
        empty_explicit_tenant = json.loads(
            await crm_mcp.list_customer_service_tickets(
                business_context_id=" ",
            )
        )
        assert empty_explicit_tenant == {
            "error": "business_context_id is required",
            "tickets": [],
            "count": 0,
        }

        opened = json.loads(
            await crm_mcp.open_customer_service_ticket(
                str(legacy_contact["id"]),
                "Missed kitchen floor",
                details="Customer reported the floor was not mopped.",
                priority="urgent",
                assignee="Juan",
            )
        )
        assert opened["success"] is True
        ticket_id = opened["ticket"]["id"]
        assert opened["ticket"]["business_context_id"] == TENANT
        assert opened["ticket"]["status"] == "open"
        assert await pool.fetchval(
            "SELECT business_context_id FROM contacts WHERE id = $1",
            legacy_contact["id"],
        ) == TENANT
        claimed_foreign = json.loads(
            await crm_mcp.open_customer_service_ticket(
                str(legacy_contact["id"]),
                "Must not steal a claimed contact",
                business_context_id=FOREIGN_TENANT,
            )
        )
        assert claimed_foreign == {
            "success": False,
            "error": "Contact not found",
        }

        for contact_id in (foreign_contact["id"], archived_contact["id"]):
            refused = json.loads(
                await crm_mcp.open_customer_service_ticket(
                    str(contact_id),
                    "Must not open",
                )
            )
            assert refused == {
                "success": False,
                "error": "Contact not found",
            }

        now = datetime.now(timezone.utc)
        await pool.executemany(
            """
            INSERT INTO customer_service_tickets (
                contact_id,
                business_context_id,
                summary,
                status,
                resolution,
                closed_at,
                created_at,
                updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
            """,
            [
                (
                    foreign_contact["id"],
                    FOREIGN_TENANT,
                    "Newer foreign open ticket",
                    "open",
                    None,
                    None,
                    now + timedelta(minutes=2),
                ),
                (
                    legacy_contact["id"],
                    TENANT,
                    "Newer tenant closed ticket",
                    "closed",
                    "Already resolved",
                    now + timedelta(minutes=1),
                    now + timedelta(minutes=1),
                ),
            ],
        )

        invalid_status = json.loads(
            await crm_mcp.list_customer_service_tickets(status="pending")
        )
        assert invalid_status == {
            "error": "status must be open, closed, or null",
            "tickets": [],
            "count": 0,
        }

        open_queue = json.loads(
            await crm_mcp.list_customer_service_tickets(limit=1)
        )
        assert open_queue["count"] == 1
        assert open_queue["tickets"][0]["id"] == ticket_id
        assert open_queue["tickets"][0]["business_context_id"] == TENANT

        foreign_update = json.loads(
            await crm_mcp.update_customer_service_ticket(
                ticket_id,
                priority="low",
                business_context_id=FOREIGN_TENANT,
            )
        )
        assert foreign_update == {
            "success": False,
            "error": "Ticket not found or not open",
        }
        foreign_close = json.loads(
            await crm_mcp.close_customer_service_ticket(
                ticket_id,
                "Must not close",
                business_context_id=FOREIGN_TENANT,
            )
        )
        assert foreign_close == {
            "success": False,
            "error": "Ticket not found",
        }

        empty_update = json.loads(
            await crm_mcp.update_customer_service_ticket(ticket_id)
        )
        assert empty_update == {
            "success": False,
            "error": "No fields provided to update",
        }
        blank_update = json.loads(
            await crm_mcp.update_customer_service_ticket(
                ticket_id,
                priority=" ",
            )
        )
        assert blank_update == {
            "success": False,
            "error": "priority is required",
        }
        overlong_update = json.loads(
            await crm_mcp.update_customer_service_ticket(
                ticket_id,
                assignee="x" * 129,
            )
        )
        assert overlong_update == {
            "success": False,
            "error": "assignee must be at most 128 characters",
        }

        updated = json.loads(
            await crm_mcp.update_customer_service_ticket(
                ticket_id,
                priority="high",
                assignee="Kennedi",
            )
        )
        assert updated["success"] is True
        assert updated["ticket"]["priority"] == "high"
        assert updated["ticket"]["assignee"] == "Kennedi"

        close_results = [
            json.loads(raw)
            for raw in await asyncio.gather(
                crm_mcp.close_customer_service_ticket(
                    ticket_id,
                    "Returned the next morning and remopped the floor.",
                ),
                crm_mcp.close_customer_service_ticket(
                    ticket_id,
                    "Issued a service credit.",
                ),
            )
        ]
        assert all(result["success"] is True for result in close_results)
        assert {
            result["already_closed"] for result in close_results
        } == {False, True}
        first_close = next(
            result for result in close_results
            if result["already_closed"] is False
        )
        repeated_close = next(
            result for result in close_results
            if result["already_closed"] is True
        )
        assert first_close["ticket"]["status"] == "closed"
        first_resolution = first_close["ticket"]["resolution"]
        first_closed_at = first_close["ticket"]["closed_at"]
        assert repeated_close["ticket"]["resolution"] == first_resolution
        assert repeated_close["ticket"]["closed_at"] == first_closed_at

        repeated = json.loads(
            await crm_mcp.close_customer_service_ticket(
                ticket_id,
                "A retry must not replace the first resolution.",
            )
        )
        assert repeated["success"] is True
        assert repeated["already_closed"] is True
        assert repeated["ticket"]["resolution"] == first_resolution
        assert repeated["ticket"]["closed_at"] == first_closed_at

        closed_update = json.loads(
            await crm_mcp.update_customer_service_ticket(
                ticket_id,
                assignee="Nobody",
            )
        )
        assert closed_update == {
            "success": False,
            "error": "Ticket not found or not open",
        }

        remaining_open = json.loads(
            await crm_mcp.list_customer_service_tickets(limit=10)
        )
        assert remaining_open == {"tickets": [], "count": 0}

        closed_rows = json.loads(
            await crm_mcp.list_customer_service_tickets(
                status="closed",
                contact_id=str(legacy_contact["id"]),
                limit=10,
            )
        )
        assert closed_rows["count"] == 2
        assert {row["business_context_id"] for row in closed_rows["tickets"]} == {
            TENANT
        }
    finally:
        crm_mcp.set_provider_override(None)
        if pool is not None:
            await pool.close()
        await admin_conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin_conn.close()
