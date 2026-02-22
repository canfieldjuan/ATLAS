"""
CRM provider abstraction for Atlas.

Provider-agnostic interface for customer/contact management.
The `contacts` table (migration 035_contacts.sql) is the single source of truth.

DatabaseCRMProvider queries Postgres directly via asyncpg.
NocoDB (http://localhost:8080) provides a browser UI over the same tables.

Usage:
    from atlas_brain.services.crm_provider import get_crm_provider

    provider = get_crm_provider()
    results = await provider.search_contacts(phone="618-555-1234")
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("atlas.services.crm_provider")


# ---------------------------------------------------------------------------
# DatabaseCRMProvider  (asyncpg direct)
# ---------------------------------------------------------------------------

class DatabaseCRMProvider:
    """
    Fallback CRM provider that queries the `contacts` table directly via asyncpg.

    Used in local dev or when Directus is not running.  Produces identical data
    to DirectusCRMProvider because both operate on the same Postgres tables.
    """

    async def health_check(self) -> bool:
        try:
            from ..storage.database import get_db_pool

            return get_db_pool().is_initialized
        except Exception:
            return False

    async def create_contact(self, data: dict[str, Any]) -> dict[str, Any]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        contact_id = str(uuid4())
        now = datetime.now(timezone.utc)
        metadata_json = json.dumps(data.get("metadata", {}))
        # Normalize email to lowercase for consistent searches
        raw_email = data.get("email")
        email = raw_email.lower() if raw_email else None

        row = await pool.fetchrow(
            """
            INSERT INTO contacts (
                id, full_name, first_name, last_name, email, phone,
                address, city, state, zip, business_context_id,
                contact_type, status, tags, notes, source, source_ref,
                created_at, updated_at, metadata
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20::jsonb
            ) RETURNING *
            """,
            contact_id,
            data.get("full_name", ""),
            data.get("first_name"),
            data.get("last_name"),
            email,  # normalized lowercase
            data.get("phone"),
            data.get("address"),
            data.get("city"),
            data.get("state"),
            data.get("zip"),
            data.get("business_context_id"),
            data.get("contact_type", "customer"),
            data.get("status", "active"),
            data.get("tags", []),
            data.get("notes"),
            data.get("source", "manual"),
            data.get("source_ref"),
            now,   # created_at ($18)
            now,   # updated_at ($19) — same value on insert
            metadata_json,
        )
        return dict(row) if row else {}

    async def get_contact(self, contact_id: str) -> Optional[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT * FROM contacts WHERE id = $1", contact_id
        )
        return dict(row) if row else None

    async def search_contacts(
        self,
        query: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        business_context_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions: list[str] = ["status != 'archived'"]
        params: list[Any] = []
        idx = 1

        if phone:
            digits = "".join(c for c in phone if c.isdigit())
            conditions.append(
                f"REGEXP_REPLACE(phone, '[^0-9]', '', 'g') LIKE ${idx}"
            )
            params.append(f"%{digits[-10:]}%")
            idx += 1
        if email:
            conditions.append(f"LOWER(email) = LOWER(${idx})")
            params.append(email)
            idx += 1
        if business_context_id:
            conditions.append(f"business_context_id = ${idx}")
            params.append(business_context_id)
            idx += 1
        if query:
            conditions.append(f"full_name ILIKE ${idx}")
            params.append(f"%{query}%")
            idx += 1

        params.append(limit)
        rows = await pool.fetch(
            f"""
            SELECT * FROM contacts
            WHERE {' AND '.join(conditions)}
            ORDER BY updated_at DESC
            LIMIT ${idx}
            """,
            *params,
        )
        return [dict(r) for r in rows]

    async def update_contact(
        self, contact_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        allowed = {
            "full_name", "first_name", "last_name", "email", "phone",
            "address", "city", "state", "zip", "contact_type", "status",
            "tags", "notes", "business_context_id", "source", "source_ref",
        }
        updates = {k: v for k, v in data.items() if k in allowed}
        if not updates:
            return await self.get_contact(contact_id)

        updates["updated_at"] = datetime.now(timezone.utc)
        set_parts: list[str] = []
        params: list[Any] = [contact_id]
        for i, (key, val) in enumerate(updates.items(), start=2):
            set_parts.append(f"{key} = ${i}")
            params.append(val)

        row = await pool.fetchrow(
            f"UPDATE contacts SET {', '.join(set_parts)} WHERE id = $1 RETURNING *",
            *params,
        )
        return dict(row) if row else None

    async def delete_contact(self, contact_id: str) -> bool:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        result = await pool.execute(
            "UPDATE contacts SET status = 'archived', updated_at = NOW() WHERE id = $1",
            contact_id,
        )
        return "UPDATE 1" in (result or "")

    async def list_contacts(
        self,
        business_context_id: Optional[str] = None,
        status: Optional[str] = "active",
        contact_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        if business_context_id:
            conditions.append(f"business_context_id = ${idx}")
            params.append(business_context_id)
            idx += 1
        if contact_type:
            conditions.append(f"contact_type = ${idx}")
            params.append(contact_type)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])
        rows = await pool.fetch(
            f"""
            SELECT * FROM contacts {where}
            ORDER BY full_name ASC
            LIMIT ${idx} OFFSET ${idx + 1}
            """,
            *params,
        )
        return [dict(r) for r in rows]

    async def log_interaction(
        self,
        contact_id: str,
        interaction_type: str,
        summary: str,
        occurred_at: Optional[str] = None,
    ) -> dict[str, Any]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        interaction_id = str(uuid4())
        occ = (
            datetime.fromisoformat(occurred_at)
            if occurred_at
            else datetime.now(timezone.utc)
        )
        row = await pool.fetchrow(
            """
            INSERT INTO contact_interactions
                (id, contact_id, interaction_type, summary, occurred_at)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            interaction_id,
            contact_id,
            interaction_type,
            summary,
            occ,
        )
        return dict(row) if row else {}

    async def get_interactions(
        self, contact_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM contact_interactions
            WHERE contact_id = $1
            ORDER BY occurred_at DESC
            LIMIT $2
            """,
            contact_id,
            limit,
        )
        return [dict(r) for r in rows]

    async def get_contact_appointments(
        self, contact_id: str
    ) -> list[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        rows = await pool.fetch(
            """
            SELECT id, start_time, end_time, service_type, status,
                   customer_name, customer_phone, customer_email,
                   customer_address, notes, created_at
            FROM appointments
            WHERE contact_id = $1
            ORDER BY start_time DESC
            LIMIT 50
            """,
            contact_id,
        )
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_crm_provider: Optional[DatabaseCRMProvider] = None


def get_crm_provider() -> DatabaseCRMProvider:
    """Return the DatabaseCRMProvider singleton (direct asyncpg queries)."""
    global _crm_provider
    if _crm_provider is None:
        _crm_provider = DatabaseCRMProvider()
        logger.info("CRM provider: DatabaseCRMProvider (direct asyncpg)")
    return _crm_provider
