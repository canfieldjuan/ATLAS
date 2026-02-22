"""
CRM provider abstraction for Atlas.

Provider-agnostic interface for customer/contact management.
The `contacts` table is the single source of truth for all customer data.

Two concrete providers are available:
  - DirectusCRMProvider  — uses the Directus REST API (preferred when Directus
                           is running; configured via ATLAS_DIRECTUS_* env vars).
  - DatabaseCRMProvider  — queries the `contacts` table directly via asyncpg
                           (fallback for local dev or when Directus is offline).

Both write to the same Postgres tables created by migration 035_contacts.sql.

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

import httpx

logger = logging.getLogger("atlas.services.crm_provider")


# ---------------------------------------------------------------------------
# DirectusCRMProvider
# ---------------------------------------------------------------------------

class DirectusCRMProvider:
    """
    CRM provider backed by the Directus REST API.

    Directus manages the `contacts` and `contact_interactions` collections on
    top of the existing atlas_postgres instance.  Authentication uses a static
    admin token (ATLAS_DIRECTUS_TOKEN) which you can generate from the Directus
    admin UI: Settings → API tokens.
    """

    def __init__(self) -> None:
        from ..config import settings
        self._cfg = settings.directus
        self._client: Optional[httpx.AsyncClient] = None

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"
        return headers

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._cfg.url,
                timeout=self._cfg.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        try:
            client = await self._ensure_client()
            resp = await client.get("/server/health", headers=self._headers())
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("Directus health check failed: %s", exc)
            return False

    # -----------------------------------------------------------------------
    # Contacts CRUD
    # -----------------------------------------------------------------------

    async def create_contact(self, data: dict[str, Any]) -> dict[str, Any]:
        client = await self._ensure_client()
        # Normalize email to lowercase for consistent _eq filtering
        if data.get("email"):
            data = dict(data, email=data["email"].lower())
        resp = await client.post(
            "/items/contacts", json=data, headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json().get("data", {})

    async def get_contact(self, contact_id: str) -> Optional[dict[str, Any]]:
        try:
            client = await self._ensure_client()
            resp = await client.get(
                f"/items/contacts/{contact_id}", headers=self._headers()
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json().get("data")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    async def search_contacts(
        self,
        query: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        business_context_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        params: dict[str, Any] = {"limit": limit, "sort": "-updated_at"}

        filters: list[dict] = []
        if phone:
            digits = "".join(c for c in phone if c.isdigit())
            # Match last 10 digits (strips country code noise)
            filters.append({"phone": {"_contains": digits[-10:]}})
        if email:
            filters.append({"email": {"_eq": email.lower()}})
        if business_context_id:
            filters.append({"business_context_id": {"_eq": business_context_id}})
        if query:
            filters.append({"full_name": {"_icontains": query}})

        if filters:
            params["filter"] = json.dumps(
                {"_and": filters} if len(filters) > 1 else filters[0]
            )

        resp = await client.get(
            "/items/contacts", params=params, headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    async def update_contact(
        self, contact_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        client = await self._ensure_client()
        data = dict(data)
        # Normalize email to lowercase if being updated
        if data.get("email"):
            data["email"] = data["email"].lower()
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        resp = await client.patch(
            f"/items/contacts/{contact_id}", json=data, headers=self._headers()
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("data")

    async def delete_contact(self, contact_id: str) -> bool:
        """Soft-delete: set status → archived."""
        client = await self._ensure_client()
        resp = await client.patch(
            f"/items/contacts/{contact_id}",
            json={
                "status": "archived",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            headers=self._headers(),
        )
        return resp.status_code in (200, 204)

    async def list_contacts(
        self,
        business_context_id: Optional[str] = None,
        status: Optional[str] = "active",
        contact_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "sort": "full_name",
        }
        filters: list[dict] = []
        if status:
            filters.append({"status": {"_eq": status}})
        if business_context_id:
            filters.append({"business_context_id": {"_eq": business_context_id}})
        if contact_type:
            filters.append({"contact_type": {"_eq": contact_type}})
        if filters:
            params["filter"] = json.dumps(
                {"_and": filters} if len(filters) > 1 else filters[0]
            )
        resp = await client.get(
            "/items/contacts", params=params, headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    # -----------------------------------------------------------------------
    # Interactions
    # -----------------------------------------------------------------------

    async def log_interaction(
        self,
        contact_id: str,
        interaction_type: str,
        summary: str,
        occurred_at: Optional[str] = None,
    ) -> dict[str, Any]:
        client = await self._ensure_client()
        payload = {
            "contact_id": contact_id,
            "interaction_type": interaction_type,
            "summary": summary,
            "occurred_at": occurred_at or datetime.now(timezone.utc).isoformat(),
        }
        resp = await client.post(
            "/items/contact_interactions", json=payload, headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json().get("data", {})

    async def get_interactions(
        self, contact_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        params = {
            "filter": json.dumps({"contact_id": {"_eq": contact_id}}),
            "sort": "-occurred_at",
            "limit": limit,
        }
        resp = await client.get(
            "/items/contact_interactions", params=params, headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    # -----------------------------------------------------------------------
    # Cross-table: appointments linked to a contact
    # -----------------------------------------------------------------------

    async def get_contact_appointments(
        self, contact_id: str
    ) -> list[dict[str, Any]]:
        """Return appointments that have been linked to this contact via contact_id FK."""
        try:
            from ..storage.database import get_db_pool

            pool = get_db_pool()
            if not pool.is_initialized:
                return []
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
        except Exception as exc:
            logger.warning(
                "Failed to fetch appointments for contact %s: %s", contact_id, exc
            )
            return []


# ---------------------------------------------------------------------------
# DatabaseCRMProvider  (asyncpg direct — fallback when Directus is offline)
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

_crm_provider: Optional[DirectusCRMProvider | DatabaseCRMProvider] = None


def get_crm_provider() -> DirectusCRMProvider | DatabaseCRMProvider:
    """
    Return the configured CRM provider singleton.

    Uses DirectusCRMProvider when ATLAS_DIRECTUS_ENABLED=true and a token is
    configured; otherwise falls back to DatabaseCRMProvider (direct asyncpg).
    """
    global _crm_provider
    if _crm_provider is None:
        from ..config import settings

        if settings.directus.enabled and settings.directus.token:
            _crm_provider = DirectusCRMProvider()
            logger.info("CRM provider: Directus at %s", settings.directus.url)
        else:
            _crm_provider = DatabaseCRMProvider()
            logger.info(
                "CRM provider: direct database "
                "(set ATLAS_DIRECTUS_ENABLED=true + ATLAS_DIRECTUS_TOKEN to use Directus)"
            )
    return _crm_provider
