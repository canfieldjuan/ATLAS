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
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("atlas.services.crm_provider")

_INTERACTION_DEDUPE_SUMMARY_MAX_CHARS = 2000
_INTERACTION_DEDUPE_ANCHOR_KEYS = (
    "crm_event_id",
    "source_ref",
    "message_id",
    "gmail_message_id",
    "email_message_id",
    "thread_id",
    "appointment_id",
    "invoice_id",
    "external_id",
)


def _normalize_interaction_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _coerce_occurrence(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        parsed = datetime.fromisoformat(str(value))
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _interaction_anchor(metadata: dict[str, Any]) -> str:
    for key in _INTERACTION_DEDUPE_ANCHOR_KEYS:
        value = metadata.get(key)
        normalized = _normalize_interaction_text(value)
        if normalized:
            return f"{key}:{normalized}"
    return ""


def _interaction_dedupe_key(
    *,
    interaction_type: str,
    summary: str,
    occurred_at: datetime,
    intent: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    normalized_type = _normalize_interaction_text(interaction_type)
    if not normalized_type:
        return None
    normalized_intent = _normalize_interaction_text(intent)
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    anchor = _interaction_anchor(metadata_dict)
    if anchor:
        basis = f"anchor|{normalized_type}|{anchor}"
        return hashlib.md5(basis.encode("utf-8")).hexdigest()
    normalized_summary = _normalize_interaction_text(summary)
    if not normalized_summary:
        return None
    bucket = occurred_at.astimezone(timezone.utc).date().isoformat()
    basis = "|".join(
        [
            "daily",
            normalized_type,
            bucket,
            normalized_intent,
            normalized_summary[:_INTERACTION_DEDUPE_SUMMARY_MAX_CHARS],
        ]
    )
    return hashlib.md5(basis.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# DatabaseCRMProvider  (asyncpg direct)
# ---------------------------------------------------------------------------

class DatabaseCRMProvider:
    """CRM provider -- queries the `contacts` table directly via asyncpg."""

    async def health_check(self) -> bool:
        try:
            from ..storage.database import get_db_pool

            return get_db_pool().is_initialized
        except Exception:
            return False

    async def create_contact(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Create a contact, returning an existing one if phone or email already matches.

        Dedup order: phone first (more unique), then email.  If a match is found the
        existing record is updated with any non-null fields from `data` so the caller
        always gets the most complete version.  This is application-level dedup;
        migration 037 should add a DB-level partial unique index for extra safety.
        """
        pipeline_fields = ("lead_stage", "lead_owner", "next_follow_up_at")
        if (
            any(data.get(field) is not None for field in pipeline_fields)
            and data.get("contact_type", "customer") != "lead"
        ):
            raise ValueError("Lead pipeline fields require contact_type='lead'")

        # --- dedup check ---
        raw_email = data.get("email")
        email = raw_email.lower() if raw_email else None
        phone = data.get("phone")

        # Tenant-scoped dedup: when the caller stamps a business_context_id,
        # match contacts within that tenant OR historical contacts with no
        # context yet (which the merge below then claims for the tenant) --
        # but never a contact belonging to a DIFFERENT context
        # (PR #2152/#2153 review findings, R3/R4/R5).
        ctx = data.get("business_context_id")

        def _ctx_compatible(candidate: dict[str, Any]) -> bool:
            if not ctx:
                return True
            existing_ctx = candidate.get("business_context_id")
            return existing_ctx is None or existing_ctx == ctx

        def _pick(matches: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
            # Prefer a real same-tenant contact over a NULL-context historical
            # row so a claimable legacy row cannot shadow the tenant's own
            # record (PR #2153 round 4, R4/R5).
            if ctx:
                for m in matches:
                    if m.get("business_context_id") == ctx:
                        return m
            for m in matches:
                if _ctx_compatible(m):
                    return m
            return None

        async def _resolve(**channel: Any) -> Optional[dict[str, Any]]:
            # Same-tenant page first, then the NULL-context (claimable) page
            # directly -- both queries name their exact population, so a crowd
            # of recently-updated foreign contacts can never page-starve the
            # match (PR #2153 rounds 6-7, R4/R5).
            if ctx:
                scoped = await self.search_contacts(business_context_id=ctx, **channel)
                if scoped:
                    return scoped[0]
                claimable = await self.search_contacts(
                    business_context_id_is_null=True, **channel
                )
                return claimable[0] if claimable else None
            return _pick(await self.search_contacts(**channel))

        existing: Optional[dict[str, Any]] = None
        if phone:
            existing = await _resolve(phone=phone)
        if existing is None and email:
            existing = await _resolve(email=email)

        if existing is not None and ctx and existing.get("business_context_id") is None:
            # Claim the NULL-context legacy match by compare-and-set before
            # merging: a concurrent claim by another tenant leaves the row
            # theirs and this create falls through to a fresh insert instead
            # of overwriting their claim.
            claimed = await self.claim_contact(str(existing["id"]), ctx)
            if claimed is None:
                existing = None
            else:
                existing = claimed

        if existing is not None:
            # Merge any new non-null fields into the existing record
            _MERGEABLE = {
                "full_name", "first_name", "last_name", "email", "phone",
                "address", "city", "state", "zip", "contact_type",
                "tags", "notes", "business_context_id", "source", "source_ref",
            }
            updates = {
                k: (v.lower() if k == "email" and v else v)
                for k, v in data.items()
                if k in _MERGEABLE and v
            }
            if updates:
                merged = await self.update_contact(existing["id"], updates)
                result = merged or existing
            else:
                result = existing
            result["_was_created"] = False
            return result

        # --- no existing contact -- insert ---
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        contact_id = str(uuid4())
        now = datetime.now(timezone.utc)
        metadata_json = json.dumps(data.get("metadata", {}))

        row = await pool.fetchrow(
            """
            INSERT INTO contacts (
                id, full_name, first_name, last_name, email, phone,
                address, city, state, zip, business_context_id,
                contact_type, status, tags, notes, source, source_ref,
                lead_stage, lead_owner, next_follow_up_at,
                created_at, updated_at, metadata
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
                $18,$19,$20,$21,$22,$23::jsonb
            ) RETURNING *
            """,
            contact_id,
            data.get("full_name", ""),
            data.get("first_name"),
            data.get("last_name"),
            email,  # normalized lowercase
            phone,
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
            data.get("lead_stage"),
            data.get("lead_owner"),
            data.get("next_follow_up_at"),
            now,   # created_at ($21)
            now,   # updated_at ($22) -- same value on insert
            metadata_json,
        )
        result = dict(row) if row else {}
        result["_was_created"] = True
        return result

    async def find_or_create_contact(
        self,
        full_name: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """
        Convenience method: find existing contact by phone/email or create a new one.

        Used by booking workflows (J3) and call intelligence (S2) to reliably
        resolve a customer to a single contact record.

        Returns the contact dict (existing or newly created).
        """
        data: dict[str, Any] = {"full_name": full_name}
        if phone:
            data["phone"] = phone
        if email:
            data["email"] = email
        data.update(extra)
        result = await self.create_contact(data)

        # Emit event for reasoning agent
        from ..reasoning.producers import emit_if_enabled
        await emit_if_enabled(
            "crm.contact_created", "crm_provider",
            {"contact_id": result.get("id", ""), "full_name": full_name,
             "email": email, "phone": phone},
            entity_type="contact",
            entity_id=result.get("id"),
        )
        return result

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
        business_context_id_is_null: bool = False,
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
        if business_context_id_is_null:
            conditions.append("business_context_id IS NULL")
        if query:
            conditions.append(f"full_name ILIKE ${idx}")
            params.append(f"%{query[:200]}%")
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
        self,
        contact_id: str,
        data: dict[str, Any],
        *,
        require_contact_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        allowed = {
            "full_name", "first_name", "last_name", "email", "phone",
            "address", "city", "state", "zip", "contact_type", "status",
            "tags", "notes", "business_context_id", "source", "source_ref",
            "metadata", "lead_stage", "lead_owner", "next_follow_up_at",
        }
        updates = {k: v for k, v in data.items() if k in allowed}
        pipeline_requested = any(
            key in updates
            for key in ("lead_stage", "lead_owner", "next_follow_up_at")
        )
        if pipeline_requested:
            if (
                "contact_type" in updates
                and updates["contact_type"] != "lead"
            ):
                raise ValueError(
                    "Lead pipeline fields require contact_type='lead'"
                )
            if require_contact_type not in (None, "lead"):
                raise ValueError(
                    "Lead pipeline fields require contact_type='lead'"
                )
            require_contact_type = "lead"
        if "email" in updates and updates["email"]:
            updates["email"] = updates["email"].lower()
        if "metadata" in updates:
            updates["metadata"] = json.dumps(updates["metadata"]) if isinstance(updates["metadata"], dict) else updates["metadata"]
        if not updates:
            return await self.get_contact(contact_id)

        updates["updated_at"] = datetime.now(timezone.utc)
        set_parts: list[str] = []
        params: list[Any] = [contact_id]
        for i, (key, val) in enumerate(updates.items(), start=2):
            cast = "::jsonb" if key == "metadata" else ""
            set_parts.append(f"{key} = ${i}{cast}")
            params.append(val)

        where = "id = $1"
        if require_contact_type is not None:
            params.append(require_contact_type)
            where += f" AND contact_type = ${len(params)}"

        row = await pool.fetchrow(
            f"UPDATE contacts SET {', '.join(set_parts)} WHERE {where} RETURNING *",
            *params,
        )
        return dict(row) if row else None

    async def claim_contact(
        self, contact_id: str, business_context_id: str
    ) -> Optional[dict[str, Any]]:
        """Compare-and-set claim of a legacy row for a tenant.

        Stamps ``business_context_id`` only while the row is still NULL (or
        already carries the same tenant, making the claim idempotent).
        Returns None when a concurrent claim moved the row to a different
        tenant, so callers can fail closed instead of overwriting it.
        """
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            UPDATE contacts
               SET business_context_id = $2, updated_at = NOW()
             WHERE id = $1
               AND (business_context_id IS NULL OR business_context_id = $2)
             RETURNING *
            """,
            contact_id,
            business_context_id,
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
        business_context_id_is_null: bool = False,
        include_unclaimed_legacy: bool = False,
        status: Optional[str] = "active",
        contact_type: Optional[str] = None,
        lead_stage: Optional[str] = None,
        lead_owner: Optional[str] = None,
        next_follow_up_before: Optional[datetime] = None,
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
        if include_unclaimed_legacy and not business_context_id:
            raise ValueError(
                "include_unclaimed_legacy requires business_context_id"
            )
        if include_unclaimed_legacy and business_context_id_is_null:
            raise ValueError(
                "include_unclaimed_legacy conflicts with business_context_id_is_null"
            )
        if business_context_id:
            operator = (
                f"(business_context_id = ${idx} OR business_context_id IS NULL)"
                if include_unclaimed_legacy
                else f"business_context_id = ${idx}"
            )
            conditions.append(operator)
            params.append(business_context_id)
            idx += 1
        if business_context_id_is_null:
            conditions.append("business_context_id IS NULL")
        if contact_type:
            conditions.append(f"contact_type = ${idx}")
            params.append(contact_type)
            idx += 1
        if lead_stage:
            conditions.append(f"lead_stage = ${idx}")
            params.append(lead_stage)
            idx += 1
        if lead_owner:
            conditions.append(f"lead_owner = ${idx}")
            params.append(lead_owner)
            idx += 1
        if next_follow_up_before is not None:
            conditions.append(f"next_follow_up_at <= ${idx}")
            params.append(next_follow_up_before)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, max(0, offset)])
        order_by = (
            "next_follow_up_at ASC, full_name ASC"
            if next_follow_up_before is not None
            else "full_name ASC"
        )
        rows = await pool.fetch(
            f"""
            SELECT * FROM contacts {where}
            ORDER BY {order_by}
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
        intent: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        interaction_id = str(uuid4())
        occ = _coerce_occurrence(occurred_at)
        metadata_json = json.dumps(metadata or {})
        dedupe_key = _interaction_dedupe_key(
            interaction_type=interaction_type,
            summary=summary,
            occurred_at=occ,
            intent=intent,
            metadata=metadata or {},
        )
        row = await pool.fetchrow(
            """
            WITH inserted AS (
                INSERT INTO contact_interactions
                    (id, contact_id, interaction_type, summary, occurred_at, intent, metadata, interaction_dedupe_key)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
                ON CONFLICT (contact_id, interaction_type, interaction_dedupe_key)
                    WHERE interaction_dedupe_key IS NOT NULL
                    DO NOTHING
                RETURNING contact_interactions.*, true AS _inserted
            )
            SELECT * FROM inserted
            UNION ALL
            SELECT ci.*, false AS _inserted
            FROM contact_interactions ci
            WHERE ci.contact_id = $2
              AND ci.interaction_type = $3
              AND ci.interaction_dedupe_key = $8
              AND NOT EXISTS (SELECT 1 FROM inserted)
            LIMIT 1
            """,
            interaction_id,
            contact_id,
            interaction_type,
            summary,
            occ,
            intent,
            metadata_json,
            dedupe_key,
        )
        result = dict(row) if row else {}
        inserted = bool(result.pop("_inserted", False))
        # Public flag for callers that gate side effects (e.g. acknowledgement
        # emails) on first-time-vs-duplicate; the raw column is stripped above.
        result["inserted"] = inserted

        if inserted:
            # Emit event for reasoning agent only for new interactions.
            from ..reasoning.producers import emit_if_enabled
            await emit_if_enabled(
                "crm.interaction_logged", "crm_provider",
                {
                    "contact_id": contact_id,
                    "interaction_id": result.get("id"),
                    "interaction_type": interaction_type,
                    "intent": intent,
                    "summary_preview": summary[:200],
                },
                entity_type="contact",
                entity_id=contact_id,
            )
        elif dedupe_key:
            logger.info(
                "Suppressed duplicate CRM interaction contact_id=%s type=%s dedupe_key=%s",
                contact_id,
                interaction_type,
                dedupe_key[:12],
            )
        return result

    async def get_interactions(
        self, contact_id: str, limit: int = 20,
        business_context_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if business_context_id:
            # Atomic tenant predicate: the page only returns while the
            # owning contact is still visible to this tenant (tenant page
            # plus NULL-context legacy), closing the window between a
            # caller's guard read and this query.
            rows = await pool.fetch(
                """
                SELECT ci.* FROM contact_interactions ci
                JOIN contacts c ON c.id = ci.contact_id
                WHERE ci.contact_id = $1
                  AND (c.business_context_id = $2 OR c.business_context_id IS NULL)
                ORDER BY ci.occurred_at DESC
                LIMIT $3
                """,
                contact_id,
                business_context_id,
                limit,
            )
        else:
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
        self, contact_id: str, business_context_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if business_context_id:
            rows = await pool.fetch(
                """
                SELECT id, start_time, end_time, service_type, status,
                       customer_name, customer_phone, customer_email,
                       customer_address, notes, created_at, business_context_id
                FROM appointments
                WHERE contact_id = $1
                  AND business_context_id = $2
                ORDER BY start_time DESC
                LIMIT 50
                """,
                contact_id,
                business_context_id,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT id, start_time, end_time, service_type, status,
                       customer_name, customer_phone, customer_email,
                       customer_address, notes, created_at, business_context_id
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
