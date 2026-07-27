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
import re
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def _transaction_connection(pool: Any):
    """Yield a transaction from Atlas' wrapper, asyncpg connection, or pool.

    Production uses ``DatabasePool.transaction``.  Supporting a raw asyncpg
    connection/pool keeps the migration integration proof on the real SQL path.
    """
    transaction = getattr(pool, "transaction", None)
    if callable(transaction):
        # ``asyncpg.Connection.transaction()`` enters as ``None``; the raw
        # connection itself remains the query object. Atlas' DatabasePool and
        # test adapters instead enter as a connection.
        if hasattr(pool, "fetchrow") and not hasattr(pool, "acquire"):
            async with transaction():
                yield pool
            return
        async with transaction() as connection:
            yield connection
        return
    # Lightweight repository test adapters expose query methods directly but
    # deliberately do not model transaction/acquire.  They are not a runtime
    # database implementation; preserving their direct query boundary keeps
    # intake's injectable test surface intact while real pools take one of the
    # transaction branches above.
    if hasattr(pool, "fetchrow") and not hasattr(pool, "acquire"):
        yield pool
        return
    async with pool.acquire() as connection:
        async with connection.transaction():
            yield connection


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


def _interaction_attribution_identity(metadata: dict[str, Any]) -> str:
    """Return a stable identity for a non-empty lead-attribution snapshot.

    The dedupe key is stored as a digest, not this value.  Retaining a distinct
    interaction when attribution changes avoids silently discarding the only
    click-level evidence that a repeat form submission carries.
    """

    attribution = metadata.get("attribution")
    if not isinstance(attribution, dict):
        return ""
    normalized = sorted(
        (
            _normalize_interaction_text(key),
            str(value) if value is not None else "",
        )
        for key, value in attribution.items()
        if _normalize_interaction_text(key)
        and value is not None
        and str(value).strip()
    )
    return json.dumps(normalized, separators=(",", ":"))


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
    attribution_identity = _interaction_attribution_identity(metadata_dict)
    if anchor:
        basis = f"anchor|{normalized_type}|{anchor}"
        if attribution_identity:
            basis = f"{basis}|attribution|{attribution_identity}"
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
    if attribution_identity:
        basis = f"{basis}|attribution|{attribution_identity}"
    return hashlib.md5(basis.encode("utf-8")).hexdigest()


async def _write_contact_interaction(
    executor: Any,
    *,
    contact_id: str,
    interaction_type: str,
    summary: str,
    occurred_at: Optional[str] = None,
    intent: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Insert or return one deduplicated interaction through ``executor``.

    ``executor`` is either the normal pool or a connection that already owns an
    inbound-delivery transaction.  Keeping the SQL here lets the combined EOM
    command use exactly the public interaction dedupe contract before it
    releases the selected contact row.
    """
    interaction_id = str(uuid4())
    occ = _coerce_occurrence(occurred_at)
    metadata_dict = metadata or {}
    metadata_json = json.dumps(metadata_dict)
    dedupe_key = _interaction_dedupe_key(
        interaction_type=interaction_type,
        summary=summary,
        occurred_at=occ,
        intent=intent,
        metadata=metadata_dict,
    )
    row = await executor.fetchrow(
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
    result["inserted"] = bool(result.pop("_inserted", False))
    return result


# ---------------------------------------------------------------------------
# DatabaseCRMProvider  (asyncpg direct)
# ---------------------------------------------------------------------------

class DatabaseCRMProvider:
    """CRM provider -- queries the `contacts` table directly via asyncpg."""

    def __init__(self, *, pool: Any | None = None) -> None:
        """Use the configured pool, or a supplied transaction-capable adapter."""
        self._pool_override = pool

    def _get_pool(self) -> Any:
        if self._pool_override is not None:
            return self._pool_override
        from ..storage.database import get_db_pool

        return get_db_pool()

    async def health_check(self) -> bool:
        try:
            from ..storage.database import get_db_pool

            return get_db_pool().is_initialized
        except Exception:
            return False

    async def _emit_contact_created(
        self,
        result: dict[str, Any],
        *,
        full_name: str,
        email: Optional[str],
        phone: Optional[str],
    ) -> None:
        """Emit the existing reasoning event after a committed contact insert."""
        from ..reasoning.producers import emit_if_enabled

        await emit_if_enabled(
            "crm.contact_created",
            "crm_provider",
            {
                "contact_id": result.get("id", ""),
                "full_name": full_name,
                "email": email,
                "phone": phone,
            },
            entity_type="contact",
            entity_id=result.get("id"),
        )

    async def _emit_interaction_logged(
        self,
        *,
        contact_id: str,
        interaction: dict[str, Any],
        interaction_type: str,
        intent: Optional[str],
        summary: str,
    ) -> None:
        """Emit the established reasoning event after an interaction commit."""
        from ..reasoning.producers import emit_if_enabled

        await emit_if_enabled(
            "crm.interaction_logged",
            "crm_provider",
            {
                "contact_id": contact_id,
                "interaction_id": interaction.get("id"),
                "interaction_type": interaction_type,
                "intent": intent,
                "summary_preview": summary[:200],
            },
            entity_type="contact",
            entity_id=contact_id,
        )

    async def resolve_or_create_eom_inbound_lead_atomic(
        self,
        *,
        full_name: str,
        phone: Optional[str],
        email: Optional[str],
        address: Optional[str],
        source: str,
        source_ref: Optional[str],
        relay_event_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        interaction: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Resolve EOM inbound identity under transaction-scoped advisory locks.

        There is intentionally no global phone/email uniqueness migration: old
        tenant data may contain legitimate historical duplicates.  Instead all
        current EOM inbound writers share a stable lock for each asserted
        identity, then perform the exact scoped lookup and insert in one
        transaction.  A legacy row is returned untouched, never claimed or
        merged from extracted/web-form data.
        """
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID
        from ..storage.database import get_db_pool

        normalized_email = str(email or "").strip().lower()
        phone_digits = re.sub(r"\D", "", str(phone or ""))
        if len(phone_digits) < 10:
            phone_digits = ""
        normalized_source = str(source or "").strip()
        normalized_source_ref = str(source_ref or "").strip()
        normalized_relay_event_id = str(relay_event_id or "").strip()
        identityless_relay = not phone_digits and not normalized_email
        if identityless_relay and not (
            normalized_source and normalized_relay_event_id
        ):
            raise ValueError(
                "EOM inbound lead requires phone, email, or a stable relay event identity"
            )
        lock_keys = []
        if phone_digits:
            lock_keys.append(f"eom-inbound:phone:{phone_digits[-10:]}")
        if normalized_email:
            lock_keys.append(f"eom-inbound:email:{normalized_email}")
        if normalized_relay_event_id:
            lock_keys.append(
                f"eom-inbound:relay:{normalized_source}:{normalized_relay_event_id}"
            )

        pool = get_db_pool()
        result: dict[str, Any] = {}
        interaction_result: Optional[dict[str, Any]] = None
        async with _transaction_connection(pool) as conn:
            lifecycle_ready = await conn.fetchval(
                """
                SELECT to_regclass('eom_lead_lifecycle_events') IS NOT NULL
                   AND to_regclass('eom_inbound_delivery_receipts') IS NOT NULL
                   AND EXISTS (
                       SELECT 1
                       FROM pg_trigger
                       WHERE tgrelid = 'contacts'::regclass
                         AND tgname = 'trg_record_eom_lead_created'
                         AND NOT tgisinternal
                         AND tgenabled IN ('O', 'A')
                   )
                """
            )
            if not lifecycle_ready:
                raise RuntimeError(
                    "EOM inbound lead ingress unavailable: lifecycle ledger or delivery receipts are not ready"
                )
            for lock_key in sorted(set(lock_keys)):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )

            delivery_receipt_exists = False
            if normalized_relay_event_id:
                # A trusted delivery receipt is an idempotency anchor, not a
                # mutable contact identity. Unlike contacts.source_ref, this
                # ledger can retain every later delivery for one known contact.
                existing = await conn.fetchrow(
                    """
                    SELECT c.*, receipt.interaction_id AS _receipt_interaction_id
                    FROM eom_inbound_delivery_receipts AS receipt
                    JOIN contacts AS c ON c.id = receipt.contact_id
                    WHERE receipt.source = $1
                      AND receipt.delivery_id = $2
                    FOR UPDATE OF receipt, c
                    """,
                    normalized_source,
                    normalized_relay_event_id,
                )
                if existing is not None:
                    result = dict(existing)
                    receipt_interaction_id = result.pop("_receipt_interaction_id", None)
                    result["_was_created"] = False
                    delivery_receipt_exists = True
                    if interaction is not None:
                        if receipt_interaction_id is not None:
                            prior_interaction = await conn.fetchrow(
                                "SELECT * FROM contact_interactions WHERE id = $1",
                                receipt_interaction_id,
                            )
                            if prior_interaction is None:
                                raise RuntimeError(
                                    "EOM inbound delivery receipt references a missing interaction"
                                )
                            interaction_result = dict(prior_interaction)
                            interaction_result["inserted"] = False
                        elif result.get("status") != "archived":
                            interaction_result = await _write_contact_interaction(
                                conn,
                                contact_id=str(result["id"]),
                                **interaction,
                            )
                            if not interaction_result.get("id"):
                                raise RuntimeError(
                                    "EOM inbound interaction insert returned no receipt ID"
                                )
                            await conn.execute(
                                """
                                UPDATE eom_inbound_delivery_receipts
                                   SET interaction_id = $3
                                 WHERE source = $1 AND delivery_id = $2
                                """,
                                normalized_source,
                                normalized_relay_event_id,
                                interaction_result["id"],
                            )
                        else:
                            # A committed delivery mapping may outlive the
                            # contact. Replaying it must not attach a fresh
                            # inbound event to that archived original.
                            interaction_result = {"inserted": False}
                if not result:
                    # Compatibility for rows created before the dedicated
                    # receipt table. New traffic never relies on this mutable
                    # contact provenance as its replay ledger.
                    existing = await conn.fetchrow(
                        """
                        SELECT * FROM contacts
                        WHERE business_context_id = $1
                          AND source = $2
                          AND source_ref = $3
                        ORDER BY id ASC
                        LIMIT 1
                        FOR UPDATE
                        """,
                        EOM_BUSINESS_CONTEXT_ID,
                        normalized_source,
                        normalized_relay_event_id,
                    )
                    if existing is not None:
                        result = dict(existing)
                        result["_was_created"] = False

            async def _find(context: Optional[str], *, channel: str, value: str):
                if context is None:
                    if channel == "phone":
                        return await conn.fetchrow(
                            """
                            SELECT * FROM contacts
                            WHERE business_context_id IS NULL
                              AND status != 'archived'
                              AND RIGHT(REGEXP_REPLACE(COALESCE(phone, ''), '[^0-9]', '', 'g'), 10)
                                  = RIGHT($1, 10)
                            ORDER BY updated_at DESC, id ASC
                            LIMIT 1
                            FOR UPDATE
                            """,
                            value,
                        )
                    return await conn.fetchrow(
                        """
                        SELECT * FROM contacts
                        WHERE business_context_id IS NULL
                          AND status != 'archived'
                          AND LOWER(email) = $1
                        ORDER BY updated_at DESC, id ASC
                        LIMIT 1
                        FOR UPDATE
                        """,
                        value,
                    )
                if channel == "phone":
                    return await conn.fetchrow(
                        f"""
                        SELECT * FROM contacts
                        WHERE business_context_id = $1
                          AND status != 'archived'
                          AND RIGHT(REGEXP_REPLACE(COALESCE(phone, ''), '[^0-9]', '', 'g'), 10)
                              = RIGHT($2, 10)
                        ORDER BY updated_at DESC, id ASC
                        LIMIT 1
                        FOR UPDATE
                        """,
                        context,
                        value,
                    )
                return await conn.fetchrow(
                    f"""
                    SELECT * FROM contacts
                    WHERE business_context_id = $1
                      AND status != 'archived'
                      AND LOWER(email) = $2
                    ORDER BY updated_at DESC, id ASC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    context,
                    value,
                )

            if not result:
                existing = None
                for channel, value in (("phone", phone_digits), ("email", normalized_email)):
                    if not value:
                        continue
                    for context in (EOM_BUSINESS_CONTEXT_ID, None):
                        existing = await _find(context, channel=channel, value=value)
                        if existing is not None:
                            result = dict(existing)
                            result["_was_created"] = False
                            break
                    if result:
                        break

            if not result:
                contact_id = str(uuid4())
                now = datetime.now(timezone.utc)
                row = await conn.fetchrow(
                    """
                    INSERT INTO contacts (
                        id, full_name, email, phone, address, business_context_id,
                        contact_type, status, tags, source, source_ref, lead_stage,
                        created_at, updated_at, metadata
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, 'lead', 'active', $7, $8, $9, 'new',
                        $10, $10, '{}'::jsonb
                    ) RETURNING *
                    """,
                    contact_id,
                    full_name.strip() or phone_digits or normalized_email or "Unknown",
                    normalized_email or None,
                    phone_digits or None,
                    address or None,
                    EOM_BUSINESS_CONTEXT_ID,
                    tags or [],
                    normalized_source or source,
                    (
                        normalized_relay_event_id
                        if identityless_relay
                        else normalized_source_ref or None
                    ),
                    now,
                )
                result = dict(row) if row else {}
                result["_was_created"] = True

            if interaction is not None and interaction_result is None:
                interaction_result = await _write_contact_interaction(
                    conn,
                    contact_id=str(result["id"]),
                    **interaction,
                )
                if not interaction_result.get("id"):
                    raise RuntimeError(
                        "EOM inbound interaction insert returned no receipt ID"
                    )

            if normalized_relay_event_id and not delivery_receipt_exists:
                await conn.execute(
                    """
                    INSERT INTO eom_inbound_delivery_receipts (
                        source, delivery_id, contact_id, interaction_id
                    ) VALUES ($1, $2, $3, $4)
                    """,
                    normalized_source,
                    normalized_relay_event_id,
                    result["id"],
                    interaction_result.get("id") if interaction_result else None,
                )

        # The prior find-or-create path emits contact-created reasoning events.
        # This happens only after the transaction commits, and its secondary
        # delivery must not turn a committed inbound lead into a failed intake.
        if result.get("_was_created"):
            try:
                await self._emit_contact_created(
                    result,
                    full_name=full_name,
                    email=email,
                    phone=phone,
                )
            except Exception:
                logger.warning(
                    "EOM inbound contact-created event failed after contact %s committed",
                    result.get("id", ""),
                    exc_info=True,
                )
        if interaction_result is not None:
            result["_inbound_interaction"] = interaction_result
            if interaction_result.get("inserted"):
                await self._emit_interaction_logged(
                    contact_id=str(result["id"]),
                    interaction=interaction_result,
                    interaction_type=str(interaction["interaction_type"]),
                    intent=interaction.get("intent"),
                    summary=str(interaction["summary"]),
                )
        return result

    async def create_contact(
        self,
        data: dict[str, Any],
        *,
        merge_existing: bool = True,
        preserve_existing: bool = False,
    ) -> dict[str, Any]:
        """
        Create a contact, returning an existing one if phone or email already matches.

        Dedup order: phone first (more unique), then email.  If a match is found the
        existing record is updated with any non-null fields from `data` so the caller
        always gets the most complete version.  This is application-level dedup;
        migration 037 should add a DB-level partial unique index for extra safety.

        ``merge_existing=False`` is the portal-reconciliation race seam: only a
        same-tenant email match is returned, without claiming or updating it.
        Phone is intentionally not used in that mode because the provider's
        substring matcher is weaker than the portal sync's normalized resolver.
        The default preserves every existing caller's claim-and-merge behavior.
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
                if not merge_existing:
                    return None
                claimable = await self.search_contacts(
                    business_context_id_is_null=True, **channel
                )
                return claimable[0] if claimable else None
            return _pick(await self.search_contacts(**channel))

        existing: Optional[dict[str, Any]] = None
        if phone and merge_existing:
            existing = await _resolve(phone=phone)
        if existing is None and email:
            existing = await _resolve(email=email)

        if (
            merge_existing
            and existing is not None
            and ctx
            and existing.get("business_context_id") is None
        ):
            # Claim the NULL-context legacy match by compare-and-set before
            # merging: a concurrent claim by another tenant leaves the row
            # theirs and this create falls through to a fresh insert instead
            # of overwriting their claim.
            claimed = await self.claim_contact(str(existing["id"]), ctx)
            if claimed is None:
                existing = None
            else:
                existing = claimed

        if existing is not None and not merge_existing:
            result = dict(existing)
            result["_was_created"] = False
            return result

        if existing is not None:
            from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

            existing_type = str(existing.get("contact_type") or "")
            incoming_type = str(data.get("contact_type") or "")
            # An EOM lead/customer type is a lifecycle decision, not inbound
            # enrichment.  Keep a matching EOM contact unchanged when a generic
            # creator asks for a different type; the later funnel transition
            # service is the sole promotion path.
            protected_eom_type = (
                ctx == EOM_BUSINESS_CONTEXT_ID
                and existing.get("business_context_id") == EOM_BUSINESS_CONTEXT_ID
                and existing_type
                and incoming_type
                and existing_type != incoming_type
            )
            if preserve_existing or protected_eom_type:
                result = dict(existing)
                result["_was_created"] = False
                return result
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
        preserve_existing: bool = False,
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
        result = await self.create_contact(data, preserve_existing=preserve_existing)

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

    async def get_contact(
        self,
        contact_id: str,
        business_context_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if business_context_id:
            row = await pool.fetchrow(
                """
                SELECT * FROM contacts
                WHERE id = $1
                  AND business_context_id = $2
                """,
                contact_id,
                business_context_id,
            )
        else:
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
        pool = self._get_pool()
        allowed = {
            "full_name", "first_name", "last_name", "email", "phone",
            "address", "city", "state", "zip", "contact_type", "status",
            "tags", "notes", "business_context_id", "source", "source_ref",
            "metadata", "lead_stage", "lead_owner", "next_follow_up_at",
        }
        updates = {k: v for k, v in data.items() if k in allowed}
        lifecycle_requested = bool({"contact_type", "lead_stage"} & updates.keys())
        ownership_requested = "business_context_id" in updates
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

        def _validate_eom_transition(existing: Any) -> None:
            if existing is None:
                return
            from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

            lifecycle_transition = any(
                field in updates and updates[field] != existing[field]
                for field in ("contact_type", "lead_stage")
            )
            eom_ownership_transition = (
                ownership_requested
                and updates["business_context_id"] != existing["business_context_id"]
                and (
                    existing["business_context_id"] == EOM_BUSINESS_CONTEXT_ID
                    or updates["business_context_id"] == EOM_BUSINESS_CONTEXT_ID
                )
            )
            if eom_ownership_transition:
                raise ValueError(
                    "EOM contact ownership changes require the funnel transition service"
                )
            if (
                lifecycle_transition
                and (
                    existing["business_context_id"] in (None, EOM_BUSINESS_CONTEXT_ID)
                    or updates.get("business_context_id") == EOM_BUSINESS_CONTEXT_ID
                )
            ):
                raise ValueError(
                    "EOM lead type and stage changes require the funnel transition service"
                )

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

        async def _write(executor: Any) -> Optional[dict[str, Any]]:
            row = await executor.fetchrow(
                f"UPDATE contacts SET {', '.join(set_parts)} WHERE {where} RETURNING *",
                *params,
            )
            return dict(row) if row else None

        if lifecycle_requested or ownership_requested:
            # This row lock is the ownership decision's linearization point:
            # validation and the permitted write share one transaction with
            # `claim_contact`'s compare-and-set UPDATE.
            async with _transaction_connection(pool) as conn:
                existing = await conn.fetchrow(
                    """
                    SELECT business_context_id, contact_type, lead_stage
                    FROM contacts
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    contact_id,
                )
                if existing is None:
                    return None
                _validate_eom_transition(existing)
                return await _write(conn)

        row = await _write(pool)
        return row

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

    async def open_customer_service_ticket(
        self,
        *,
        contact_id: str,
        business_context_id: str,
        summary: str,
        details: Optional[str] = None,
        priority: Optional[str] = None,
        assignee: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Atomically claim a visible contact and open its tenant ticket."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            WITH visible_contact AS (
                UPDATE contacts
                   SET business_context_id = $2,
                       updated_at = CASE
                           WHEN business_context_id IS NULL THEN NOW()
                           ELSE updated_at
                       END
                 WHERE id = $1
                   AND status != 'archived'
                   AND (
                       business_context_id IS NULL
                       OR business_context_id = $2
                   )
                 RETURNING id
            )
            INSERT INTO customer_service_tickets (
                contact_id,
                business_context_id,
                summary,
                details,
                priority,
                assignee
            )
            SELECT id, $2, $3, $4, $5, $6
            FROM visible_contact
            RETURNING *
            """,
            contact_id,
            business_context_id,
            summary,
            details,
            priority,
            assignee,
        )
        return dict(row) if row else None

    async def list_customer_service_tickets(
        self,
        *,
        business_context_id: str,
        status: Optional[str] = "open",
        contact_id: Optional[str] = None,
        priority: Optional[str] = None,
        assignee: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List one tenant's tickets with every filter applied before paging."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        conditions = ["business_context_id = $1"]
        params: list[Any] = [business_context_id]

        for column, value in (
            ("status", status),
            ("contact_id", contact_id),
            ("priority", priority),
            ("assignee", assignee),
        ):
            if value is not None:
                params.append(value)
                conditions.append(f"{column} = ${len(params)}")

        params.extend([limit, max(0, offset)])
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM customer_service_tickets
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC, id DESC
            LIMIT ${len(params) - 1} OFFSET ${len(params)}
            """,
            *params,
        )
        return [dict(row) for row in rows]

    async def update_customer_service_ticket(
        self,
        *,
        ticket_id: str,
        business_context_id: str,
        data: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Update mutable fields only while the tenant ticket is open."""
        from ..storage.database import get_db_pool

        allowed = {"summary", "details", "priority", "assignee"}
        updates = {key: value for key, value in data.items() if key in allowed}
        if not updates:
            return None

        params: list[Any] = [ticket_id, business_context_id]
        assignments: list[str] = []
        for key, value in updates.items():
            params.append(value)
            assignments.append(f"{key} = ${len(params)}")
        assignments.append("updated_at = NOW()")

        pool = get_db_pool()
        row = await pool.fetchrow(
            f"""
            UPDATE customer_service_tickets
               SET {', '.join(assignments)}
             WHERE id = $1
               AND business_context_id = $2
               AND status = 'open'
             RETURNING *
            """,
            *params,
        )
        return dict(row) if row else None

    async def close_customer_service_ticket(
        self,
        *,
        ticket_id: str,
        business_context_id: str,
        resolution: str,
    ) -> Optional[dict[str, Any]]:
        """Close once; retries return the original tenant-scoped resolution."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            WITH locked_ticket AS (
                SELECT *
                FROM customer_service_tickets
                WHERE id = $1
                  AND business_context_id = $2
                FOR UPDATE
            ),
            closed_now AS (
                UPDATE customer_service_tickets AS ticket
                   SET status = 'closed',
                       resolution = $3,
                       closed_at = NOW(),
                       updated_at = NOW()
                  FROM locked_ticket
                 WHERE ticket.id = locked_ticket.id
                   AND locked_ticket.status = 'open'
                 RETURNING ticket.*, false AS already_closed
            )
            SELECT * FROM closed_now
            UNION ALL
            SELECT locked_ticket.*, true AS already_closed
            FROM locked_ticket
            WHERE locked_ticket.status = 'closed'
              AND NOT EXISTS (SELECT 1 FROM closed_now)
            LIMIT 1
            """,
            ticket_id,
            business_context_id,
            resolution,
        )
        return dict(row) if row else None

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
        include_unclaimed_legacy: bool = True,
    ) -> list[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if business_context_id:
            contact_scope = (
                "(c.business_context_id = $2 OR c.business_context_id IS NULL)"
                if include_unclaimed_legacy
                else "c.business_context_id = $2"
            )
            # Atomic tenant predicate: the page only returns while the owning
            # contact remains visible under the caller's selected strict or
            # tenant-plus-legacy mode.
            rows = await pool.fetch(
                f"""
                SELECT ci.* FROM contact_interactions ci
                JOIN contacts c ON c.id = ci.contact_id
                WHERE ci.contact_id = $1
                  AND {contact_scope}
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

    async def update_contact_appointment_operations(
        self,
        *,
        contact_id: str,
        appointment_id: str,
        business_context_id: str,
        data: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Claim a visible contact and update one linked tenant appointment."""
        from ..storage.database import get_db_pool

        allowed = {
            "recurrence_interval",
            "recurrence_unit",
            "assigned_cleaner",
            "per_visit_price",
        }
        updates = {key: value for key, value in data.items() if key in allowed}
        if not updates:
            return None

        params: list[Any] = [contact_id, appointment_id, business_context_id]
        assignments: list[str] = []
        for key, value in updates.items():
            params.append(value)
            assignments.append(f"{key} = ${len(params)}")
        assignments.append("updated_at = NOW()")

        pool = get_db_pool()
        row = await pool.fetchrow(
            f"""
            WITH target_appointment AS (
                SELECT id, contact_id
                FROM appointments
                WHERE id = $2
                  AND contact_id = $1
                  AND business_context_id = $3
            ),
            visible_contact AS (
                UPDATE contacts AS contact
                   SET business_context_id = $3,
                       updated_at = CASE
                           WHEN contact.business_context_id IS NULL THEN NOW()
                           ELSE contact.updated_at
                       END
                  FROM target_appointment AS appointment
                 WHERE contact.id = appointment.contact_id
                   AND contact.status != 'archived'
                   AND (
                       contact.business_context_id IS NULL
                       OR contact.business_context_id = $3
                   )
                 RETURNING contact.id
            )
            UPDATE appointments AS appointment
               SET {', '.join(assignments)}
              FROM visible_contact AS contact
             WHERE appointment.id = $2
               AND appointment.contact_id = contact.id
               AND appointment.business_context_id = $3
             RETURNING appointment.*
            """,
            *params,
        )
        return dict(row) if row else None

    async def get_contact_appointments(
        self,
        contact_id: str,
        business_context_id: Optional[str] = None,
        include_unclaimed_legacy: bool = True,
    ) -> list[dict[str, Any]]:
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if business_context_id:
            contact_scope = (
                """
                (
                    contact.business_context_id IS NULL
                    OR contact.business_context_id = $2
                )
                """
                if include_unclaimed_legacy
                else "contact.business_context_id = $2"
            )
            rows = await pool.fetch(
                f"""
                SELECT appointment.id,
                       appointment.start_time,
                       appointment.end_time,
                       appointment.service_type,
                       appointment.status,
                       appointment.customer_name,
                       appointment.customer_phone,
                       appointment.customer_email,
                       appointment.customer_address,
                       appointment.notes,
                       appointment.created_at,
                       appointment.business_context_id,
                       appointment.recurrence_interval,
                       appointment.recurrence_unit,
                       appointment.assigned_cleaner,
                       appointment.per_visit_price
                FROM appointments AS appointment
                JOIN contacts AS contact
                  ON contact.id = appointment.contact_id
                WHERE appointment.contact_id = $1
                  AND appointment.business_context_id = $2
                  AND {contact_scope}
                ORDER BY appointment.start_time DESC
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
                       customer_address, notes, created_at, business_context_id,
                       recurrence_interval, recurrence_unit, assigned_cleaner,
                       per_visit_price
                FROM appointments
                WHERE contact_id = $1
                ORDER BY start_time DESC
                LIMIT 50
                """,
                contact_id,
            )
        return [dict(r) for r in rows]

    async def finalize_eom_customer_handoff(
        self,
        *,
        contact_id: str,
        tracker_customer_id: int,
        tracker_site_id: int,
        approval_key: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Atomically link a tracker Customer/Site and promote one EOM lead.

        The tracker commits its operational Customer and initial Site before
        calling this method.  Atlas only stores their opaque identifiers, so it
        cannot become a second owner for the estimate's rate or schedule.

        The admitted execution model is one PostgreSQL transaction per callback.
        Every callback takes transaction-scoped advisory locks for its approval
        key, contact, tracker Customer, and tracker Site in sorted order before
        reading any handoff row. That serializes duplicate external ownership
        and same-key callbacks without a lock-order cycle; after a winner
        commits, a waiting caller rereads and verifies the canonical completed
        transition or rejects the conflicting payload. The table's unique
        constraints remain the database backstop for callers that do not use
        this service method.
        """
        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        def _result(row: Any, *, idempotent: bool) -> dict[str, Any]:
            return {
                "handoff_id": str(row["id"]),
                "contact_id": str(row["contact_id"]),
                "tracker_customer_id": int(row["tracker_customer_id"]),
                "tracker_site_id": int(row["tracker_site_id"]),
                "approval_key": str(row["approval_key"]),
                "idempotent": idempotent,
            }

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            lock_keys = sorted(
                {
                    f"eom-customer-handoff:approval:{approval_key}",
                    f"eom-customer-handoff:contact:{contact_id}",
                    f"eom-customer-handoff:tracker-customer:{tracker_customer_id}",
                    f"eom-customer-handoff:tracker-site:{tracker_site_id}",
                }
            )
            for lock_key in lock_keys:
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            existing_key = await conn.fetchrow(
                """
                SELECT id, contact_id, approval_key, tracker_customer_id, tracker_site_id
                FROM eom_customer_handoffs
                WHERE approval_key = $1
                FOR UPDATE
                """,
                approval_key,
            )
            if existing_key is not None:
                if (
                    str(existing_key["contact_id"]) != str(contact_id)
                    or int(existing_key["tracker_customer_id"]) != tracker_customer_id
                    or int(existing_key["tracker_site_id"]) != tracker_site_id
                ):
                    raise EOMLeadConversionError(
                        409,
                        "Approval key already belongs to a different customer handoff",
                    )
                replay_contact = await conn.fetchrow(
                    """
                    SELECT business_context_id, contact_type, lead_stage, status
                    FROM contacts
                    WHERE id = $1
                    """,
                    existing_key["contact_id"],
                )
                replay_lifecycle_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM eom_lead_lifecycle_events
                        WHERE contact_id = $1
                          AND event_type = 'customer_approved'
                          AND source = 'eom_office'
                          AND operation_key = $2
                          AND metadata @> jsonb_build_object(
                              'tracker_customer_id', $3::bigint,
                              'tracker_site_id', $4::bigint
                          )
                    )
                    """,
                    existing_key["contact_id"],
                    existing_key["approval_key"],
                    existing_key["tracker_customer_id"],
                    existing_key["tracker_site_id"],
                )
                if (
                    replay_contact is None
                    or replay_contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
                    or replay_contact["contact_type"] != "customer"
                    or replay_contact["lead_stage"] is not None
                    or not replay_lifecycle_exists
                ):
                    raise EOMLeadConversionError(
                        409,
                        "Existing EOM customer handoff is not a completed finalization",
                    )
                return _result(existing_key, idempotent=True)

            contact = await conn.fetchrow(
                """
                SELECT id, business_context_id, contact_type, lead_stage, status
                FROM contacts
                WHERE id = $1
                FOR UPDATE
                """,
                contact_id,
            )
            if contact is None or contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID:
                raise EOMLeadConversionError(404, "EOM lead was not found")

            existing_contact = await conn.fetchrow(
                """
                SELECT id, contact_id, approval_key, tracker_customer_id, tracker_site_id
                FROM eom_customer_handoffs
                WHERE contact_id = $1
                FOR UPDATE
                """,
                contact_id,
            )
            if existing_contact is not None:
                raise EOMLeadConversionError(
                    409,
                    "EOM lead already has a different customer handoff",
                )
            existing_tracker_link = await conn.fetchrow(
                """
                SELECT id, contact_id, approval_key, tracker_customer_id, tracker_site_id
                FROM eom_customer_handoffs
                WHERE tracker_customer_id = $1 OR tracker_site_id = $2
                FOR UPDATE
                """,
                tracker_customer_id,
                tracker_site_id,
            )
            if existing_tracker_link is not None:
                raise EOMLeadConversionError(
                    409,
                    "Tracker Customer or Site already belongs to an EOM customer handoff",
                )
            if contact["status"] != "active":
                raise EOMLeadConversionError(409, "EOM lead must be active before approval")
            if contact["contact_type"] != "lead":
                raise EOMLeadConversionError(409, "EOM contact is not a lead")
            if contact["lead_stage"] != "new":
                raise EOMLeadConversionError(409, "EOM lead is not ready for approval")

            await conn.execute(
                "SELECT set_config('atlas.eom_customer_handoff_finalization', 'true', true)"
            )
            handoff = await conn.fetchrow(
                """
                INSERT INTO eom_customer_handoffs (
                    contact_id, approval_key, tracker_customer_id, tracker_site_id,
                    approved_by_employee_id, approved_by_name
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, contact_id, approval_key, tracker_customer_id, tracker_site_id
                """,
                contact_id,
                approval_key,
                tracker_customer_id,
                tracker_site_id,
                actor_id,
                actor_name,
            )
            updated = await conn.fetchrow(
                """
                UPDATE contacts
                SET contact_type = 'customer', lead_stage = NULL, updated_at = NOW()
                WHERE id = $1
                  AND business_context_id = $2
                  AND contact_type = 'lead'
                  AND lead_stage = 'new'
                  AND status = 'active'
                RETURNING id
                """,
                contact_id,
                EOM_BUSINESS_CONTEXT_ID,
            )
            if updated is None:
                raise RuntimeError("EOM lead changed during customer handoff finalization")
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, metadata
                )
                VALUES ($1, 'customer_approved', 'new', NULL, $2, 'eom_office', $3,
                        jsonb_build_object(
                            'tracker_customer_id', $4::bigint,
                            'tracker_site_id', $5::bigint,
                            'approved_by_employee_id', $6::bigint
                        ))
                """,
                contact_id,
                f"employee:{actor_id}:{actor_name}",
                approval_key,
                tracker_customer_id,
                tracker_site_id,
                actor_id,
            )
            return _result(handoff, idempotent=False)


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
