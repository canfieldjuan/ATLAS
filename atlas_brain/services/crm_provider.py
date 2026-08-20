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
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Collection, Mapping, Optional, Sequence
from uuid import UUID, uuid4

from .eom_public_onboarding_tokens import (
    build_eom_public_onboarding_link,
    eom_public_onboarding_hmac_key_fingerprint,
    format_eom_public_onboarding_token,
)

logger = logging.getLogger("atlas.services.crm_provider")

_INTERACTION_DEDUPE_SUMMARY_MAX_CHARS = 2000

# How old a 'sending' claim must be before operator reconciliation (revoke
# or confirm-sent) may act on it. An active approve-send holds the window
# between its transport POST and its confirmation far below this, so a
# fresh claim can never be recorded as revoked while the customer email may
# already be delivered. Well inside Resend's 24h idempotency-dedupe window,
# so an operator-driven retry after reconciliation stays duplicate-safe.
_EOM_ONBOARDING_SENDING_STALE_AFTER_MINUTES = 15
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
_STORED_PHONE_IDENTITY_DIGITS_SQL = (
    "REGEXP_REPLACE("
    "REGEXP_REPLACE("
    "COALESCE(phone, ''), "
    "'([0-9])[-[:space:],;#/()]*(extension|ext|x)\\.?[[:space:]]*[0-9]+[[:space:]]*$', "
    "'\\1', "
    "'i'"
    "), "
    "'[^0-9]', "
    "'', "
    "'g'"
    ")"
)
_STORED_EMAIL_IDENTITY_SQL = (
    "LOWER("
    "REGEXP_REPLACE("
    "COALESCE(email, ''), "
    "'(^[[:space:]]+|[[:space:]]+$)', "
    "'', "
    "'g'"
    ")"
    ")"
)


@dataclass(frozen=True)
class _EOMBookingFamily:
    """One EOM booking kind sharing the durable Calendar-boundary engine.

    The estimate and first-clean bookings run the exact same
    prepare -> Calendar -> complete lifecycle; a family carries only the
    constants that differ: ledger event names, the stages a lead may be in
    when the booking is requested, the stage a completed booking advances
    the lead to, and whether completion enqueues the onboarding email
    draft. Both families share the operation-key advisory-lock namespaces
    (operation keys are globally unique across contacts and families), so
    the customer-handoff execution fence covers every family with one
    probe per key.
    """

    label: str
    requested_event: str
    booked_event: str
    failed_event: str
    ambiguous_event: str
    admission_stages: tuple[str, ...]
    already_booked_stage: str
    target_stage: str
    summary_prefix: str
    enqueues_onboarding_draft: bool

    @property
    def event_types(self) -> tuple[str, ...]:
        return (
            self.requested_event,
            self.booked_event,
            self.failed_event,
            self.ambiguous_event,
        )

    @property
    def terminal_events(self) -> frozenset[str]:
        return frozenset(
            {self.booked_event, self.failed_event, self.ambiguous_event}
        )


_ESTIMATE_BOOKING_FAMILY = _EOMBookingFamily(
    label="estimate",
    requested_event="estimate_booking_requested",
    booked_event="estimate_booked",
    failed_event="estimate_booking_calendar_failed",
    ambiguous_event="estimate_booking_calendar_ambiguous",
    admission_stages=("new", "estimate_booked"),
    already_booked_stage="estimate_booked",
    target_stage="estimate_booked",
    summary_prefix="Estimate",
    enqueues_onboarding_draft=False,
)

_FIRST_CLEAN_BOOKING_FAMILY = _EOMBookingFamily(
    label="first clean",
    requested_event="first_clean_booking_requested",
    booked_event="first_clean_booked",
    failed_event="first_clean_booking_calendar_failed",
    ambiguous_event="first_clean_booking_calendar_ambiguous",
    admission_stages=("new", "estimate_booked", "won"),
    already_booked_stage="won",
    target_stage="won",
    summary_prefix="First clean",
    enqueues_onboarding_draft=True,
)

_EOM_BOOKING_FAMILIES = (_ESTIMATE_BOOKING_FAMILY, _FIRST_CLEAN_BOOKING_FAMILY)
_ALL_EOM_BOOKING_EVENT_TYPES = tuple(
    event for family in _EOM_BOOKING_FAMILIES for event in family.event_types
)
_ALL_EOM_BOOKED_EVENTS = frozenset(
    family.booked_event for family in _EOM_BOOKING_FAMILIES
)
_ALL_EOM_TERMINAL_EVENTS = frozenset(
    event for family in _EOM_BOOKING_FAMILIES for event in family.terminal_events
)
_ALL_EOM_AMBIGUOUS_EVENTS = frozenset(
    family.ambiguous_event for family in _EOM_BOOKING_FAMILIES
)
_ALL_EOM_REQUESTED_EVENTS = frozenset(
    family.requested_event for family in _EOM_BOOKING_FAMILIES
)
_EOM_LOST_RESTORABLE_STAGES = ("new", "estimate_booked")
_EOM_ACTIVE_LEAD_STAGES = ("new", "estimate_booked", "won")
_EOM_LOST_REPLAY_DISPOSITION_EVENTS = ("lead_lost", "lead_reopened")
_EOM_OPERATOR_CONTACT_SOURCES_METADATA_KEY = "eom_operator_contact_sources"
_EOM_WON_LOSS_CANCELLATION_REQUESTED_EVENT = "first_clean_cancellation_requested"
_EOM_WON_LOSS_CANCELLATION_UNSETTLED_EVENT = (
    "first_clean_cancellation_calendar_unsettled"
)
_EOM_WON_LOSS_CANCELLATION_COMPLETED_EVENT = "first_clean_cancelled"
_EOM_WON_LOSS_EVENT_TYPES = (
    _EOM_WON_LOSS_CANCELLATION_REQUESTED_EVENT,
    _EOM_WON_LOSS_CANCELLATION_UNSETTLED_EVENT,
    _EOM_WON_LOSS_CANCELLATION_COMPLETED_EVENT,
    "lead_lost",
)


def _eom_identity_lock_key(channel: str, value: str) -> str:
    return f"eom-contact-identity:{channel}:{value}"


def _eom_won_lead_loss_execution_lock_key(contact_id: str) -> str:
    """Return the one contact fence spanning a won-lead Calendar teardown."""

    return f"eom-won-lead-loss:execution:{contact_id}"


def _eom_customer_handoff_lock_keys(
    *,
    approval_key: str,
    contact_id: str,
    tracker_customer_id: int,
    tracker_site_id: int,
) -> list[str]:
    """One sorted lock vocabulary shared by office and public finalizers."""

    return sorted(
        {
            f"eom-customer-handoff:approval:{approval_key}",
            f"eom-customer-handoff:contact:{contact_id}",
            f"eom-customer-handoff:tracker-customer:{tracker_customer_id}",
            f"eom-customer-handoff:tracker-site:{tracker_site_id}",
            _eom_won_lead_loss_execution_lock_key(contact_id),
        }
    )


@asynccontextmanager
async def _transaction_connection(pool: Any, connection: Any | None = None):
    """Yield a transaction from Atlas' wrapper, asyncpg connection, or pool.

    Production uses ``DatabasePool.transaction``.  Supporting a raw asyncpg
    connection/pool keeps the migration integration proof on the real SQL path.
    A caller that already owns a transaction supplies its connection so related
    lifecycle writes remain one atomic operation rather than nesting a second
    transaction.
    """
    if connection is not None:
        yield connection
        return
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


async def _eom_disposition_replay_was_superseded(
    conn: Any,
    *,
    contact_id: str,
    replay_event_type: str,
    replay_event_id: Any,
    replay_lifecycle_sequence: Any,
) -> bool:
    """Return whether a lost/reopen replay row no longer owns the lead state."""
    if replay_lifecycle_sequence is not None:
        return bool(
            await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM eom_lead_lifecycle_events
                    WHERE contact_id = $1
                      AND event_type = ANY($2::varchar[])
                      AND lifecycle_sequence IS NOT NULL
                      AND lifecycle_sequence > $3
                )
                """,
                contact_id,
                list(_EOM_LOST_REPLAY_DISPOSITION_EVENTS),
                int(replay_lifecycle_sequence),
            )
        )
    if replay_event_type == "lead_reopened":
        legacy_replay = await conn.fetchrow(
            """
            SELECT
                COUNT(*) AS disposition_count,
                COUNT(*) FILTER (
                    WHERE event.id = $3
                      AND event.event_type = 'lead_reopened'
                      AND event.from_stage = 'lost'
                      AND event.to_stage = ANY($4::varchar[])
                      AND event.lifecycle_sequence IS NULL
                ) AS replay_reopen_count,
                COUNT(*) FILTER (
                    WHERE event.id <> $3
                      AND event.event_type = 'lead_lost'
                      AND event.to_stage = 'lost'
                      AND event.from_stage = replay.to_stage
                      AND event.lifecycle_sequence IS NULL
                ) AS legacy_loss_predecessor_count
            FROM eom_lead_lifecycle_events AS event
            LEFT JOIN eom_lead_lifecycle_events AS replay
              ON replay.id = $3
             AND replay.contact_id = $1
             AND replay.event_type = 'lead_reopened'
             AND replay.lifecycle_sequence IS NULL
            WHERE event.contact_id = $1
              AND event.event_type = ANY($2::varchar[])
            """,
            contact_id,
            list(_EOM_LOST_REPLAY_DISPOSITION_EVENTS),
            replay_event_id,
            list(_EOM_LOST_RESTORABLE_STAGES),
        )
        return not (
            legacy_replay is not None
            and int(legacy_replay["disposition_count"] or 0) == 2
            and int(legacy_replay["replay_reopen_count"] or 0) == 1
            and int(legacy_replay["legacy_loss_predecessor_count"] or 0) == 1
        )
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = ANY($2::varchar[])
                  AND id <> $3
            )
            """,
            contact_id,
            list(_EOM_LOST_REPLAY_DISPOSITION_EVENTS),
            replay_event_id,
        )
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
        if _normalize_interaction_text(key) and value is not None and str(value).strip()
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


# --------------------------------------------------------------------------
# Billing recipients (EOM Slice 1A)
# --------------------------------------------------------------------------

BILLING_RECIPIENT_PAGE_SIZE = 500
# Bounds the scan when eligible rows are sparse. 20 x 500 = 10k
# candidates, far beyond the real EOM contact count.
BILLING_RECIPIENT_MAX_PAGES = 20
BILLING_RECIPIENT_ELIGIBLE_STATUS = "active"

# Whitespace that makes an address unusable. btrim's one-argument form strips
# SPACES ONLY, so an email of "\t" survived it and was reported eligible --
# then handed to the mail path as a recipient. Every trim below names the set.
BILLING_RECIPIENT_BLANK_CHARS = " \t\r\n\x0b\x0c"

# Public reason tokens. `wrong_tenant` is deliberately absent: the lookup is
# tenant-scoped in SQL, so a contact belonging to another tenant is simply not
# found and this code never learns otherwise.
# Only an account may be invoiced. A lead, prospect or vendor with an address
# still satisfies "active with an email", and offering one as a billing
# recipient would let an invoice be addressed to a party that is not a customer.
# Verified against live data before encoding: every EOM contact that actually
# receives invoices is contact_type='customer'; all leads have zero.
BILLING_RECIPIENT_ELIGIBLE_CONTACT_TYPE = "customer"

# Deliberately permissive: one @, no whitespace either side, and a dot in the
# domain. Trimming alone let "not-an-address" through as a recipient, but a
# stricter grammar would reject deliverable real-world addresses -- checked
# against live data, this rejects none of the addresses currently invoiced.

BILLING_RECIPIENT_REASONS = ("not_found", "inactive", "no_email")


class EOMBillingDeliveryMethod(str, Enum):
    """Explicit EOM billing-delivery policy values owned by canonical CRM."""

    GMAIL_PDF = "gmail_pdf"
    MANUAL_SQUARE = "manual_square"
    NO_INVOICE_RESIDENTIAL_RECEIPT = "no_invoice_residential_receipt"


EOM_BILLING_DELIVERY_METHODS = frozenset(
    method.value for method in EOMBillingDeliveryMethod
)


def _require_eom_billing_delivery_method(value: Any) -> str:
    """Return one closed delivery policy value or reject an unsafe direct call."""

    if isinstance(value, EOMBillingDeliveryMethod):
        return value.value
    if isinstance(value, str) and value in EOM_BILLING_DELIVERY_METHODS:
        return value
    raise ValueError("delivery_method must be a supported EOM billing delivery method")


def _eom_billing_delivery_preference_projection(
    row: Any,
    *,
    changed: bool | None = None,
) -> dict[str, Any]:
    """Project one canonical preference without deriving a delivery fallback."""

    result = {
        "contactId": str(row["contact_id"]),
        "deliveryMethod": (
            str(row["delivery_method"])
            if row["delivery_method"] is not None
            else None
        ),
        "createdAt": row["created_at"],
        "createdBy": row["created_by"],
        "updatedAt": row["updated_at"],
        "updatedBy": row["updated_by"],
    }
    if changed is not None:
        result["changed"] = changed
    return result


def _billing_recipient_projection(row: Any, email: str) -> dict[str, Any]:
    """Build the five-field public object from an ALREADY-ELIGIBLE row.

    Eligibility is decided before this is called, never after. Constructing a
    full contact response and blanking fields on the way out would make "never
    a partial row" depend on every failure branch remembering to redact.

    ``email`` is passed in as the CANONICAL form rather than read off the row:
    validity and representation are one decision, and the stored column is not
    the answer to either. SQL ``btrim`` leaves Unicode edge whitespace and does
    not lowercase, so emitting the column would report
    ``\u00a0ap@example.com\u00a0`` eligible while returning an address nothing
    can send to.
    """
    return {
        "contactId": str(row["id"]),
        "displayName": row["full_name"],
        "email": email,
        "eligible": True,
        "reason": None,
    }


def _billing_recipient_refusal(contact_id: Any, reason: str) -> dict[str, Any]:
    """The only shape an ineligible answer may take: identity and cause only."""
    return {
        "contactId": str(contact_id),
        "displayName": None,
        "email": None,
        "eligible": False,
        "reason": reason,
    }


class DatabaseCRMProvider:
    """CRM provider -- queries the `contacts` table directly via asyncpg."""

    def __init__(self, *, pool: Any | None = None) -> None:
        """Use the configured pool, or a supplied transaction-capable adapter."""
        self._pool_override = pool

    def _get_pool(self) -> Any:
        pool_override = getattr(self, "_pool_override", None)
        if pool_override is not None:
            return pool_override
        from ..storage.database import get_db_pool

        return get_db_pool()

    @property
    def pool(self) -> Any:
        """The store this provider reads and writes.

        Evidence writers that describe rows owned by this provider (e.g.
        sent_emails history for an onboarding draft) must share this pool;
        the EOM funnel binds the provider to its own connection string, so
        the global pool may be a different database entirely.
        """
        return self._get_pool()

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
        from .eom_lead_ingress import (
            EOM_BUSINESS_CONTEXT_ID,
            normalise_eom_phone_digits,
        )

        normalized_email = str(email or "").strip().lower()
        phone_digits = normalise_eom_phone_digits(phone)
        if len(phone_digits) < 10:
            phone_digits = ""
        normalized_source = str(source or "").strip()
        normalized_source_ref = str(source_ref or "").strip()
        normalized_relay_event_id = str(relay_event_id or "").strip()
        identityless_relay = not phone_digits and not normalized_email
        if identityless_relay and not (normalized_source and normalized_relay_event_id):
            raise ValueError(
                "EOM inbound lead requires phone, email, or a stable relay event identity"
            )
        lock_keys = []
        if phone_digits:
            lock_keys.append(_eom_identity_lock_key("phone", phone_digits[-10:]))
        if normalized_email:
            lock_keys.append(_eom_identity_lock_key("email", normalized_email))
        if normalized_relay_event_id:
            lock_keys.append(
                f"eom-inbound:relay:{normalized_source}:{normalized_relay_event_id}"
            )

        pool = self._get_pool()
        result: dict[str, Any] = {}
        interaction_result: Optional[dict[str, Any]] = None
        async with _transaction_connection(pool) as conn:
            lifecycle_ready = await conn.fetchval("""
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
                """)
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
                        """
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
                    """
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
                for channel, value in (
                    ("phone", phone_digits),
                    ("email", normalized_email),
                ):
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

    async def _insert_contact_row(
        self,
        executor: Any,
        data: Mapping[str, Any],
        *,
        customer_type: str | None = None,
    ) -> dict[str, Any]:
        """Insert one contact through the provider-owned persistence site.

        ``customer_type`` is a KEYWORD, not a member of ``data``, and only the
        operator mutation path passes it. It drives billing shape, so a generic
        writer must not be able to set it by dropping a key into the dict: that
        would bypass ``_normalize_customer_type``, the authenticated funnel
        boundary, and the lifecycle event that records who changed it. Passing
        it inside ``data`` is a programming error and is refused outright
        rather than ignored, because silently dropping a caller's intent is how
        this field would end up wrong without anyone noticing.
        """
        if "customer_type" in data:
            raise ValueError(
                "customer_type must be set through the operator mutation "
                "boundary, not through a generic contact insert"
            )

        contact_id = str(uuid4())
        now = datetime.now(timezone.utc)
        raw_email = data.get("email")
        email = raw_email.lower() if raw_email else None
        metadata_json = json.dumps(data.get("metadata", {}))

        # customer_type is named ONLY when the caller supplied it. This helper is
        # shared by every contact writer -- the MCP create_contact tool and the
        # inbound find_or_create_contact among them -- and startup deliberately
        # continues after a failed migration. Naming the column unconditionally
        # would turn a pending migration 366 into UndefinedColumnError for
        # callers that never asked for the field, which is a regression they did
        # not sign up for. Omitting it lets the column default apply, exactly as
        # before this change; a caller that DOES pass it still fails loudly,
        # which is correct because its intent cannot be honoured.
        optional_columns: list[str] = []
        optional_values: list[Any] = []
        if customer_type is not None:
            optional_columns.append("customer_type")
            optional_values.append(customer_type)

        column_sql = ", ".join(optional_columns)
        if column_sql:
            column_sql = ", " + column_sql
        placeholder_sql = "".join(
            f",${index}" for index in range(21, 21 + len(optional_values))
        )
        metadata_index = 23 + len(optional_values)

        row = await executor.fetchrow(
            f"""
            INSERT INTO contacts (
                id, full_name, first_name, last_name, email, phone,
                address, city, state, zip, business_context_id,
                contact_type, status, tags, notes, source, source_ref,
                lead_stage, lead_owner, next_follow_up_at{column_sql},
                created_at, updated_at, metadata
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
                $18,$19,$20{placeholder_sql},
                ${metadata_index - 2},${metadata_index - 1},${metadata_index}::jsonb
            ) RETURNING *
            """,
            contact_id,
            data.get("full_name", ""),
            data.get("first_name"),
            data.get("last_name"),
            email,
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
            data.get("lead_stage"),
            data.get("lead_owner"),
            data.get("next_follow_up_at"),
            *optional_values,
            now,
            now,
            metadata_json,
        )
        result = dict(row) if row else {}
        result["_was_created"] = True
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
        # Refused HERE, at the public entry, not at the private insert.
        # create_contact returns through a dedup/merge branch that never reaches
        # _insert_contact_row, so a guard sited at the insert lets a matched
        # contact report success while the classification is silently dropped.
        # customer_type drives billing shape and belongs to the authenticated
        # operator boundary alone; refusing it at every door is the only
        # placement that holds for all of them.
        if "customer_type" in data:
            raise ValueError(
                "customer_type must be set through the operator mutation "
                "boundary, not through a generic contact create"
            )
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
                "full_name",
                "first_name",
                "last_name",
                "email",
                "phone",
                "address",
                "city",
                "state",
                "zip",
                "contact_type",
                "tags",
                "notes",
                "business_context_id",
                "source",
                "source_ref",
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
        pool = self._get_pool()
        return await self._insert_contact_row(pool, data)

    async def mutate_eom_operator_contact_atomic(self, *, command: Any) -> dict[str, Any]:
        """Create or edit one EOM contact under the operator mutation contract."""

        from .eom_crm_mutations import (
            EOM_OPERATOR_CONTACT_CREATED,
            EOM_OPERATOR_CONTACT_EVENT_TYPES,
            EOM_OPERATOR_CONTACT_TYPES,
            EOM_OPERATOR_CONTACT_UPDATED,
            EOM_BUSINESS_CONTEXT_ID,
            EOMOperatorContactMutationError,
        )

        def _metadata_from_row(value: Any) -> dict[str, Any]:
            if value is None:
                return {}
            loaded: Any = value
            if isinstance(value, str):
                try:
                    loaded = json.loads(value)
                except json.JSONDecodeError:
                    raise EOMOperatorContactMutationError(
                        409, "EOM operator contact metadata must be an object"
                    )
            if not isinstance(loaded, Mapping):
                raise EOMOperatorContactMutationError(
                    409, "EOM operator contact metadata must be an object"
                )
            return dict(loaded)

        def _event_result(
            *,
            row: Mapping[str, Any],
            event_type: str,
            idempotent: bool,
        ) -> dict[str, Any]:
            return {
                "contact_id": str(row["id"]),
                "operation": event_type,
                "idempotent": idempotent,
                "contact": dict(row),
            }

        def _dedupe_rows_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            by_id: dict[str, dict[str, Any]] = {}
            for row in rows:
                by_id.setdefault(str(row["id"]), row)
            return list(by_id.values())

        def _operator_provenance_metadata(
            target: Mapping[str, Any],
        ) -> tuple[dict[str, Any], bool]:
            metadata = _metadata_from_row(target.get("metadata"))
            source_record = {
                "source": command.contact_source,
                "source_channel": command.source_channel,
                "source_ref": command.source_ref,
            }
            if _EOM_OPERATOR_CONTACT_SOURCES_METADATA_KEY not in metadata:
                sources = {}
            else:
                raw_sources = metadata[_EOM_OPERATOR_CONTACT_SOURCES_METADATA_KEY]
                if not isinstance(raw_sources, Mapping):
                    raise EOMOperatorContactMutationError(
                        409,
                        "EOM operator contact provenance metadata must be an object",
                    )
                sources = {}
                for key, record in raw_sources.items():
                    if not isinstance(key, str) or not isinstance(record, Mapping):
                        raise EOMOperatorContactMutationError(
                            409,
                            "EOM operator contact provenance metadata must be an object",
                        )
                    if set(record) != {"source", "source_channel", "source_ref"}:
                        raise EOMOperatorContactMutationError(
                            409,
                            "EOM operator contact provenance metadata must be an object",
                        )
                    if not all(isinstance(record[field], str) for field in record):
                        raise EOMOperatorContactMutationError(
                            409,
                            "EOM operator contact provenance metadata must be an object",
                        )
                    sources[key] = dict(record)
            if sources.get(command.contact_source_ref) == source_record:
                return metadata, False
            sources[command.contact_source_ref] = source_record
            metadata[_EOM_OPERATOR_CONTACT_SOURCES_METADATA_KEY] = sources
            return metadata, True

        async def _select_exact_matches(
            conn: Any,
            *,
            include_contact_id: Any | None = None,
        ) -> list[dict[str, Any]]:
            phone = command.fields.get("phone")
            email = command.fields.get("email")
            rows = await conn.fetch(
                f"""
                WITH candidate_ids AS (
                    SELECT id
                    FROM contacts
                    WHERE $4::uuid IS NOT NULL
                      AND id = $4::uuid
                      AND (business_context_id = $1 OR business_context_id IS NULL)
                    UNION
                    SELECT id
                    FROM contacts
                    WHERE business_context_id = $1
                      AND source = $2
                      AND source_ref = $3
                      AND status != 'archived'
                    UNION
                    SELECT id
                    FROM contacts
                    WHERE business_context_id = $1
                      AND status != 'archived'
                      AND jsonb_typeof(metadata -> $5) = 'object'
                      AND metadata -> $5 -> $3 = jsonb_build_object(
                          'source', $2::text,
                          'source_channel', $8::text,
                          'source_ref', $9::text
                      )
                    UNION
                    SELECT id
                    FROM contacts
                    WHERE $6::text IS NOT NULL
                      AND (business_context_id = $1 OR business_context_id IS NULL)
                      AND status != 'archived'
                      AND RIGHT({_STORED_PHONE_IDENTITY_DIGITS_SQL}, 10)
                          = RIGHT($6::text, 10)
                    UNION
                    SELECT id
                    FROM contacts
                    WHERE $7::text IS NOT NULL
                      AND (business_context_id = $1 OR business_context_id IS NULL)
                      AND status != 'archived'
                      AND {_STORED_EMAIL_IDENTITY_SQL} = $7::text
                )
                SELECT contacts.*
                FROM contacts
                JOIN candidate_ids ON candidate_ids.id = contacts.id
                ORDER BY contacts.id
                FOR UPDATE
                """,
                EOM_BUSINESS_CONTEXT_ID,
                command.contact_source,
                command.contact_source_ref,
                include_contact_id,
                _EOM_OPERATOR_CONTACT_SOURCES_METADATA_KEY,
                phone,
                email,
                command.source_channel,
                command.source_ref,
            )
            return [dict(row) for row in rows]

        async def _reject_malformed_source_provenance(conn: Any) -> None:
            malformed = await conn.fetchrow(
                """
                SELECT id
                FROM contacts
                WHERE business_context_id = $1
                  AND status != 'archived'
                  AND metadata ? $2
                  AND (
                      jsonb_typeof(metadata -> $2) != 'object'
                      OR (
                          jsonb_typeof(metadata -> $2) = 'object'
                          AND (metadata -> $2) ? $3
                          AND metadata -> $2 -> $3 != jsonb_build_object(
                              'source', $4::text,
                              'source_channel', $5::text,
                              'source_ref', $6::text
                          )
                      )
                  )
                LIMIT 1
                """,
                EOM_BUSINESS_CONTEXT_ID,
                _EOM_OPERATOR_CONTACT_SOURCES_METADATA_KEY,
                command.contact_source_ref,
                command.contact_source,
                command.source_channel,
                command.source_ref,
            )
            if malformed is not None:
                raise EOMOperatorContactMutationError(
                    409, "EOM operator contact provenance metadata must be an object"
                )

        async def _resolve_target(conn: Any) -> dict[str, Any] | None:
            await _reject_malformed_source_provenance(conn)
            matches = _dedupe_rows_by_id(
                await _select_exact_matches(
                    conn,
                    include_contact_id=command.contact_id,
                )
            )
            if command.contact_id:
                target_id = str(command.contact_id)
                row = next(
                    (
                        candidate
                        for candidate in matches
                        if str(candidate["id"]) == target_id
                    ),
                    None,
                )
                if row is None:
                    raise EOMOperatorContactMutationError(
                        404, "EOM contact was not found"
                    )
                if row.get("status") == "archived":
                    raise EOMOperatorContactMutationError(
                        409, "Archived EOM contacts cannot be edited"
                    )
                conflicts = [
                    candidate
                    for candidate in matches
                    if str(candidate["id"]) != target_id
                ]
                if conflicts:
                    raise EOMOperatorContactMutationError(
                        409, "Operator contact identity belongs to another contact"
                    )
                return row

            if len(matches) > 1:
                raise EOMOperatorContactMutationError(
                    409, "Operator contact identity matched multiple contacts"
                )
            return matches[0] if matches else None

        def _assert_contact_type_matches(target: Mapping[str, Any]) -> None:
            stored_type = str(target.get("contact_type") or "")
            if stored_type not in EOM_OPERATOR_CONTACT_TYPES:
                raise EOMOperatorContactMutationError(
                    409,
                    "EOM operator contact updates require an existing lead or customer",
                )
            if (
                stored_type == "lead"
                and target.get("lead_stage") not in _EOM_ACTIVE_LEAD_STAGES
            ):
                raise EOMOperatorContactMutationError(
                    409, "EOM operator lead updates require a supported lead stage"
                )
            if stored_type == "lead" and target.get("status") != "active":
                raise EOMOperatorContactMutationError(
                    409, "EOM operator lead updates require an active lead"
                )
            if command.contact_type is None:
                return
            if stored_type != command.contact_type:
                raise EOMOperatorContactMutationError(
                    409,
                    "contactType changes require the EOM lifecycle transition service",
                )

        async def _update_existing(
            conn: Any, target: Mapping[str, Any]
        ) -> dict[str, Any]:
            _assert_contact_type_matches(target)
            updates = {
                key: value
                for key, value in command.fields.items()
                if target.get(key) != value
            }
            if target.get("business_context_id") is None:
                updates["business_context_id"] = EOM_BUSINESS_CONTEXT_ID
            provenance_metadata, provenance_changed = _operator_provenance_metadata(target)
            if provenance_changed:
                updates["metadata"] = provenance_metadata
            if not updates:
                return dict(target)
            updates["updated_at"] = datetime.now(timezone.utc)
            params: list[Any] = [target["id"]]
            set_parts: list[str] = []
            for index, (key, value) in enumerate(updates.items(), start=2):
                cast = "::jsonb" if key == "metadata" else ""
                set_parts.append(f"{key} = ${index}{cast}")
                params.append(
                    json.dumps(value, sort_keys=True) if key == "metadata" else value
                )
            context_index = len(params) + 1
            row = await conn.fetchrow(
                f"""
                UPDATE contacts
                SET {', '.join(set_parts)}
                WHERE id = $1
                  AND (
                      business_context_id = ${context_index}
                      OR business_context_id IS NULL
                  )
                RETURNING *
                """,
                *params,
                EOM_BUSINESS_CONTEXT_ID,
            )
            if row is None:
                raise RuntimeError("EOM operator contact update lost its target row")
            return dict(row)

        async def _create_contact(conn: Any) -> dict[str, Any]:
            full_name = command.fields.get("full_name")
            if not full_name:
                raise EOMOperatorContactMutationError(
                    422, "fullName is required when no existing contact matches"
                )
            contact_type = command.contact_type or "customer"
            fields = dict(command.fields)
            # Lifted out of the dict on purpose -- see _insert_contact_row.
            operator_customer_type = fields.pop("customer_type", None)
            data = {
                **fields,
                "full_name": full_name,
                "business_context_id": EOM_BUSINESS_CONTEXT_ID,
                "contact_type": contact_type,
                "status": "active",
                "source": command.contact_source,
                "source_ref": command.contact_source_ref,
                "lead_stage": "new" if contact_type == "lead" else None,
                "metadata": _operator_provenance_metadata({})[0],
            }
            return await self._insert_contact_row(
                conn, data, customer_type=operator_customer_type
            )

        async def _write_lifecycle_event(
            conn: Any,
            *,
            contact: Mapping[str, Any],
            event_type: str,
            previous: Mapping[str, Any] | None,
        ) -> None:
            metadata = {
                **command.lifecycle_metadata,
                "contact_type": contact.get("contact_type"),
            }
            if previous is not None:
                changed = sorted(
                    key
                    for key in command.fields
                    if previous.get(key) != contact.get(key)
                )
                metadata["changed_fields"] = changed
                # The value that was overwritten, not just its field name.
                #
                # This boundary is create-OR-return: an operator create can
                # resolve to an existing contact matched on phone or email and
                # then overwrite its identity as operator intent. Recording only
                # which fields moved makes that irreversible and unreviewable --
                # there is no contact history table, so the prior value exists
                # nowhere else the moment the UPDATE commits.
                #
                # Observed live on 2026-08-08: an office customer create matched
                # a calendar_import contact by phone and rewrote its full_name,
                # and the previous name could not be recovered from anything.
                metadata["previous_values"] = {
                    key: previous.get(key) for key in changed
                }
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, metadata
                )
                VALUES ($1, $2, $3, $4, $5, 'eom_office', $6, $7::jsonb)
                """,
                contact["id"],
                event_type,
                previous.get("lead_stage") if previous is not None else None,
                contact.get("lead_stage"),
                f"employee:{command.actor_id}:{command.actor_name}",
                command.operation_key,
                json.dumps(metadata, sort_keys=True),
            )

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            lock_keys = {
                f"eom-operator-contact:operation:{command.operation_key}",
            }
            if command.contact_id:
                lock_keys.add(f"eom-operator-contact:contact:{command.contact_id}")
            phone = command.fields.get("phone")
            if phone:
                lock_keys.add(_eom_identity_lock_key("phone", phone[-10:]))
            email = command.fields.get("email")
            if email:
                lock_keys.add(_eom_identity_lock_key("email", email))
            lock_keys.add(
                f"eom-operator-contact:source:{command.contact_source_ref}"
            )
            for lock_key in sorted(lock_keys):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )

            existing_events = await conn.fetch(
                """
                SELECT contact_id, event_type, metadata
                FROM eom_lead_lifecycle_events
                WHERE operation_key = $1
                  AND event_type = ANY($2::varchar[])
                FOR UPDATE
                """,
                command.operation_key,
                list(EOM_OPERATOR_CONTACT_EVENT_TYPES),
            )
            if len(existing_events) > 1:
                raise EOMOperatorContactMutationError(
                    409, "Operator contact key has multiple receipts"
                )
            if existing_events:
                event = existing_events[0]
                metadata = _metadata_from_row(event["metadata"])
                if metadata.get("request_fingerprint") != command.request_fingerprint:
                    raise EOMOperatorContactMutationError(
                        409,
                        "Idempotency-Key already belongs to a different contact mutation",
                    )
                contact = await conn.fetchrow(
                    "SELECT * FROM contacts WHERE id = $1",
                    event["contact_id"],
                )
                if contact is None:
                    raise EOMOperatorContactMutationError(
                        409, "Operator contact receipt has no contact"
                    )
                return _event_result(
                    row=contact,
                    event_type=str(event["event_type"]),
                    idempotent=True,
                )

            target = await _resolve_target(conn)
            if target is None:
                contact = await _create_contact(conn)
                event_type = EOM_OPERATOR_CONTACT_CREATED
                previous = None
            else:
                previous = dict(target)
                contact = await _update_existing(conn, target)
                event_type = EOM_OPERATOR_CONTACT_UPDATED
            await _write_lifecycle_event(
                conn,
                contact=contact,
                event_type=event_type,
                previous=previous,
            )
            return _event_result(
                row=contact,
                event_type=event_type,
                idempotent=False,
            )

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
        if "customer_type" in extra:
            raise ValueError(
                "customer_type must be set through the operator mutation "
                "boundary, not through find_or_create_contact"
            )
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
        pool = self._get_pool()
        conditions: list[str] = ["status != 'archived'"]
        params: list[Any] = []
        idx = 1

        if phone:
            from .eom_lead_ingress import normalise_eom_phone_digits

            digits = normalise_eom_phone_digits(phone)
            stored_digits = (
                "REGEXP_REPLACE(COALESCE(phone, ''), '[^0-9]', '', 'g')"
            )
            if not digits:
                conditions.append("FALSE")
            elif len(digits) < 10:
                conditions.append(f"{stored_digits} LIKE ${idx}")
                params.append(f"%{digits}%")
                idx += 1
            else:
                conditions.append(
                    f"({stored_digits} LIKE ${idx} "
                    f"OR {stored_digits} LIKE ${idx + 1} "
                    f"OR RIGHT({stored_digits}, 10) = RIGHT(${idx + 2}, 10))"
                )
                params.extend((f"%{digits}%", f"%{digits[-10:]}%", digits))
                idx += 3
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
            "full_name",
            "first_name",
            "last_name",
            "email",
            "phone",
            "address",
            "city",
            "state",
            "zip",
            "contact_type",
            "status",
            "tags",
            "notes",
            "business_context_id",
            "source",
            "source_ref",
            "metadata",
            "lead_stage",
            "lead_owner",
            "next_follow_up_at",
        }
        updates = {k: v for k, v in data.items() if k in allowed}
        lifecycle_requested = bool({"contact_type", "lead_stage"} & updates.keys())
        ownership_requested = "business_context_id" in updates
        status_requested = "status" in updates
        pipeline_requested = any(
            key in updates for key in ("lead_stage", "lead_owner", "next_follow_up_at")
        )
        if pipeline_requested:
            if "contact_type" in updates and updates["contact_type"] != "lead":
                raise ValueError("Lead pipeline fields require contact_type='lead'")
            if require_contact_type not in (None, "lead"):
                raise ValueError("Lead pipeline fields require contact_type='lead'")
            require_contact_type = "lead"
        if "email" in updates and updates["email"]:
            updates["email"] = updates["email"].lower()
        if "metadata" in updates:
            updates["metadata"] = (
                json.dumps(updates["metadata"])
                if isinstance(updates["metadata"], dict)
                else updates["metadata"]
            )
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
            if lifecycle_transition and (
                existing["business_context_id"] in (None, EOM_BUSINESS_CONTEXT_ID)
                or updates.get("business_context_id") == EOM_BUSINESS_CONTEXT_ID
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

        if lifecycle_requested or ownership_requested or status_requested:
            # Lifecycle/ownership validation and status fencing share the
            # permitted write's transaction. Ownership still uses the contact
            # row lock as its linearization point; status uses the won-loss
            # advisory lock before the mutation.
            async with _transaction_connection(pool) as conn:
                if status_requested:
                    # A generic status write can make won-loss completion fail
                    # after Calendar DELETE just as an archive can. The shared
                    # advisory lock and durable fence are owned here rather
                    # than at an MCP caller so every provider caller observes
                    # the same closed lifecycle boundary.
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                        _eom_won_lead_loss_execution_lock_key(contact_id),
                    )
                    await self._assert_eom_won_lead_loss_cancellation_fence(
                        conn, contact_id=contact_id
                    )
                if lifecycle_requested or ownership_requested:
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
        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            # Archive is a generic status mutation. It must share the won-loss
            # execution boundary so it cannot strand a prepared cancellation
            # after Calendar DELETE or an uncertain external result.
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                _eom_won_lead_loss_execution_lock_key(contact_id),
            )
            await self._assert_eom_won_lead_loss_cancellation_fence(
                conn, contact_id=contact_id
            )
            result = await conn.execute(
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
            raise ValueError("include_unclaimed_legacy requires business_context_id")
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

    async def list_eom_new_lead_review_items(
        self,
        *,
        limit: int = 100,
        cursor_created_at: datetime | None = None,
        cursor_contact_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Return the closed office-review projection for active EOM leads.

        This is intentionally separate from ``list_contacts``: the generic
        method returns complete CRM rows, while the office funnel boundary may
        expose only the small identity/readiness projection required to start
        the existing booking and customer-handoff commands.
        """
        cursor_clause = ""
        params: list[Any] = [limit]
        if cursor_created_at is not None and cursor_contact_id is not None:
            cursor_clause = "AND (c.created_at, c.id) < ($2::timestamptz, $3::uuid)"
            params.extend([cursor_created_at, cursor_contact_id])
        pool = self._get_pool()
        rows = await pool.fetch(
            f"""
            SELECT
                c.id AS contact_id,
                c.full_name,
                COALESCE(latest_intake.submitted_email, c.email) AS email,
                COALESCE(latest_intake.submitted_phone, c.phone) AS phone,
                c.address,
                c.source,
                c.lead_stage,
                c.created_at
            FROM contacts AS c
            LEFT JOIN LATERAL (
                SELECT
                    NULLIF(ci.metadata->>'submitted_email', '') AS submitted_email,
                    NULLIF(ci.metadata->>'submitted_phone', '') AS submitted_phone
                FROM contact_interactions AS ci
                WHERE ci.contact_id = c.id
                  AND ci.interaction_type = 'web_form'
                  AND ci.intent = 'estimate_request'
                  AND (
                      NULLIF(ci.metadata->>'submitted_email', '') IS NOT NULL
                      OR NULLIF(ci.metadata->>'submitted_phone', '') IS NOT NULL
                  )
                ORDER BY ci.occurred_at DESC, ci.created_at DESC, ci.id DESC
                LIMIT 1
            ) AS latest_intake ON TRUE
            WHERE c.business_context_id = 'effingham_maids'
              AND c.status = 'active'
              AND c.contact_type = 'lead'
              AND c.lead_stage IN ('new', 'estimate_booked', 'won')
              {cursor_clause}
            ORDER BY c.created_at DESC, c.id DESC
            LIMIT $1
            """,
            *params,
        )
        return [dict(row) for row in rows]

    async def list_billing_recipients(
        self,
        *,
        search: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """List EOM contacts assignable as an invoice recipient. Eligible only.

        Lives on the CRM provider, not the receivables service, because that is
        where canonical EOM contact reads live -- get_eom_funnel_crm_provider()
        pins this to the dedicated funnel CRM pool. ReceivablesService.pool
        resolves the separate global DSN, which in the deployed slim topology is
        a different database, so a contacts query there would read the wrong one.

        Discloses display name and email deliberately: an operator choosing
        where money-related mail goes has to see the address. It stays a narrow
        billing projection -- no phone, address, notes, tags, source, lifecycle.
        """
        from .eom_crm_mutations import normalize_contact_email
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        pool = self._get_pool()
        requested = max(1, min(limit, 500))
        base_conditions = [
            "business_context_id = $1",
            "status = $2",
            "contact_type = $4",
            "email IS NOT NULL",
        ]
        base_args: list[Any] = [
            EOM_BUSINESS_CONTEXT_ID,
            BILLING_RECIPIENT_ELIGIBLE_STATUS,
            BILLING_RECIPIENT_BLANK_CHARS,
            BILLING_RECIPIENT_ELIGIBLE_CONTACT_TYPE,
        ]
        if search and search.strip():
            base_args.append(f"%{search.strip()}%")
            base_conditions.append(
                f"(full_name ILIKE ${len(base_args)} OR email ILIKE ${len(base_args)})"
            )

        # Page until `requested` ELIGIBLE rows are found, not until `requested`
        # candidates have been read. The address grammar is decided in Python,
        # so a single SQL LIMIT lets rejected rows displace eligible ones
        # ordered after them -- 500 malformed rows followed by one valid
        # recipient would return an empty list for limit=1.
        #
        # Keyset on `id`, which is IMMUTABLE. An OFFSET can skip or repeat a
        # row under concurrent writes -- but so can a keyset built on a
        # MUTABLE column: `full_name` is editable through the operator contact
        # mutation, so renaming an unvisited row to sort before the cursor
        # drops it, and renaming a visited row to sort after it emits the
        # contact twice. Ordering for display is applied to the assembled
        # result instead, which is bounded by `requested`.
        results: list[dict[str, Any]] = []
        cursor_id: Any = None
        pages = 0
        while len(results) < requested and pages < BILLING_RECIPIENT_MAX_PAGES:
            pages += 1
            conditions = list(base_conditions)
            args = list(base_args)
            if cursor_id is not None:
                args.append(cursor_id)
                conditions.append(f"id > ${len(args)}")
            args.append(BILLING_RECIPIENT_PAGE_SIZE)
            rows = await pool.fetch(
                f"""
                SELECT id, full_name, btrim(email, $3) AS email
                FROM contacts
                WHERE {' AND '.join(conditions)}
                ORDER BY id
                LIMIT ${len(args)}
                """,
                *args,
            )
            if not rows:
                break
            for row in rows:
                canonical_email = normalize_contact_email(row["email"])
                if canonical_email is None:
                    continue
                results.append(
                    _billing_recipient_projection(row, canonical_email)
                )
                if len(results) == requested:
                    break
            cursor_id = rows[-1]["id"]
            if len(rows) < BILLING_RECIPIENT_PAGE_SIZE:
                break  # candidates exhausted

        if len(results) < requested and pages >= BILLING_RECIPIENT_MAX_PAGES:
            # Never truncate silently. Hitting this means the eligible rows are
            # sparser than the scan budget, which is a data-quality signal, not
            # a normal result.
            logger.warning(
                "billing recipients: scan cap reached after %d pages with "
                "%d/%d eligible found; result is truncated",
                pages,
                len(results),
                requested,
            )
        # Display order, applied once to the assembled page. The scan order is
        # an implementation detail of stable paging and is not what an operator
        # should see.
        results.sort(key=lambda item: (item["displayName"] or "", item["contactId"]))
        return results

    async def get_eom_payment_customer(
        self, contact_id: UUID
    ) -> dict[str, Any] | None:
        """Return the canonical active customer snapshot for an EOM payment.

        This is intentionally not the public billing-recipient projection.  A
        residential payment remains valid without an email address, so it must
        carry the active-customer identity and classified type while preserving
        a normalized email-or-``None`` outcome for the ledger receipt outbox.
        The query is tenant-scoped and admits only active account records.
        """
        from .eom_crm_mutations import normalize_contact_email
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        row = await self._get_pool().fetchrow(
            """
            SELECT id, full_name, customer_type,
                   btrim(COALESCE(email, ''), $3) AS email
            FROM contacts
            WHERE id = $1
              AND business_context_id = $2
              AND status = $4
              AND contact_type = $5
            """,
            contact_id,
            EOM_BUSINESS_CONTEXT_ID,
            BILLING_RECIPIENT_BLANK_CHARS,
            BILLING_RECIPIENT_ELIGIBLE_STATUS,
            BILLING_RECIPIENT_ELIGIBLE_CONTACT_TYPE,
        )
        if row is None:
            return None
        return {
            "contact_id": row["id"],
            "customer_name": str(row["full_name"] or "").strip() or "Customer",
            "customer_type": str(row["customer_type"] or "unknown"),
            "recipient_email": normalize_contact_email(row["email"]),
        }

    async def get_eom_billing_delivery_preference(
        self,
        contact_id: UUID,
    ) -> dict[str, Any] | None:
        """Return an active canonical customer's explicit delivery policy.

        A missing profile row is intentionally distinct from a missing customer:
        it returns the admitted contact with every preference field ``None``.
        Callers must keep the candidate blocked rather than deriving Gmail or
        Square from customer type, email, or historical invoice behavior.
        """
        preferences = await self.get_eom_billing_delivery_preferences((contact_id,))
        return preferences.get(contact_id)

    async def get_eom_billing_delivery_preferences(
        self,
        contact_ids: Collection[UUID],
    ) -> dict[UUID, dict[str, Any]]:
        """Return canonical delivery policies in one tenant-scoped read.

        A caller receives only active EOM customers. Every admitted customer is
        represented even when it has no profile row, so a missing map entry is
        always non-canonical/not-current evidence rather than an inferred
        delivery method. This keeps candidate preview reads bounded without
        relaxing the single-customer route's identity boundary.
        """
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        requested = tuple(sorted(set(contact_ids), key=str))
        if not requested:
            return {}
        rows = await self._get_pool().fetch(
            """
            SELECT c.id AS contact_id,
                   p.delivery_method,
                   p.created_at,
                   p.created_by,
                   p.updated_at,
                   p.updated_by
            FROM contacts AS c
            LEFT JOIN eom_billing_delivery_preferences AS p
              ON p.contact_id = c.id
            WHERE c.id = ANY($1::uuid[])
              AND c.business_context_id = $2
              AND c.status = $3
              AND c.contact_type = $4
            """,
            requested,
            EOM_BUSINESS_CONTEXT_ID,
            BILLING_RECIPIENT_ELIGIBLE_STATUS,
            BILLING_RECIPIENT_ELIGIBLE_CONTACT_TYPE,
        )
        return {
            row["contact_id"]: _eom_billing_delivery_preference_projection(row)
            for row in rows
        }

    async def set_eom_billing_delivery_preference(
        self,
        *,
        contact_id: UUID,
        delivery_method: str | EOMBillingDeliveryMethod,
        actor: str,
    ) -> dict[str, Any] | None:
        """Atomically set an explicit delivery policy for one active customer.

        The canonical customer row is the transaction's linearization point.
        Holding it `FOR UPDATE` makes the active-tenant admission decision and
        profile upsert one operation for all callers of the primary CRM pool.
        Equal-method retries deliberately return the existing evidence without
        modifying its actor or timestamp.
        """
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        selected_method = _require_eom_billing_delivery_method(delivery_method)
        if not isinstance(actor, str):
            raise ValueError("actor must be a non-empty string")
        normalized_actor = actor.strip()
        if not normalized_actor or len(normalized_actor) > 128:
            raise ValueError("actor must be a non-empty string of at most 128 characters")

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            customer = await conn.fetchrow(
                """
                SELECT id
                FROM contacts
                WHERE id = $1
                  AND business_context_id = $2
                  AND status = $3
                  AND contact_type = $4
                FOR UPDATE
                """,
                contact_id,
                EOM_BUSINESS_CONTEXT_ID,
                BILLING_RECIPIENT_ELIGIBLE_STATUS,
                BILLING_RECIPIENT_ELIGIBLE_CONTACT_TYPE,
            )
            if customer is None:
                return None

            existing = await conn.fetchrow(
                """
                SELECT contact_id, delivery_method, created_at, created_by,
                       updated_at, updated_by
                FROM eom_billing_delivery_preferences
                WHERE contact_id = $1
                """,
                contact_id,
            )
            if existing is not None and existing["delivery_method"] == selected_method:
                return _eom_billing_delivery_preference_projection(
                    existing,
                    changed=False,
                )

            if existing is None:
                updated = await conn.fetchrow(
                    """
                    INSERT INTO eom_billing_delivery_preferences (
                        contact_id, delivery_method, created_by, created_at,
                        updated_by, updated_at
                    )
                    VALUES (
                        $1, $2, $3, statement_timestamp(), $3, statement_timestamp()
                    )
                    RETURNING contact_id, delivery_method, created_at, created_by,
                              updated_at, updated_by
                    """,
                    contact_id,
                    selected_method,
                    normalized_actor,
                )
            else:
                updated = await conn.fetchrow(
                    """
                    UPDATE eom_billing_delivery_preferences
                    SET delivery_method = $2,
                        updated_by = $3,
                        updated_at = statement_timestamp()
                    WHERE contact_id = $1
                    RETURNING contact_id, delivery_method, created_at, created_by,
                              updated_at, updated_by
                    """,
                    contact_id,
                    selected_method,
                    normalized_actor,
                )
            if updated is None:
                raise RuntimeError("EOM billing delivery preference write was lost")
            return _eom_billing_delivery_preference_projection(updated, changed=True)

    async def billing_recipients_schema_ready(self) -> bool:
        """True when the billing-recipient and payment-customer reads work.

        An initialized pool only proves a connection opened. A configured but
        partially migrated database passes that check and then fails every
        recipient read with an undefined column, so readiness has to name the
        columns the queries depend on. LIMIT 0 validates them without reading
        a row.
        """
        try:
            await self._get_pool().fetch(
                """
                SELECT id, full_name, email, status, contact_type, customer_type,
                       business_context_id
                FROM contacts
                LIMIT 0
                """
            )
        except Exception:
            return False
        return True

    async def get_billing_recipient(self, contact_id: UUID) -> dict[str, Any]:
        """Answer whether ONE contact may receive invoices, and who it is.

        The authoritative validation used before storing a billing contact and
        again when resolving a recipient at invoice time -- eligibility can
        lapse between those moments, which is why the resolver re-asks rather
        than trusting a stored id.

        The tenant predicate is IN the query. A contact under another tenant
        returns no row and is reported not_found, identical to one that does not
        exist, so no later branch or log can betray the difference.
        """
        from .eom_crm_mutations import normalize_contact_email
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        pool = self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT id, full_name, status, contact_type,
                   btrim(COALESCE(email, ''), $3) AS email
            FROM contacts
            WHERE id = $1 AND business_context_id = $2
            """,
            contact_id,
            EOM_BUSINESS_CONTEXT_ID,
            BILLING_RECIPIENT_BLANK_CHARS,
        )
        if row is None:
            return _billing_recipient_refusal(contact_id, "not_found")
        # Allow-list the single live status rather than deny-listing the others:
        # the column carries no CHECK constraint, so enumerating would silently
        # admit any future value.
        if row["status"] != BILLING_RECIPIENT_ELIGIBLE_STATUS:
            return _billing_recipient_refusal(contact_id, "inactive")
        if row["contact_type"] != BILLING_RECIPIENT_ELIGIBLE_CONTACT_TYPE:
            # Not an account. Reported as `inactive` rather than a new public
            # token: from the caller's side it is the same answer -- this
            # contact may not receive invoices -- and the reason set stays the
            # four the contract declares.
            return _billing_recipient_refusal(contact_id, "inactive")
        # Nonblank is not usable, and usable is decided by the CANONICAL
        # grammar rather than a second expression of it. An SQL regex here
        # admitted `a@b..com` and `a@.b.com`, which the canonical validator
        # rejects -- so a caller could be offered a recipient the canonical
        # write path would refuse.
        canonical_email = normalize_contact_email(row["email"])
        if canonical_email is None:
            return _billing_recipient_refusal(contact_id, "no_email")
        return _billing_recipient_projection(row, canonical_email)

    async def list_known_eom_contact_ids(
        self,
        *,
        contact_ids: Sequence[UUID],
    ) -> list[dict[str, Any]]:
        """Return the subset of ``contact_ids`` that name a live EOM contact.

        Answers link verification for systems that store an Atlas contact id of
        their own. Tenant scope is part of the answer, not a filter applied
        afterwards: an id belonging to another business context is simply not
        in the result, so no caller can use this to probe outside EOM.

        Archived and lost contacts still count as known. The question is
        whether the link resolves, and a link to a contact that was closed is
        intact -- it is a dangling or cross-tenant id that means the write
        boundary was bypassed.

        Returns rows of ``{id, customer_type, customer_type_revision}`` rather than bare
        ids so a caller mirroring the classification can refresh it from the
        same tenant-scoped read (ATLAS #2357). The revision is assigned by the
        database on each customer_type change, so it lets that caller order the
        observed type evidence without depending on an application clock. The
        tenant predicate is unchanged and still part of the query, so a contact
        in another business context is absent here exactly as before -- it
        cannot leak a type it never returns an id for.
        """
        if not contact_ids:
            return []
        pool = self._get_pool()
        rows = await pool.fetch(
            """
            SELECT c.id, c.customer_type, c.customer_type_revision
            FROM contacts AS c
            WHERE c.business_context_id = 'effingham_maids'
              AND c.id = ANY($1::uuid[])
            """,
            list(contact_ids),
        )
        return [dict(row) for row in rows]

    @staticmethod
    def _eom_estimate_booking_metadata(
        *,
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        expected_calendar_event_id: str,
        requested_calendar_id: str | None = None,
        calendar_event: dict[str, Any] | None = None,
        calendar_event_id: str | None = None,
        actor_id: int | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "scheduled_start": scheduled_start.astimezone(timezone.utc).isoformat(),
            "scheduled_end": scheduled_end.astimezone(timezone.utc).isoformat(),
            "calendar_id": calendar_id,
            "notes": notes or "",
            "expected_calendar_event_id": expected_calendar_event_id,
        }
        if requested_calendar_id is not None:
            metadata["requested_calendar_id"] = requested_calendar_id
        if calendar_event is not None:
            metadata["calendar_event"] = calendar_event
        if calendar_event_id is not None:
            metadata["calendar_event_id"] = calendar_event_id
        if actor_id is not None:
            metadata["scheduled_by_employee_id"] = actor_id
        return metadata

    @staticmethod
    def _eom_booking_summary(
        family: _EOMBookingFamily, contact: Mapping[str, Any]
    ) -> str:
        name = str(contact.get("full_name") or "").strip() or "EOM lead"
        return f"{family.summary_prefix}: {name}"

    @staticmethod
    def _eom_estimate_booking_description(notes: str | None) -> str:
        parts = ["Scheduled from the private EOM lead funnel."]
        if notes:
            parts.append(notes)
        return "\n\n".join(parts)

    @staticmethod
    def _eom_booking_calendar_event(
        family: _EOMBookingFamily,
        *,
        contact: Mapping[str, Any],
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        expected_calendar_event_id: str,
    ) -> dict[str, Any]:
        return {
            "summary": DatabaseCRMProvider._eom_booking_summary(family, contact),
            "start": scheduled_start.astimezone(timezone.utc).isoformat(),
            "end": scheduled_end.astimezone(timezone.utc).isoformat(),
            "location": contact.get("address"),
            "description": DatabaseCRMProvider._eom_estimate_booking_description(notes),
            "calendar_id": calendar_id,
            "event_id": expected_calendar_event_id,
        }

    @staticmethod
    def _eom_estimate_booking_calendar_event_from_metadata(
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not metadata:
            return None
        event = metadata.get("calendar_event")
        if not isinstance(event, dict):
            return None
        return event

    @staticmethod
    def _eom_estimate_booking_metadata_from_row(
        metadata: Any,
    ) -> dict[str, Any]:
        if isinstance(metadata, Mapping):
            return dict(metadata)
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return {}

    @staticmethod
    def _eom_estimate_booking_payload_matches(
        metadata: dict[str, Any] | None,
        *,
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        expected_calendar_event_id: str,
    ) -> bool:
        if not metadata:
            return False
        expected = DatabaseCRMProvider._eom_estimate_booking_metadata(
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            calendar_id=calendar_id,
            notes=notes,
            expected_calendar_event_id=expected_calendar_event_id,
        )
        return all(metadata.get(key) == value for key, value in expected.items())

    @staticmethod
    def _eom_estimate_booking_operation_is_terminal(event_types: set[str]) -> bool:
        # One operation key only ever carries one family's events, so the
        # cross-family union is exact for per-operation terminality.
        return bool(event_types & _ALL_EOM_TERMINAL_EVENTS)

    @asynccontextmanager
    async def eom_estimate_booking_execution_lock(self, *, booking_key: str):
        """Serialize one booking key's external Calendar attempt.

        Shared by every EOM booking family (estimate and first clean):
        operation keys are globally unique across contacts and families, so
        one lock namespace covers all of them and the handoff fence needs
        only one probe per key. The historical "estimate" in the key prefix
        is kept so in-flight deployments and the merged handoff fence stay
        byte-compatible.

        The session advisory lock is held across the whole
        prepare -> Calendar -> complete span so that
        finalize_eom_customer_handoff can detect an in-flight same-key
        execution with pg_try_advisory_xact_lock: a terminal failed marker
        alone must not admit handoff while a concurrent same-key call could
        still produce a stronger (booked/ambiguous) outcome.

        Yields an execution-scoped provider bound to the same connection
        that holds the lock. The whole booking must run on that one pooled
        connection: reserving the lock on one connection while the
        lifecycle steps acquire a second would let max_size concurrent
        bookings exhaust the pool and deadlock behind their own locks.
        """
        from .eom_lead_conversion import EOMLeadConversionError

        lock_key = f"eom-estimate-booking:execution:{booking_key}"
        pool = self._get_pool()
        acquire = getattr(pool, "acquire", None)
        if callable(acquire):
            conn = await pool.acquire()
            release = pool.release
        else:
            # Repository test adapters expose query methods directly; the
            # lock then lives on their single session.
            conn = pool
            release = None
        acquired = False
        try:
            acquired = bool(
                await conn.fetchval(
                    "SELECT pg_try_advisory_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            )
            if not acquired:
                raise EOMLeadConversionError(
                    409,
                    "EOM estimate booking is already executing for this key",
                )
            yield DatabaseCRMProvider(pool=conn)
        finally:
            if acquired:
                await conn.fetchval(
                    "SELECT pg_advisory_unlock(hashtextextended($1, 0))",
                    lock_key,
                )
            if release is not None:
                await release(conn)

    @asynccontextmanager
    async def eom_won_lead_loss_execution_lock(self, *, contact_id: str):
        """Fence one won-lead teardown across its external Calendar DELETE.

        A database transaction cannot contain an external Google Calendar
        request.  This session advisory lock is therefore held across the
        prepare -> DELETE -> complete span.  Draft claims and both customer
        handoff paths take the same key transactionally before their first
        mutation, so they wait for the terminal decision instead of racing a
        welcome send or conversion through a cancellation window.

        The yielded provider is bound to the locking pooled connection.  That
        mirrors ``eom_estimate_booking_execution_lock``: using a second pooled
        connection for prepare/complete could deadlock a small pool against the
        session that owns the lock.
        """
        from .eom_lead_conversion import EOMLeadConversionError

        lock_key = _eom_won_lead_loss_execution_lock_key(contact_id)
        pool = self._get_pool()
        acquire = getattr(pool, "acquire", None)
        if callable(acquire):
            conn = await pool.acquire()
            release = pool.release
        else:
            conn = pool
            release = None
        acquired = False
        try:
            acquired = bool(
                await conn.fetchval(
                    "SELECT pg_try_advisory_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            )
            if not acquired:
                raise EOMLeadConversionError(
                    409, "EOM won lead loss is already executing for this lead"
                )
            yield DatabaseCRMProvider(pool=conn)
        finally:
            if acquired:
                await conn.fetchval(
                    "SELECT pg_advisory_unlock(hashtextextended($1, 0))",
                    lock_key,
                )
            if release is not None:
                await release(conn)

    async def prepare_eom_estimate_booking(
        self,
        *,
        contact_id: str,
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        booking_key: str,
        expected_calendar_event_id: str,
        actor_id: int,
        actor_name: str,
        calendar_id_explicit: bool = True,
    ) -> dict[str, Any]:
        """Claim one lead/booking key before the estimate Calendar write."""
        return await self._prepare_eom_booking(
            _ESTIMATE_BOOKING_FAMILY,
            contact_id=contact_id,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            calendar_id=calendar_id,
            notes=notes,
            booking_key=booking_key,
            expected_calendar_event_id=expected_calendar_event_id,
            actor_id=actor_id,
            actor_name=actor_name,
            calendar_id_explicit=calendar_id_explicit,
        )

    async def prepare_eom_first_clean_booking(
        self,
        *,
        contact_id: str,
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        booking_key: str,
        expected_calendar_event_id: str,
        actor_id: int,
        actor_name: str,
        calendar_id_explicit: bool = True,
        requested_calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """Claim one lead/booking key before the first-clean Calendar write."""
        return await self._prepare_eom_booking(
            _FIRST_CLEAN_BOOKING_FAMILY,
            contact_id=contact_id,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            calendar_id=calendar_id,
            notes=notes,
            booking_key=booking_key,
            expected_calendar_event_id=expected_calendar_event_id,
            actor_id=actor_id,
            actor_name=actor_name,
            calendar_id_explicit=calendar_id_explicit,
            requested_calendar_id=requested_calendar_id,
        )

    async def _prepare_eom_booking(
        self,
        family: _EOMBookingFamily,
        *,
        contact_id: str,
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        booking_key: str,
        expected_calendar_event_id: str,
        actor_id: int,
        actor_name: str,
        calendar_id_explicit: bool = True,
        requested_calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """Claim one lead/booking key before the external Calendar side effect.

        Row-lock order contract: the contact row is locked BEFORE any
        eom_lead_lifecycle_events rows, matching
        finalize_eom_customer_handoff (contact first, then lifecycle rows).
        Locking lifecycle rows first here deadlocks against a concurrent
        handoff finalization that already holds the contact row and is
        waiting on the same lifecycle rows (Postgres 40P01).
        """
        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            lock_keys = sorted(
                {
                    f"eom-estimate-booking:booking:{booking_key}",
                    f"eom-estimate-booking:contact:{contact_id}",
                }
            )
            for lock_key in lock_keys:
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )

            contact = await conn.fetchrow(
                """
                SELECT id, full_name, email, phone, address, business_context_id,
                       contact_type, lead_stage, status
                FROM contacts
                WHERE id = $1
                FOR UPDATE
                """,
                contact_id,
            )
            if (
                contact is None
                or contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
            ):
                raise EOMLeadConversionError(404, "EOM lead was not found")

            key_events = await conn.fetch(
                """
                SELECT contact_id, event_type, operation_key, metadata
                FROM eom_lead_lifecycle_events
                WHERE operation_key = $1
                  AND event_type = ANY($2::varchar[])
                FOR UPDATE
                """,
                booking_key,
                list(_ALL_EOM_BOOKING_EVENT_TYPES),
            )
            family_events = set(family.event_types)
            for event in key_events:
                if str(event["contact_id"]) != contact_id:
                    raise EOMLeadConversionError(
                        409,
                        "Booking key already belongs to a different EOM lead",
                    )
                if event["event_type"] not in family_events:
                    raise EOMLeadConversionError(
                        409,
                        "Booking key already belongs to a different EOM booking",
                    )

            events = await conn.fetch(
                """
                SELECT event_type, operation_key, metadata
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = ANY($2::varchar[])
                FOR UPDATE
                """,
                contact_id,
                list(_ALL_EOM_BOOKING_EVENT_TYPES),
            )
            request_for_key = None
            booked_for_key = None
            failed_for_key = None
            ambiguous_for_key = None
            operation_event_types: dict[str, set[str]] = {}
            for event in events:
                event_key = event["operation_key"]
                operation_event_types.setdefault(event_key, set()).add(
                    event["event_type"]
                )
                if event_key == booking_key:
                    if event["event_type"] == family.requested_event:
                        request_for_key = event
                    elif event["event_type"] == family.booked_event:
                        booked_for_key = event
                    elif event["event_type"] == family.failed_event:
                        failed_for_key = event
                    elif event["event_type"] == family.ambiguous_event:
                        ambiguous_for_key = event

            # Another operation blocks this one when it is unsettled in ANY
            # family (pending or ambiguous work must reconcile first), or
            # when it completed THIS family's booking (one estimate and one
            # first clean per lead). A completed booking in the other family
            # never blocks -- estimate booked -> first clean booked is the
            # funnel's normal path -- and a booked outcome dominates that
            # operation's own historical ambiguity/failed markers (the same
            # precedence ladder the completion writers and handoff enforce),
            # so a reconciled estimate with a stale ambiguous row cannot
            # permanently wedge the first clean.
            def _other_operation_blocks(event_types: set[str]) -> bool:
                if family.booked_event in event_types:
                    return True
                if event_types & _ALL_EOM_BOOKED_EVENTS:
                    return False
                if event_types & _ALL_EOM_AMBIGUOUS_EVENTS:
                    return True
                return bool(
                    event_types & _ALL_EOM_REQUESTED_EVENTS
                ) and not self._eom_estimate_booking_operation_is_terminal(
                    event_types
                )

            other_operation = next(
                (
                    operation_key
                    for operation_key, event_types in operation_event_types.items()
                    if operation_key != booking_key
                    and _other_operation_blocks(event_types)
                ),
                None,
            )
            if other_operation is not None:
                other_types = operation_event_types.get(other_operation, set())
                if other_types & family_events:
                    raise EOMLeadConversionError(
                        409,
                        f"EOM lead already has a different {family.label} booking",
                    )
                raise EOMLeadConversionError(
                    409,
                    "EOM lead has another booking operation in progress",
                )
            if request_for_key is not None:
                request_metadata = self._eom_estimate_booking_metadata_from_row(
                    request_for_key["metadata"]
                )
                if not calendar_id_explicit:
                    # The caller omitted calendar_id; a delayed same-key retry
                    # must replay against the persisted request snapshot, not
                    # whatever the configured default has drifted to since.
                    snapshot_calendar_id = str(
                        request_metadata.get("calendar_id") or ""
                    ).strip()
                    if snapshot_calendar_id:
                        calendar_id = snapshot_calendar_id
                elif (
                    booked_for_key is not None
                    and family.enqueues_onboarding_draft
                    and str(calendar_id).strip().casefold() == "primary"
                    and str(
                        request_metadata.get("requested_calendar_id") or ""
                    ).strip().casefold()
                    == "primary"
                ):
                    # A completed same-key replay is closed from immutable
                    # lifecycle metadata.  Do not make it depend on today's
                    # OAuth principal: ``primary`` is only the caller's
                    # original alias, while the booked request already holds
                    # the concrete Calendar target.  Normalizing it here keeps
                    # the existing full payload comparison, then the replay
                    # returns before the booking service can call Calendar.
                    snapshot_calendar_id = str(
                        request_metadata.get("calendar_id") or ""
                    ).strip()
                    if snapshot_calendar_id:
                        calendar_id = snapshot_calendar_id
                elif (
                    booked_for_key is not None
                    and family.enqueues_onboarding_draft
                    and str(calendar_id).strip().casefold() == "primary"
                    and str(
                        request_metadata.get("requested_calendar_id") or ""
                    ).strip()
                ):
                    # `primary` is an alias only when the immutable request
                    # snapshot proves the original call used it. A concrete
                    # original request must not accept a changed alias simply
                    # because today's primary happens to resolve to that ID.
                    raise EOMLeadConversionError(
                        409,
                        "Booking key already belongs to a different "
                        f"{family.label} booking",
                    )
                elif (
                    family.enqueues_onboarding_draft
                    and str(calendar_id).strip().casefold() == "primary"
                ):
                    # A caller that explicitly supplied ``primary`` needs the
                    # booking service to resolve the current principal before
                    # this payload can be compared to the stored concrete ID.
                    # This read-only preflight also keeps a replay from ever
                    # changing an existing booking's Calendar target.
                    return {
                        "contact_id": str(contact["id"]),
                        "lead_stage": str(contact["lead_stage"]),
                        "status": "calendar_identity_required",
                        "calendar_event_id": None,
                        "expected_calendar_event_id": expected_calendar_event_id,
                        "idempotent": True,
                        "requires_calendar_identity": True,
                        "contact": dict(contact),
                    }
                if not self._eom_estimate_booking_payload_matches(
                    request_metadata,
                    scheduled_start=scheduled_start,
                    scheduled_end=scheduled_end,
                    calendar_id=calendar_id,
                    notes=notes,
                    expected_calendar_event_id=expected_calendar_event_id,
                ):
                    raise EOMLeadConversionError(
                        409,
                        "Booking key already belongs to a different "
                        f"{family.label} booking",
                    )
                if booked_for_key is not None:
                    replay: dict[str, Any] = {
                        "contact_id": str(contact["id"]),
                        "lead_stage": family.target_stage,
                        "status": family.booked_event,
                        "calendar_event_id": expected_calendar_event_id,
                        "expected_calendar_event_id": expected_calendar_event_id,
                        "idempotent": True,
                        "contact": dict(contact),
                        "calendar_event": self._eom_estimate_booking_calendar_event_from_metadata(
                            request_metadata
                        ),
                    }
                    if family.enqueues_onboarding_draft:
                        replay["onboarding_draft_id"] = (
                            await self._eom_onboarding_draft_id_for_operation(
                                conn, booking_key
                            )
                        )
                    return replay
                if ambiguous_for_key is not None:
                    raise EOMLeadConversionError(
                        409,
                        f"EOM {family.label} booking requires calendar "
                        "reconciliation",
                    )
                if failed_for_key is not None:
                    raise EOMLeadConversionError(
                        409,
                        f"EOM {family.label} booking attempt failed; "
                        "use a new booking key",
                    )
                if contact["status"] != "active":
                    raise EOMLeadConversionError(
                        409, "EOM lead must be active before booking"
                    )
                if contact["contact_type"] != "lead":
                    raise EOMLeadConversionError(409, "EOM contact is not a lead")
                if contact["lead_stage"] not in family.admission_stages:
                    raise EOMLeadConversionError(
                        409,
                        f"EOM lead is not ready for {family.label} booking",
                    )
                return {
                    "contact_id": str(contact["id"]),
                    "lead_stage": str(contact["lead_stage"]),
                    "status": "calendar_pending",
                    "calendar_event_id": None,
                    "expected_calendar_event_id": expected_calendar_event_id,
                    "idempotent": True,
                    "contact": dict(contact),
                    "calendar_event": self._eom_estimate_booking_calendar_event_from_metadata(
                        request_metadata
                    ),
                }

            if contact["status"] != "active":
                raise EOMLeadConversionError(
                    409, "EOM lead must be active before booking"
                )
            if contact["contact_type"] != "lead":
                raise EOMLeadConversionError(409, "EOM contact is not a lead")
            if contact["lead_stage"] not in family.admission_stages:
                raise EOMLeadConversionError(
                    409,
                    f"EOM lead is not ready for {family.label} booking",
                )
            if contact["lead_stage"] == family.already_booked_stage:
                raise EOMLeadConversionError(
                    409,
                    f"EOM lead already has a different {family.label} booking",
                )
            if (
                family.enqueues_onboarding_draft
                and str(calendar_id).strip().casefold() == "primary"
            ):
                # Do not write an ambiguous alias into first-clean lifecycle
                # evidence. The booking service resolves it through Calendar
                # and calls prepare again with the concrete ID.
                return {
                    "contact_id": str(contact["id"]),
                    "lead_stage": str(contact["lead_stage"]),
                    "status": "calendar_identity_required",
                    "calendar_event_id": None,
                    "expected_calendar_event_id": expected_calendar_event_id,
                    "idempotent": False,
                    "requires_calendar_identity": True,
                    "contact": dict(contact),
                }

            calendar_event = self._eom_booking_calendar_event(
                family,
                contact=contact,
                scheduled_start=scheduled_start,
                scheduled_end=scheduled_end,
                calendar_id=calendar_id,
                notes=notes,
                expected_calendar_event_id=expected_calendar_event_id,
            )
            from_stage = str(contact["lead_stage"])
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, metadata
                )
                VALUES ($1, $5, $6, $6, $2, 'eom_office', $3, $4::jsonb)
                """,
                contact_id,
                f"employee:{actor_id}:{actor_name}",
                booking_key,
                json.dumps(
                    self._eom_estimate_booking_metadata(
                        scheduled_start=scheduled_start,
                        scheduled_end=scheduled_end,
                        calendar_id=calendar_id,
                        notes=notes,
                        expected_calendar_event_id=expected_calendar_event_id,
                        requested_calendar_id=(
                            requested_calendar_id
                            if family.enqueues_onboarding_draft
                            else None
                        ),
                        calendar_event=calendar_event,
                        actor_id=actor_id,
                    )
                ),
                family.requested_event,
                from_stage,
            )
            return {
                "contact_id": str(contact["id"]),
                "lead_stage": from_stage,
                "status": "calendar_pending",
                "calendar_event_id": None,
                "expected_calendar_event_id": expected_calendar_event_id,
                "idempotent": False,
                "contact": dict(contact),
                "calendar_event": calendar_event,
            }

    @staticmethod
    async def _eom_onboarding_draft_id_for_operation(
        conn: Any, operation_key: str
    ) -> str | None:
        row = await conn.fetchrow(
            "SELECT id FROM eom_onboarding_email_drafts WHERE operation_key = $1",
            operation_key,
        )
        return str(row["id"]) if row else None

    async def mark_eom_estimate_booking_calendar_ambiguous(
        self,
        *,
        contact_id: str,
        booking_key: str,
        expected_calendar_event_id: str,
        observed_calendar_event_id: str,
        actor_id: int,
        actor_name: str,
    ) -> None:
        await self._mark_eom_booking_calendar_ambiguous(
            _ESTIMATE_BOOKING_FAMILY,
            contact_id=contact_id,
            booking_key=booking_key,
            expected_calendar_event_id=expected_calendar_event_id,
            observed_calendar_event_id=observed_calendar_event_id,
            actor_id=actor_id,
            actor_name=actor_name,
        )

    async def mark_eom_first_clean_booking_calendar_ambiguous(
        self,
        *,
        contact_id: str,
        booking_key: str,
        expected_calendar_event_id: str,
        observed_calendar_event_id: str,
        actor_id: int,
        actor_name: str,
    ) -> None:
        await self._mark_eom_booking_calendar_ambiguous(
            _FIRST_CLEAN_BOOKING_FAMILY,
            contact_id=contact_id,
            booking_key=booking_key,
            expected_calendar_event_id=expected_calendar_event_id,
            observed_calendar_event_id=observed_calendar_event_id,
            actor_id=actor_id,
            actor_name=actor_name,
        )

    async def _mark_eom_booking_calendar_ambiguous(
        self,
        family: _EOMBookingFamily,
        *,
        contact_id: str,
        booking_key: str,
        expected_calendar_event_id: str,
        observed_calendar_event_id: str,
        actor_id: int,
        actor_name: str,
    ) -> None:
        """Record an unsafe Calendar response without promoting the lead."""
        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"eom-estimate-booking:booking:{booking_key}",
            )
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, metadata
                )
                SELECT c.id, $6::varchar,
                       c.lead_stage, c.lead_stage,
                       $2::varchar, 'eom_office', $3::varchar, jsonb_build_object(
                           'expected_calendar_event_id', $4::text,
                           'observed_calendar_event_id', $5::text
                       )
                FROM contacts c
                WHERE c.id = $1::uuid
                  AND NOT EXISTS (
                    SELECT 1
                    FROM eom_lead_lifecycle_events
                    WHERE contact_id = $1::uuid
                      AND operation_key = $3::varchar
                      AND event_type = $7::varchar
                )
                ON CONFLICT (contact_id, event_type, operation_key)
                    WHERE operation_key IS NOT NULL
                    DO NOTHING
                """,
                contact_id,
                f"employee:{actor_id}:{actor_name}",
                booking_key,
                expected_calendar_event_id,
                observed_calendar_event_id,
                family.ambiguous_event,
                family.booked_event,
            )

    async def mark_eom_estimate_booking_calendar_failed(
        self,
        *,
        contact_id: str,
        booking_key: str,
        expected_calendar_event_id: str,
        calendar_error: str | None,
        calendar_message: str,
        actor_id: int,
        actor_name: str,
    ) -> None:
        await self._mark_eom_booking_calendar_failed(
            _ESTIMATE_BOOKING_FAMILY,
            contact_id=contact_id,
            booking_key=booking_key,
            expected_calendar_event_id=expected_calendar_event_id,
            calendar_error=calendar_error,
            calendar_message=calendar_message,
            actor_id=actor_id,
            actor_name=actor_name,
        )

    async def mark_eom_first_clean_booking_calendar_failed(
        self,
        *,
        contact_id: str,
        booking_key: str,
        expected_calendar_event_id: str,
        calendar_error: str | None,
        calendar_message: str,
        actor_id: int,
        actor_name: str,
    ) -> None:
        await self._mark_eom_booking_calendar_failed(
            _FIRST_CLEAN_BOOKING_FAMILY,
            contact_id=contact_id,
            booking_key=booking_key,
            expected_calendar_event_id=expected_calendar_event_id,
            calendar_error=calendar_error,
            calendar_message=calendar_message,
            actor_id=actor_id,
            actor_name=actor_name,
        )

    async def _mark_eom_booking_calendar_failed(
        self,
        family: _EOMBookingFamily,
        *,
        contact_id: str,
        booking_key: str,
        expected_calendar_event_id: str,
        calendar_error: str | None,
        calendar_message: str,
        actor_id: int,
        actor_name: str,
    ) -> None:
        """Record a definitive Calendar failure as a terminal booking attempt."""
        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            lock_keys = sorted(
                {
                    f"eom-estimate-booking:booking:{booking_key}",
                    f"eom-estimate-booking:contact:{contact_id}",
                }
            )
            for lock_key in lock_keys:
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, metadata, reason
                )
                SELECT c.id, $7::varchar,
                       c.lead_stage, c.lead_stage,
                       $2::varchar, 'eom_office', $3::varchar, jsonb_build_object(
                            'expected_calendar_event_id', $4::text,
                            'calendar_error', $5::text,
                            'calendar_message', $6::text
                       ), $6::text
                FROM contacts c
                WHERE c.id = $1::uuid
                  AND NOT EXISTS (
                    SELECT 1
                    FROM eom_lead_lifecycle_events
                    WHERE contact_id = $1::uuid
                      AND operation_key = $3::varchar
                      AND event_type IN ($8::varchar, $9::varchar)
                )
                ON CONFLICT (contact_id, event_type, operation_key)
                    WHERE operation_key IS NOT NULL
                    DO NOTHING
                """,
                contact_id,
                f"employee:{actor_id}:{actor_name}",
                booking_key,
                expected_calendar_event_id,
                calendar_error,
                calendar_message,
                family.failed_event,
                family.booked_event,
                family.ambiguous_event,
            )

    async def complete_eom_estimate_booking(
        self,
        *,
        contact_id: str,
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        booking_key: str,
        expected_calendar_event_id: str,
        calendar_event_id: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        return await self._complete_eom_booking(
            _ESTIMATE_BOOKING_FAMILY,
            contact_id=contact_id,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            calendar_id=calendar_id,
            notes=notes,
            booking_key=booking_key,
            expected_calendar_event_id=expected_calendar_event_id,
            calendar_event_id=calendar_event_id,
            actor_id=actor_id,
            actor_name=actor_name,
        )

    async def complete_eom_first_clean_booking(
        self,
        *,
        contact_id: str,
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        booking_key: str,
        expected_calendar_event_id: str,
        calendar_event_id: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        return await self._complete_eom_booking(
            _FIRST_CLEAN_BOOKING_FAMILY,
            contact_id=contact_id,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            calendar_id=calendar_id,
            notes=notes,
            booking_key=booking_key,
            expected_calendar_event_id=expected_calendar_event_id,
            calendar_event_id=calendar_event_id,
            actor_id=actor_id,
            actor_name=actor_name,
        )

    async def _complete_eom_booking(
        self,
        family: _EOMBookingFamily,
        *,
        contact_id: str,
        scheduled_start: datetime,
        scheduled_end: datetime,
        calendar_id: str,
        notes: str | None,
        booking_key: str,
        expected_calendar_event_id: str,
        calendar_event_id: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Complete the lead-stage transition after Calendar returns the expected ID."""
        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        if calendar_event_id != expected_calendar_event_id:
            raise EOMLeadConversionError(
                409,
                f"Calendar event id does not match prepared {family.label} booking",
            )

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            lock_keys = sorted(
                {
                    f"eom-estimate-booking:booking:{booking_key}",
                    f"eom-estimate-booking:contact:{contact_id}",
                }
            )
            for lock_key in lock_keys:
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )

            contact = await conn.fetchrow(
                """
                SELECT id, full_name, email, phone, address, business_context_id,
                       contact_type, lead_stage, status
                FROM contacts
                WHERE id = $1
                FOR UPDATE
                """,
                contact_id,
            )
            if (
                contact is None
                or contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
            ):
                raise EOMLeadConversionError(404, "EOM lead was not found")
            # No status re-check here: admission was validated at prepare
            # time, and the Calendar event now exists. NocoDB holds an
            # UPDATE (status) grant, so an operator can archive the lead
            # while the Calendar call is in flight; refusing to record the
            # booked outcome would orphan a real appointment. Downstream
            # surfaces (review queue, customer handoff) apply their own
            # active-status admission.
            if contact["contact_type"] != "lead":
                raise EOMLeadConversionError(409, "EOM contact is not a lead")

            events = await conn.fetch(
                """
                SELECT event_type, operation_key, metadata
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = ANY($2::varchar[])
                FOR UPDATE
                """,
                contact_id,
                list(family.event_types),
            )
            request_for_key = None
            booked_for_key = None
            for event in events:
                if event["operation_key"] != booking_key:
                    continue
                if event["event_type"] == family.requested_event:
                    request_for_key = event
                elif event["event_type"] == family.booked_event:
                    booked_for_key = event
            if (
                request_for_key is None
                or not self._eom_estimate_booking_payload_matches(
                    self._eom_estimate_booking_metadata_from_row(
                        request_for_key["metadata"]
                    ),
                    scheduled_start=scheduled_start,
                    scheduled_end=scheduled_end,
                    calendar_id=calendar_id,
                    notes=notes,
                    expected_calendar_event_id=expected_calendar_event_id,
                )
            ):
                raise EOMLeadConversionError(
                    409,
                    f"EOM {family.label} booking was not prepared for this payload",
                )
            if booked_for_key is not None:
                replay: dict[str, Any] = {
                    "contact_id": str(contact["id"]),
                    "lead_stage": str(contact["lead_stage"]),
                    "status": family.booked_event,
                    "calendar_event_id": calendar_event_id,
                    "expected_calendar_event_id": expected_calendar_event_id,
                    "idempotent": True,
                }
                if family.enqueues_onboarding_draft:
                    replay["onboarding_draft_id"] = (
                        await self._eom_onboarding_draft_id_for_operation(
                            conn, booking_key
                        )
                    )
                return replay
            completion_stages = tuple(
                stage
                for stage in family.admission_stages
                if stage != family.already_booked_stage
            )
            if contact["lead_stage"] not in completion_stages:
                raise EOMLeadConversionError(
                    409, f"EOM lead is not ready for {family.label} booking"
                )
            from_stage = str(contact["lead_stage"])

            updated = await conn.fetchrow(
                """
                UPDATE contacts
                SET lead_stage = $3, updated_at = NOW()
                WHERE id = $1
                  AND business_context_id = $2
                  AND contact_type = 'lead'
                  AND lead_stage = ANY($4::varchar[])
                RETURNING id, lead_stage
                """,
                contact_id,
                EOM_BUSINESS_CONTEXT_ID,
                family.target_stage,
                list(completion_stages),
            )
            if updated is None:
                raise RuntimeError(
                    f"EOM lead changed during {family.label} booking"
                )
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, metadata
                )
                VALUES ($1, $5, $6, $7, $2, 'eom_office', $3, $4::jsonb)
                """,
                contact_id,
                f"employee:{actor_id}:{actor_name}",
                booking_key,
                json.dumps(
                    self._eom_estimate_booking_metadata(
                        scheduled_start=scheduled_start,
                        scheduled_end=scheduled_end,
                        calendar_id=calendar_id,
                        notes=notes,
                        expected_calendar_event_id=expected_calendar_event_id,
                        calendar_event_id=calendar_event_id,
                        actor_id=actor_id,
                    )
                ),
                family.booked_event,
                from_stage,
                family.target_stage,
            )
            result: dict[str, Any] = {
                "contact_id": str(contact["id"]),
                "lead_stage": str(updated["lead_stage"]),
                "status": family.booked_event,
                "calendar_event_id": calendar_event_id,
                "expected_calendar_event_id": expected_calendar_event_id,
                "idempotent": False,
            }
            if family.enqueues_onboarding_draft:
                result["onboarding_draft_id"] = (
                    await self._enqueue_eom_onboarding_email_draft(
                        conn, contact=contact, operation_key=booking_key
                    )
                )
            return result

    @staticmethod
    async def _enqueue_eom_onboarding_email_draft(
        conn: Any, *, contact: Mapping[str, Any], operation_key: str
    ) -> str | None:
        """Snapshot the onboarding email as one pending draft row.

        Runs in the same transaction as the won transition, so a booked
        first clean without a draft row is impossible. Nothing is sent
        here: the draft stays 'pending' until the approval surface claims
        it with the single-send contract documented in migration 360. A
        contact with no email is enqueued with blocker='no_email' rather
        than skipped, so the approval queue surfaces the gap. Replay is
        idempotent via UNIQUE(operation_key).

        The recipient resolves through the same latest-intake projection
        the office review queue shows: ingress deliberately leaves
        contacts.email unchanged when an existing contact re-submits with
        a new address (the new address lives in the web_form interaction
        metadata), so snapshotting contacts.email alone could target an
        obsolete inbox.
        """
        from ..templates.email import format_onboarding_welcome

        latest_intake_email = await conn.fetchval(
            """
            SELECT NULLIF(ci.metadata->>'submitted_email', '')
            FROM contact_interactions AS ci
            WHERE ci.contact_id = $1::uuid
              AND ci.interaction_type = 'web_form'
              AND ci.intent = 'estimate_request'
              AND (
                  NULLIF(ci.metadata->>'submitted_email', '') IS NOT NULL
                  OR NULLIF(ci.metadata->>'submitted_phone', '') IS NOT NULL
              )
            ORDER BY ci.occurred_at DESC, ci.created_at DESC, ci.id DESC
            LIMIT 1
            """,
            str(contact["id"]),
        )
        recipient = (
            str(latest_intake_email or "").strip()
            or str(contact.get("email") or "").strip()
            or None
        )
        subject, body = format_onboarding_welcome(
            client_name=str(contact.get("full_name") or "")
        )
        row = await conn.fetchrow(
            """
            INSERT INTO eom_onboarding_email_drafts (
                contact_id, operation_key, recipient_email, blocker,
                subject, body
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6)
            ON CONFLICT (operation_key) DO NOTHING
            RETURNING id
            """,
            str(contact["id"]),
            operation_key,
            recipient,
            None if recipient else "no_email",
            subject,
            body,
        )
        if row is not None:
            return str(row["id"])
        return await DatabaseCRMProvider._eom_onboarding_draft_id_for_operation(
            conn, operation_key
        )

    @staticmethod
    def _eom_onboarding_draft_closed(
        row: Mapping[str, Any], *, idempotent: bool = False
    ) -> dict[str, Any]:
        """Return the closed JSON-safe draft shape the funnel routes expose."""

        def _iso(value: Any) -> str | None:
            return value.isoformat() if value is not None else None

        return {
            "draft_id": str(row["id"]),
            "contact_id": str(row["contact_id"]),
            "status": str(row["status"]),
            "recipient_email": row["recipient_email"],
            "blocker": row["blocker"],
            "subject": str(row["subject"]),
            "body": str(row["body"]),
            "created_at": _iso(row["created_at"]),
            "claimed_at": _iso(row["claimed_at"]),
            "sent_at": _iso(row["sent_at"]),
            "revoked_at": _iso(row["revoked_at"]),
            "approved_by_name": row["approved_by_name"],
            "idempotent": idempotent,
        }

    async def list_eom_onboarding_drafts(
        self,
        *,
        status: str = "pending",
        limit: int = 100,
        cursor_created_at: datetime | None = None,
        cursor_draft_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Return the closed office-review projection of onboarding drafts."""
        from .eom_lead_conversion import EOMLeadConversionError

        if status not in ("pending", "sending", "sent", "revoked"):
            raise EOMLeadConversionError(
                422, "EOM onboarding draft status filter is not recognized"
            )
        cursor_clause = ""
        params: list[Any] = [status, limit]
        if cursor_created_at is not None and cursor_draft_id is not None:
            cursor_clause = "AND (d.created_at, d.id) < ($3::timestamptz, $4::uuid)"
            params.extend([cursor_created_at, cursor_draft_id])
        pool = self._get_pool()
        rows = await pool.fetch(
            f"""
            SELECT
                d.id AS draft_id,
                d.contact_id,
                c.full_name,
                d.recipient_email,
                d.blocker,
                d.subject,
                d.body,
                d.status,
                d.created_at,
                d.claimed_at,
                d.sent_at,
                d.revoked_at,
                d.approved_by_name
            FROM eom_onboarding_email_drafts AS d
            JOIN contacts AS c ON c.id = d.contact_id
            WHERE d.status = $1
              {cursor_clause}
            ORDER BY d.created_at DESC, d.id DESC
            LIMIT $2
            """,
            *params,
        )
        return [dict(row) for row in rows]

    async def list_eom_public_onboarding_issued_links(
        self,
        *,
        accepted_signing_key_fingerprints: Collection[str],
        limit: int = 100,
        cursor_issued_at: datetime | None = None,
        cursor_draft_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Return current issued-token evidence for the private office queue.

        Draft ``sent`` state proves delivery evidence, not whether the customer
        can still use the link. The token row, accepted signing-key fingerprint,
        and current draft/contact readiness predicate jointly own that decision.
        This read deliberately exposes no token ID, signing material, approval
        key, or bearer.
        """

        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        accepted_fingerprints = tuple(accepted_signing_key_fingerprints)
        if not accepted_fingerprints:
            return []
        cursor_clause = ""
        params: list[Any] = [accepted_fingerprints, limit, EOM_BUSINESS_CONTEXT_ID]
        if cursor_issued_at is not None and cursor_draft_id is not None:
            cursor_clause = "AND (token.issued_at, token.draft_id) < ($4::timestamptz, $5::uuid)"
            params.extend([cursor_issued_at, cursor_draft_id])
        pool = self._get_pool()
        if not await pool.fetchval(
            "SELECT to_regclass('eom_public_onboarding_tokens') IS NOT NULL"
        ):
            raise EOMLeadConversionError(
                503, "Public onboarding token storage is unavailable"
            )
        rows = await pool.fetch(
            f"""
            SELECT
                token.draft_id,
                token.contact_id,
                contact.full_name,
                draft.recipient_email,
                token.status,
                token.issued_at
            FROM eom_public_onboarding_tokens AS token
            JOIN eom_onboarding_email_drafts AS draft ON draft.id = token.draft_id
            JOIN contacts AS contact ON contact.id = token.contact_id
            WHERE token.status = 'issued'
              AND token.signing_key_fingerprint = ANY($1::varchar[])
              AND draft.status IN ('sending', 'sent')
              AND contact.business_context_id = $3
              AND contact.status = 'active'
              AND contact.contact_type = 'lead'
              AND contact.lead_stage = 'won'
              {cursor_clause}
            ORDER BY token.issued_at DESC, token.draft_id DESC
            LIMIT $2
            """,
            *params,
        )
        return [dict(row) for row in rows]

    async def get_eom_onboarding_draft(self, draft_id: str) -> dict[str, Any] | None:
        pool = self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM eom_onboarding_email_drafts WHERE id = $1::uuid",
            str(draft_id),
        )
        return dict(row) if row else None

    async def update_eom_onboarding_draft(
        self,
        *,
        draft_id: str,
        subject: str | None = None,
        body: str | None = None,
        recipient_email: str | None = None,
    ) -> dict[str, Any]:
        """Edit a draft while it is still pending.

        Setting a recipient clears blocker='no_email': the draft becomes
        claimable under migration 360's readiness predicate. Any other
        status rejects 409 -- a claimed, sent, or revoked snapshot is
        evidence and must not mutate.
        """
        from .eom_lead_conversion import EOMLeadConversionError

        set_fragments: list[str] = []
        params: list[Any] = [str(draft_id)]
        for column, value in (
            ("subject", subject),
            ("body", body),
        ):
            if value is not None:
                params.append(value)
                set_fragments.append(f"{column} = ${len(params)}")
        if recipient_email is not None:
            params.append(recipient_email)
            set_fragments.append(f"recipient_email = ${len(params)}")
            set_fragments.append("blocker = NULL")
        if not set_fragments:
            raise EOMLeadConversionError(
                422, "EOM onboarding draft edit requires at least one field"
            )
        pool = self._get_pool()
        row = await pool.fetchrow(
            f"""
            UPDATE eom_onboarding_email_drafts
               SET {', '.join(set_fragments)}
             WHERE id = $1::uuid
               AND status = 'pending'
             RETURNING *
            """,
            *params,
        )
        if row is not None:
            return self._eom_onboarding_draft_closed(row)
        existing = await self.get_eom_onboarding_draft(draft_id)
        if existing is None:
            raise EOMLeadConversionError(404, "EOM onboarding draft not found")
        raise EOMLeadConversionError(
            409,
            "EOM onboarding draft is "
            f"{existing['status']}; only pending drafts can be edited",
        )

    async def claim_eom_onboarding_draft(
        self,
        *,
        draft_id: str,
        actor_id: int,
        actor_name: str,
        public_onboarding_base_url: str | None = None,
        public_onboarding_hmac_secret: str | None = None,
    ) -> dict[str, Any]:
        """Atomically claim one pending draft into 'sending' (migration 360).

        The readiness predicate is part of the claim: a blocked or
        recipient-less row is never claimable. Zero updated rows settle to
        an idempotent replay only when the draft is already sent; every
        other state is a 4xx the office can act on.
        """
        from .eom_lead_conversion import EOMLeadConversionError

        if (public_onboarding_base_url is None) != (
            public_onboarding_hmac_secret is None
        ):
            raise RuntimeError("public onboarding link configuration must be paired")

        pool = self._get_pool()
        public_onboarding_enabled = public_onboarding_base_url is not None
        if not public_onboarding_enabled:
            # Read the contact only to acquire the shared teardown fence; the
            # UPDATE remains the claim's atomic readiness predicate.  Without
            # this lock a pending draft could move to sending while a won lead
            # is between its durable cancellation prepare and Calendar DELETE.
            async with _transaction_connection(pool) as conn:
                draft_hint = await conn.fetchrow(
                    """
                    SELECT contact_id
                    FROM eom_onboarding_email_drafts
                    WHERE id = $1::uuid
                    """,
                    str(draft_id),
                )
                if draft_hint is not None:
                    draft_contact_id = str(draft_hint["contact_id"])
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                        _eom_won_lead_loss_execution_lock_key(draft_contact_id),
                    )
                    await self._assert_eom_won_lead_loss_cancellation_fence(
                        conn, contact_id=draft_contact_id
                    )
                row = await conn.fetchrow(
                    """
                    UPDATE eom_onboarding_email_drafts AS d
                       SET status = 'sending', claimed_at = NOW(),
                           approved_by_employee_id = $2, approved_by_name = $3
                     WHERE d.id = $1::uuid
                       AND d.status = 'pending'
                       AND d.blocker IS NULL
                       AND d.recipient_email IS NOT NULL
                       AND EXISTS (
                           SELECT 1
                           FROM contacts AS c
                           WHERE c.id = d.contact_id
                             AND c.business_context_id = 'effingham_maids'
                             AND c.status = 'active'
                       )
                     RETURNING *
                    """,
                    str(draft_id),
                    actor_id,
                    actor_name,
                )
            if row is not None:
                return {
                    "claimed": True,
                    "draft": self._eom_onboarding_draft_closed(row),
                }
        else:
            # Token issuance and the status claim share one transaction. A
            # failed insert (including an unexpected live-token conflict) rolls
            # the draft back to pending, so no email is ever sent with an
            # unredeemable link.
            token_id = uuid4()
            approval_key = f"eom-public-onboarding:{token_id}"
            raw_token = format_eom_public_onboarding_token(
                token_id=token_id,
                secret=str(public_onboarding_hmac_secret),
            )
            signing_key_fingerprint = eom_public_onboarding_hmac_key_fingerprint(
                secret=str(public_onboarding_hmac_secret)
            )
            link = build_eom_public_onboarding_link(
                base_url=str(public_onboarding_base_url),
                token=raw_token,
            )
            async with _transaction_connection(pool) as conn:
                # Token issuance is itself an alternate handoff admission path,
                # not merely an email concern. Take the shared contact lock
                # before changing the draft so a concurrent office handoff
                # cannot finalize the lead between the enabled-claim predicate
                # and token insertion. This hint deliberately takes no row lock:
                # public redemption takes the contact lock before the draft lock,
                # and reversing that order here would create a cycle.
                draft_hint = await conn.fetchrow(
                    """
                    SELECT contact_id
                    FROM eom_onboarding_email_drafts
                    WHERE id = $1::uuid
                    """,
                    str(draft_id),
                )
                if draft_hint is not None:
                    draft_contact_id = str(draft_hint["contact_id"])
                    for lock_key in sorted(
                        {
                            f"eom-customer-handoff:contact:{draft_contact_id}",
                            _eom_won_lead_loss_execution_lock_key(draft_contact_id),
                        }
                    ):
                        await conn.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                            lock_key,
                        )
                    await self._assert_eom_won_lead_loss_cancellation_fence(
                        conn, contact_id=draft_contact_id
                    )
                row = await conn.fetchrow(
                    """
                    UPDATE eom_onboarding_email_drafts AS d
                       SET status = 'sending', claimed_at = NOW(),
                           approved_by_employee_id = $2, approved_by_name = $3
                     WHERE d.id = $1::uuid
                       AND d.status = 'pending'
                       AND d.blocker IS NULL
                       AND d.recipient_email IS NOT NULL
                       AND EXISTS (
                           SELECT 1
                           FROM contacts AS c
                           WHERE c.id = d.contact_id
                             AND c.business_context_id = 'effingham_maids'
                             AND c.status = 'active'
                             AND c.contact_type = 'lead'
                             AND c.lead_stage = 'won'
                       )
                     RETURNING *
                    """,
                    str(draft_id),
                    actor_id,
                    actor_name,
                )
                if row is not None:
                    token_row = await conn.fetchrow(
                        """
                        INSERT INTO eom_public_onboarding_tokens (
                            id, draft_id, contact_id, approval_key,
                            approved_by_employee_id, approved_by_name,
                            signing_key_fingerprint, prefill_full_name,
                            prefill_email, prefill_phone, prefill_address,
                            prefill_city, prefill_state, prefill_zip,
                            prefill_customer_type
                        )
                        SELECT
                            $1, $2::uuid, $3::uuid, $4, $5, $6, $7,
                            contact.full_name, $8, contact.phone,
                            contact.address, contact.city, contact.state,
                            contact.zip, contact.customer_type
                        FROM contacts AS contact
                        WHERE contact.id = $3::uuid
                        ON CONFLICT DO NOTHING
                        RETURNING id
                        """,
                        token_id,
                        str(draft_id),
                        str(row["contact_id"]),
                        approval_key,
                        actor_id,
                        actor_name,
                        signing_key_fingerprint,
                        str(row["recipient_email"]),
                    )
                    if token_row is None:
                        raise EOMLeadConversionError(
                            409,
                            "EOM onboarding draft already has a public onboarding link",
                        )
                    return {
                        "claimed": True,
                        "draft": self._eom_onboarding_draft_closed(row),
                        # Ephemeral transport material only. The raw bearer
                        # never becomes part of the durable draft projection.
                        "public_onboarding_link": link,
                    }
        existing = await self.get_eom_onboarding_draft(draft_id)
        if existing is None:
            raise EOMLeadConversionError(404, "EOM onboarding draft not found")
        status = str(existing["status"])
        if status == "sent":
            return {
                "claimed": False,
                "draft": self._eom_onboarding_draft_closed(
                    existing, idempotent=True
                ),
            }
        if status == "sending":
            raise EOMLeadConversionError(
                409,
                "EOM onboarding draft send is already in flight or requires "
                "reconciliation",
            )
        if status == "revoked":
            raise EOMLeadConversionError(409, "EOM onboarding draft is revoked")
        if existing["blocker"]:
            raise EOMLeadConversionError(
                409,
                f"EOM onboarding draft is blocked: {existing['blocker']}",
            )
        if existing["recipient_email"] is None:
            raise EOMLeadConversionError(
                409, "EOM onboarding draft has no recipient email"
            )
        # The draft itself was claimable, so the contact guard is the only
        # predicate left: same admission the booking family applies before
        # any customer-facing EOM action.
        raise EOMLeadConversionError(
            409,
            "EOM onboarding draft contact is not an active "
            "effingham_maids contact",
        )

    async def confirm_eom_onboarding_draft_sent(
        self, *, draft_id: str, require_stale: bool = False
    ) -> dict[str, Any]:
        """Confirm delivery for a claimed draft (sending -> sent).

        The approve flow confirms immediately after transport acceptance
        (require_stale=False: it KNOWS the send outcome). The operator
        reconciliation route passes require_stale=True so a fresh claim --
        a send that may still be mid-flight with an unknown outcome --
        cannot be recorded as delivered before it settles or goes stale.
        """
        from .eom_lead_conversion import EOMLeadConversionError

        pool = self._get_pool()
        stale_clause = (
            "AND claimed_at <= NOW() - make_interval(mins => $2)"
            if require_stale
            else ""
        )
        params: list[Any] = [str(draft_id)]
        if require_stale:
            params.append(_EOM_ONBOARDING_SENDING_STALE_AFTER_MINUTES)
        row = await pool.fetchrow(
            f"""
            UPDATE eom_onboarding_email_drafts
               SET status = 'sent', sent_at = NOW()
             WHERE id = $1::uuid
               AND status = 'sending'
               {stale_clause}
             RETURNING *
            """,
            *params,
        )
        if row is not None:
            return self._eom_onboarding_draft_closed(row)
        existing = await self.get_eom_onboarding_draft(draft_id)
        if existing is None:
            raise EOMLeadConversionError(404, "EOM onboarding draft not found")
        status = str(existing["status"])
        if status == "sent":
            return self._eom_onboarding_draft_closed(existing, idempotent=True)
        if status == "revoked":
            raise EOMLeadConversionError(
                409,
                "EOM onboarding draft was revoked while sending; reconcile "
                "against the transport log",
            )
        if status == "sending":
            raise EOMLeadConversionError(
                409,
                "EOM onboarding draft send is still in flight; reconcile "
                "only after the claim goes stale",
            )
        raise EOMLeadConversionError(
            409, "EOM onboarding draft has not been claimed for sending"
        )

    async def revoke_eom_onboarding_draft(self, *, draft_id: str) -> dict[str, Any]:
        """Revoke a pending draft, or reconcile a STALE 'sending' one.

        Revoking from 'sending' is the migration-360 operator recovery
        action after checking the transport log, and it is admitted only
        once the claim is stale: an active send between the transport POST
        and its confirmation must not be recordable as revoked when the
        customer email may already be delivered. A sent draft is immutable
        delivery evidence and rejects.
        """
        from .eom_lead_conversion import EOMLeadConversionError

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            # The public finalizer uses this exact lock before it locks the
            # token/draft row. A stale-send reconciliation therefore cannot
            # revoke a link between its validation and redemption.
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"eom-public-onboarding:draft:{draft_id}",
            )
            row = await conn.fetchrow(
                """
                UPDATE eom_onboarding_email_drafts
                   SET status = 'revoked', revoked_at = NOW()
                 WHERE id = $1::uuid
                   AND (
                       status = 'pending'
                       OR (
                           status = 'sending'
                           AND claimed_at <= NOW() - make_interval(mins => $2)
                       )
                   )
                 RETURNING *
                """,
                str(draft_id),
                _EOM_ONBOARDING_SENDING_STALE_AFTER_MINUTES,
            )
            if row is not None:
                if await conn.fetchval(
                    "SELECT to_regclass('eom_public_onboarding_tokens') IS NOT NULL"
                ):
                    await conn.execute(
                        """
                        UPDATE eom_public_onboarding_tokens
                           SET status = 'revoked', revoked_at = NOW()
                         WHERE draft_id = $1::uuid
                           AND status = 'issued'
                        """,
                        str(draft_id),
                    )
                return self._eom_onboarding_draft_closed(row)
        existing = await self.get_eom_onboarding_draft(draft_id)
        if existing is None:
            raise EOMLeadConversionError(404, "EOM onboarding draft not found")
        status = str(existing["status"])
        if status == "revoked":
            return self._eom_onboarding_draft_closed(existing, idempotent=True)
        if status == "sending":
            raise EOMLeadConversionError(
                409,
                "EOM onboarding draft send is still in flight; reconcile "
                "only after the claim goes stale",
            )
        raise EOMLeadConversionError(
            409, "EOM onboarding draft was already sent and cannot be revoked"
        )

    async def _get_eom_public_onboarding_session_result(
        self, *, token_id: str, signing_key_fingerprint: str
    ) -> dict[str, Any]:
        """Read one durable token state for the public session projections.

        The sibling public methods intentionally choose their own output
        allowlists. The browser-safe session keeps Atlas identifiers out of its
        response, while the tracker-only context route needs those opaque IDs
        to retain a recoverable local reservation.
        """

        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        pool = self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT
                token.id AS token_id,
                token.draft_id,
                token.status AS token_status,
                token.handoff_id,
                draft.status AS draft_status,
                contact.id AS contact_id,
                contact.business_context_id,
                contact.contact_type,
                contact.lead_stage,
                contact.status AS contact_status,
                token.prefill_full_name,
                token.prefill_email,
                token.prefill_phone,
                token.prefill_address,
                token.prefill_city,
                token.prefill_state,
                token.prefill_zip,
                token.prefill_customer_type,
                handoff.tracker_customer_id,
                handoff.tracker_site_id
            FROM eom_public_onboarding_tokens AS token
            JOIN eom_onboarding_email_drafts AS draft ON draft.id = token.draft_id
            JOIN contacts AS contact ON contact.id = token.contact_id
            LEFT JOIN eom_customer_handoffs AS handoff ON handoff.id = token.handoff_id
            WHERE token.id = $1::uuid
              AND token.signing_key_fingerprint = $2
            """,
            token_id,
            signing_key_fingerprint,
        )
        if row is None:
            raise EOMLeadConversionError(404, "Public onboarding link is unavailable")
        token_status = str(row["token_status"])
        if token_status == "redeemed":
            if (
                row["handoff_id"] is None
                or row["tracker_customer_id"] is None
                or row["tracker_site_id"] is None
            ):
                raise EOMLeadConversionError(
                    409, "Public onboarding completion evidence is incomplete"
                )
            return {
                "status": "completed",
                "token_id": str(row["token_id"]),
                "draft_id": str(row["draft_id"]),
                "contact_id": str(row["contact_id"]),
                "tracker_customer_id": int(row["tracker_customer_id"]),
                "tracker_site_id": int(row["tracker_site_id"]),
                "handoff_id": str(row["handoff_id"]),
                "idempotent": True,
            }
        if token_status != "issued":
            raise EOMLeadConversionError(404, "Public onboarding link is unavailable")
        if (
            row["draft_status"] not in ("sending", "sent")
            or row["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
            or row["contact_status"] != "active"
            or row["contact_type"] != "lead"
            or row["lead_stage"] != "won"
        ):
            raise EOMLeadConversionError(404, "Public onboarding link is unavailable")
        return {
            "status": "ready",
            "token_id": str(row["token_id"]),
            "draft_id": str(row["draft_id"]),
            "contact_id": str(row["contact_id"]),
            "full_name": str(row["prefill_full_name"]),
            "email": row["prefill_email"],
            "phone": row["prefill_phone"],
            "address": row["prefill_address"],
            "city": row["prefill_city"],
            "state": row["prefill_state"],
            "zip": row["prefill_zip"],
            "customer_type": row["prefill_customer_type"],
        }

    async def get_eom_public_onboarding_session(
        self, *, token_id: str, signing_key_fingerprint: str
    ) -> dict[str, Any]:
        """Return the existing browser-safe token-bound projection."""

        result = await self._get_eom_public_onboarding_session_result(
            token_id=token_id,
            signing_key_fingerprint=signing_key_fingerprint,
        )
        if result["status"] == "completed":
            return {
                field: result[field]
                for field in (
                    "status",
                    "contact_id",
                    "tracker_customer_id",
                    "tracker_site_id",
                    "handoff_id",
                    "idempotent",
                )
            }
        return {
            field: result[field]
            for field in (
                "status",
                "contact_id",
                "full_name",
                "email",
                "phone",
                "address",
                "city",
                "state",
                "zip",
                "customer_type",
            )
        }

    async def get_eom_public_onboarding_tracker_context(
        self, *, token_id: str, signing_key_fingerprint: str
    ) -> dict[str, Any]:
        """Return the private token/draft/contact context for the Tracker only."""

        return await self._get_eom_public_onboarding_session_result(
            token_id=token_id,
            signing_key_fingerprint=signing_key_fingerprint,
        )

    async def complete_eom_public_onboarding(
        self,
        *,
        token_id: str,
        signing_key_fingerprint: str,
        tracker_customer_id: int,
        tracker_site_id: int,
    ) -> dict[str, Any]:
        """Redeem one issued link into the existing immutable handoff.

        The initial token read merely supplies the fixed lock vocabulary. The
        decisive token/draft/contact read happens after all lock keys are held;
        that is the linearization point shared with the ordinary office handoff.
        ``finalize_eom_customer_handoff`` receives this same connection, so its
        lead transition, lifecycle evidence, handoff insert, and token redemption
        either all commit or all roll back together.
        """

        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            token_hint = await conn.fetchrow(
                """
                SELECT draft_id, contact_id, approval_key
                FROM eom_public_onboarding_tokens
                WHERE id = $1::uuid
                  AND signing_key_fingerprint = $2
                """,
                token_id,
                signing_key_fingerprint,
            )
            if token_hint is None:
                raise EOMLeadConversionError(
                    404, "Public onboarding link is unavailable"
                )
            lock_keys = _eom_customer_handoff_lock_keys(
                approval_key=str(token_hint["approval_key"]),
                contact_id=str(token_hint["contact_id"]),
                tracker_customer_id=tracker_customer_id,
                tracker_site_id=tracker_site_id,
            )
            lock_keys.append(
                f"eom-public-onboarding:draft:{token_hint['draft_id']}"
            )
            for lock_key in sorted(set(lock_keys)):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            await self._assert_eom_won_lead_loss_cancellation_fence(
                conn, contact_id=str(token_hint["contact_id"])
            )
            row = await conn.fetchrow(
                """
                SELECT
                    token.id AS token_id,
                    token.status AS token_status,
                    token.approval_key,
                    token.approved_by_employee_id,
                    token.approved_by_name,
                    token.handoff_id,
                    draft.status AS draft_status,
                    contact.id AS contact_id,
                    contact.business_context_id,
                    contact.contact_type,
                    contact.lead_stage,
                    contact.status AS contact_status,
                    handoff.tracker_customer_id,
                    handoff.tracker_site_id,
                    handoff.approval_key AS handoff_approval_key
                FROM eom_public_onboarding_tokens AS token
                JOIN eom_onboarding_email_drafts AS draft ON draft.id = token.draft_id
                JOIN contacts AS contact ON contact.id = token.contact_id
                LEFT JOIN eom_customer_handoffs AS handoff ON handoff.id = token.handoff_id
                WHERE token.id = $1::uuid
                  AND token.signing_key_fingerprint = $2
                FOR UPDATE OF token, draft, contact
                """,
                token_id,
                signing_key_fingerprint,
            )
            if row is None:
                raise EOMLeadConversionError(
                    404, "Public onboarding link is unavailable"
                )
            token_status = str(row["token_status"])
            if token_status == "redeemed":
                if (
                    row["handoff_id"] is None
                    or row["tracker_customer_id"] is None
                    or row["tracker_site_id"] is None
                    or str(row["handoff_approval_key"]) != str(row["approval_key"])
                ):
                    raise EOMLeadConversionError(
                        409, "Public onboarding completion evidence is incomplete"
                    )
                if (
                    int(row["tracker_customer_id"]) != tracker_customer_id
                    or int(row["tracker_site_id"]) != tracker_site_id
                ):
                    raise EOMLeadConversionError(
                        409,
                        "Public onboarding link was already completed for different tracker records",
                    )
                return {
                    "status": "completed",
                    "contact_id": str(row["contact_id"]),
                    "tracker_customer_id": tracker_customer_id,
                    "tracker_site_id": tracker_site_id,
                    "handoff_id": str(row["handoff_id"]),
                    "idempotent": True,
                }
            if token_status != "issued":
                raise EOMLeadConversionError(
                    409, "Public onboarding link is unavailable"
                )
            if (
                row["draft_status"] not in ("sending", "sent")
                or row["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
                or row["contact_status"] != "active"
                or row["contact_type"] != "lead"
                or row["lead_stage"] != "won"
            ):
                raise EOMLeadConversionError(
                    409, "Public onboarding link is unavailable"
                )
            handoff = await self.finalize_eom_customer_handoff(
                contact_id=str(row["contact_id"]),
                tracker_customer_id=tracker_customer_id,
                tracker_site_id=tracker_site_id,
                approval_key=str(row["approval_key"]),
                actor_id=int(row["approved_by_employee_id"]),
                actor_name=str(row["approved_by_name"]),
                connection=conn,
                permitted_public_onboarding_token_id=str(row["token_id"]),
            )
            redeemed = await conn.fetchrow(
                """
                UPDATE eom_public_onboarding_tokens
                   SET status = 'redeemed', redeemed_at = NOW(), handoff_id = $2::uuid
                 WHERE id = $1::uuid
                   AND status = 'issued'
                 RETURNING id, contact_id, handoff_id
                """,
                token_id,
                handoff["handoff_id"],
            )
            if redeemed is None:
                raise RuntimeError("public onboarding token changed during finalization")
            return {
                "status": "completed",
                "contact_id": str(redeemed["contact_id"]),
                "tracker_customer_id": tracker_customer_id,
                "tracker_site_id": tracker_site_id,
                "handoff_id": str(redeemed["handoff_id"]),
                "idempotent": bool(handoff["idempotent"]),
            }

    async def recover_eom_public_onboarding(
        self,
        *,
        token_id: str,
        contact_id: str,
        tracker_customer_id: int,
        tracker_site_id: int,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Finish a Tracker-local reservation without retaining a raw bearer.

        The recovery operation owns one token/contact/Tracker-ID decision. It
        uses the same sorted handoff lock vocabulary as public and office
        finalizers, plus the token's draft lock. An issued token is revoked and
        finalized through the existing office path in the same transaction, so
        a failed handoff cannot strand it in a new terminal state.
        """

        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        def _completed(handoff_id: object, *, idempotent: bool) -> dict[str, Any]:
            return {
                "status": "completed",
                "contact_id": contact_id,
                "tracker_customer_id": tracker_customer_id,
                "tracker_site_id": tracker_site_id,
                "handoff_id": str(handoff_id),
                "idempotent": idempotent,
            }

        recovery_approval_key = f"eom-public-onboarding-recovery:{token_id}"
        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            if not await conn.fetchval(
                "SELECT to_regclass('eom_public_onboarding_tokens') IS NOT NULL"
            ):
                raise EOMLeadConversionError(
                    503, "Public onboarding token storage is unavailable"
                )
            token_hint = await conn.fetchrow(
                """
                SELECT draft_id
                FROM eom_public_onboarding_tokens
                WHERE id = $1::uuid
                  AND contact_id = $2::uuid
                """,
                token_id,
                contact_id,
            )
            if token_hint is None:
                raise EOMLeadConversionError(
                    404, "Public onboarding recovery was not found"
                )
            lock_keys = _eom_customer_handoff_lock_keys(
                approval_key=recovery_approval_key,
                contact_id=contact_id,
                tracker_customer_id=tracker_customer_id,
                tracker_site_id=tracker_site_id,
            )
            lock_keys.append(
                f"eom-public-onboarding:draft:{token_hint['draft_id']}"
            )
            for lock_key in sorted(set(lock_keys)):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            await self._assert_eom_won_lead_loss_cancellation_fence(
                conn, contact_id=contact_id
            )
            row = await conn.fetchrow(
                """
                SELECT
                    token.id AS token_id,
                    token.status AS token_status,
                    token.approval_key,
                    token.handoff_id,
                    draft.status AS draft_status,
                    contact.id AS contact_id,
                    contact.business_context_id,
                    contact.contact_type,
                    contact.lead_stage,
                    contact.status AS contact_status,
                    handoff.tracker_customer_id,
                    handoff.tracker_site_id,
                    handoff.approval_key AS handoff_approval_key
                FROM eom_public_onboarding_tokens AS token
                JOIN eom_onboarding_email_drafts AS draft ON draft.id = token.draft_id
                JOIN contacts AS contact ON contact.id = token.contact_id
                LEFT JOIN eom_customer_handoffs AS handoff ON handoff.id = token.handoff_id
                WHERE token.id = $1::uuid
                  AND token.contact_id = $2::uuid
                FOR UPDATE OF token, draft, contact
                """,
                token_id,
                contact_id,
            )
            if row is None:
                raise RuntimeError("public onboarding token changed during recovery")
            token_status = str(row["token_status"])
            if token_status == "redeemed":
                if (
                    row["handoff_id"] is None
                    or row["tracker_customer_id"] is None
                    or row["tracker_site_id"] is None
                    or str(row["handoff_approval_key"]) != str(row["approval_key"])
                ):
                    raise EOMLeadConversionError(
                        409, "Public onboarding completion evidence is incomplete"
                    )
                if (
                    int(row["tracker_customer_id"]) != tracker_customer_id
                    or int(row["tracker_site_id"]) != tracker_site_id
                ):
                    raise EOMLeadConversionError(
                        409,
                        "Public onboarding link was already completed for different tracker records",
                    )
                return _completed(row["handoff_id"], idempotent=True)
            if token_status not in ("issued", "revoked"):
                raise EOMLeadConversionError(
                    409, "Public onboarding recovery is unavailable"
                )

            existing_handoff = await conn.fetchrow(
                """
                SELECT id, tracker_customer_id, tracker_site_id
                FROM eom_customer_handoffs
                WHERE contact_id = $1::uuid
                FOR UPDATE
                """,
                contact_id,
            )
            if existing_handoff is not None:
                if (
                    int(existing_handoff["tracker_customer_id"])
                    != tracker_customer_id
                    or int(existing_handoff["tracker_site_id"]) != tracker_site_id
                ):
                    raise EOMLeadConversionError(
                        409,
                        "EOM lead already has a different customer handoff",
                    )
                if token_status == "issued":
                    revoked = await conn.fetchrow(
                        """
                        UPDATE eom_public_onboarding_tokens
                           SET status = 'revoked', revoked_at = NOW()
                         WHERE id = $1::uuid
                           AND status = 'issued'
                         RETURNING id
                        """,
                        token_id,
                    )
                    if revoked is None:
                        raise RuntimeError(
                            "public onboarding token changed during recovery"
                        )
                return _completed(existing_handoff["id"], idempotent=True)

            if (
                row["draft_status"] not in ("sending", "sent")
                or row["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
                or row["contact_status"] != "active"
                or row["contact_type"] != "lead"
                or row["lead_stage"] != "won"
            ):
                raise EOMLeadConversionError(
                    409, "Public onboarding recovery is unavailable"
                )
            if token_status == "issued":
                revoked = await conn.fetchrow(
                    """
                    UPDATE eom_public_onboarding_tokens
                       SET status = 'revoked', revoked_at = NOW()
                     WHERE id = $1::uuid
                       AND status = 'issued'
                     RETURNING id
                    """,
                    token_id,
                )
                if revoked is None:
                    raise RuntimeError("public onboarding token changed during recovery")
            handoff = await self.finalize_eom_customer_handoff(
                contact_id=str(row["contact_id"]),
                tracker_customer_id=tracker_customer_id,
                tracker_site_id=tracker_site_id,
                approval_key=recovery_approval_key,
                actor_id=actor_id,
                actor_name=actor_name,
                connection=conn,
            )
            return _completed(
                handoff["handoff_id"], idempotent=bool(handoff["idempotent"])
            )

    async def revoke_eom_public_onboarding_token(
        self, *, draft_id: str
    ) -> dict[str, Any]:
        """Revoke an issued public link without rewriting sent-email evidence."""

        from .eom_lead_conversion import EOMLeadConversionError

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            if not await conn.fetchval(
                "SELECT to_regclass('eom_public_onboarding_tokens') IS NOT NULL"
            ):
                raise EOMLeadConversionError(
                    503, "Public onboarding token storage is unavailable"
                )
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"eom-public-onboarding:draft:{draft_id}",
            )
            row = await conn.fetchrow(
                """
                SELECT id, contact_id, status
                FROM eom_public_onboarding_tokens
                WHERE draft_id = $1::uuid
                FOR UPDATE
                """,
                draft_id,
            )
            if row is None:
                raise EOMLeadConversionError(
                    404, "Public onboarding link was not found"
                )
            token_status = str(row["status"])
            if token_status == "redeemed":
                raise EOMLeadConversionError(
                    409, "Completed public onboarding cannot be revoked"
                )
            if token_status == "revoked":
                return {
                    "token_id": str(row["id"]),
                    "contact_id": str(row["contact_id"]),
                    "status": "revoked",
                    "idempotent": True,
                }
            revoked = await conn.fetchrow(
                """
                UPDATE eom_public_onboarding_tokens
                   SET status = 'revoked', revoked_at = NOW()
                 WHERE id = $1::uuid
                   AND status = 'issued'
                 RETURNING id, contact_id
                """,
                row["id"],
            )
            if revoked is None:
                raise RuntimeError("public onboarding token changed during revocation")
            return {
                "token_id": str(revoked["id"]),
                "contact_id": str(revoked["contact_id"]),
                "status": "revoked",
                "idempotent": False,
            }

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
        # Interaction evidence must land in the store this provider is
        # bound to: the EOM funnel runs the provider against its own
        # connection string, where the global pool may be a different
        # database (or uninitialized).
        pool = self._get_pool()
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
                "crm.interaction_logged",
                "crm_provider",
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
        self,
        contact_id: str,
        limit: int = 20,
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

    async def _assert_eom_public_onboarding_fence(
        self,
        conn: Any,
        *,
        contact_id: str,
        permitted_token_id: str | None,
    ) -> None:
        """Fence office conversion while an issued public link owns the lead.

        A code deployment can precede migration 383 while the feature is
        disabled. The relation probe preserves the existing office handoff in
        that deliberately dormant state; once the table exists, the matching
        row is locked under the same sorted advisory-lock vocabulary as both
        finalizers, so either channel observes one serialized decision.
        """

        from .eom_lead_conversion import EOMLeadConversionError

        relation_exists = await conn.fetchval(
            "SELECT to_regclass('eom_public_onboarding_tokens') IS NOT NULL"
        )
        if not relation_exists:
            if permitted_token_id is not None:
                raise EOMLeadConversionError(
                    503, "Public onboarding token storage is unavailable"
                )
            return
        issued = await conn.fetchrow(
            """
            SELECT id
            FROM eom_public_onboarding_tokens
            WHERE contact_id = $1::uuid
              AND status = 'issued'
            FOR UPDATE
            """,
            contact_id,
        )
        if issued is None:
            if permitted_token_id is not None:
                raise EOMLeadConversionError(
                    409, "Public onboarding link is no longer available"
                )
            return
        if permitted_token_id is not None and str(issued["id"]) == str(
            permitted_token_id
        ):
            return
        raise EOMLeadConversionError(
            409,
            "An active public onboarding link must be revoked before office approval",
        )

    async def _assert_eom_won_lead_loss_cancellation_fence(
        self,
        conn: Any,
        *,
        contact_id: str,
        permitted_operation_key: str | None = None,
    ) -> None:
        """Block competing writes while a durable won-loss cancellation is open.

        The shared advisory lock serializes a live executor, but it vanishes if
        the executor crashes or returns after an uncertain Calendar DELETE. The
        requested lifecycle row survives that exit and is the authoritative
        fence until the same operation records its atomic cancellation/loss
        completion. ``permitted_operation_key`` is used only by prepare: a
        retry may reuse its own evidence, but no second key may start another
        cancellation for the same contact.
        """

        from .eom_lead_conversion import EOMLeadConversionError

        unresolved = await conn.fetchrow(
            """
            SELECT requested.operation_key
            FROM eom_lead_lifecycle_events AS requested
            WHERE requested.contact_id = $1::uuid
              AND requested.event_type = $2::varchar
              AND (
                  $4::varchar IS NULL
                  OR requested.operation_key IS DISTINCT FROM $4::varchar
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM eom_lead_lifecycle_events AS completed
                  WHERE completed.contact_id = requested.contact_id
                    AND completed.event_type = $3::varchar
                    AND completed.operation_key = requested.operation_key
              )
            ORDER BY requested.created_at DESC, requested.id DESC
            LIMIT 1
            FOR UPDATE OF requested
            """,
            contact_id,
            _EOM_WON_LOSS_CANCELLATION_REQUESTED_EVENT,
            _EOM_WON_LOSS_CANCELLATION_COMPLETED_EVENT,
            permitted_operation_key,
        )
        if unresolved is not None:
            raise EOMLeadConversionError(
                409,
                "EOM won lead loss cancellation requires reconciliation before "
                "the lead can change",
            )

    async def finalize_eom_customer_handoff(
        self,
        *,
        contact_id: str,
        tracker_customer_id: int,
        tracker_site_id: int,
        approval_key: str,
        actor_id: int,
        actor_name: str,
        connection: Any | None = None,
        permitted_public_onboarding_token_id: str | None = None,
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

        # This is derived from the only alternate admission path. It is not an
        # independently caller-controlled metadata field, so future internal
        # callers cannot label an ordinary office conversion as public.
        completion_channel = (
            "public_onboarding"
            if permitted_public_onboarding_token_id is not None
            else "office"
        )

        pool = self._get_pool()
        async with _transaction_connection(pool, connection) as conn:
            lock_keys = _eom_customer_handoff_lock_keys(
                approval_key=approval_key,
                contact_id=contact_id,
                tracker_customer_id=tracker_customer_id,
                tracker_site_id=tracker_site_id,
            )
            for lock_key in lock_keys:
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            await self._assert_eom_won_lead_loss_cancellation_fence(
                conn, contact_id=contact_id
            )
            await self._assert_eom_public_onboarding_fence(
                conn,
                contact_id=contact_id,
                permitted_token_id=permitted_public_onboarding_token_id,
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
            if (
                contact is None
                or contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
            ):
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
            booking_events = await conn.fetch(
                """
                SELECT event_type, operation_key
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = ANY($2::varchar[])
                FOR UPDATE
                """,
                contact_id,
                list(_ALL_EOM_BOOKING_EVENT_TYPES),
            )
            booking_event_types: dict[str, set[str]] = {}
            for event in booking_events:
                booking_event_types.setdefault(event["operation_key"], set()).add(
                    event["event_type"]
                )
            # A terminal failed marker alone does not prove the key settled:
            # a concurrent same-key call past preparation may still be talking
            # to Calendar and can produce a stronger booked/ambiguous outcome.
            # The executor of EITHER booking family holds a session advisory
            # lock on eom-estimate-booking:execution:<key> for its whole
            # attempt, so a failed try-lock here means an execution is in
            # flight and handoff must stay fenced until it settles.
            for operation_key, event_types in booking_event_types.items():
                if event_types & _ALL_EOM_BOOKED_EVENTS:
                    continue
                execution_settled = bool(
                    await conn.fetchval(
                        "SELECT pg_try_advisory_xact_lock(hashtextextended($1, 0))",
                        f"eom-estimate-booking:execution:{operation_key}",
                    )
                )
                if not execution_settled:
                    raise EOMLeadConversionError(
                        409,
                        "EOM booking is still executing; retry after it settles",
                    )
            blocking_booking = next(
                (
                    event_types
                    for event_types in booking_event_types.values()
                    if (
                        bool(event_types & _ALL_EOM_AMBIGUOUS_EVENTS)
                        and not (event_types & _ALL_EOM_BOOKED_EVENTS)
                    )
                    or (
                        bool(event_types & _ALL_EOM_REQUESTED_EVENTS)
                        and not self._eom_estimate_booking_operation_is_terminal(
                            event_types
                        )
                    )
                ),
                None,
            )
            if blocking_booking is not None:
                if blocking_booking & _ALL_EOM_AMBIGUOUS_EVENTS:
                    raise EOMLeadConversionError(
                        409,
                        "EOM booking requires calendar reconciliation",
                    )
                raise EOMLeadConversionError(
                    409,
                    "EOM booking is still pending calendar completion",
                )
            if contact["status"] != "active":
                raise EOMLeadConversionError(
                    409, "EOM lead must be active before approval"
                )
            if contact["contact_type"] != "lead":
                raise EOMLeadConversionError(409, "EOM contact is not a lead")
            if contact["lead_stage"] not in ("new", "estimate_booked", "won"):
                raise EOMLeadConversionError(409, "EOM lead is not ready for approval")
            from_stage = str(contact["lead_stage"])

            updated = await conn.fetchrow(
                """
                UPDATE contacts
                SET contact_type = 'customer', lead_stage = NULL, updated_at = NOW()
                WHERE id = $1
                  AND business_context_id = $2
                  AND contact_type = 'lead'
                  AND lead_stage = $3
                  AND status = 'active'
                RETURNING id
                """,
                contact_id,
                EOM_BUSINESS_CONTEXT_ID,
                from_stage,
            )
            if updated is None:
                raise RuntimeError(
                    "EOM lead changed during customer handoff finalization"
                )
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, metadata
                )
                VALUES ($1, 'customer_approved', $7, NULL, $2, 'eom_office', $3,
                        jsonb_build_object(
                            'tracker_customer_id', $4::bigint,
                            'tracker_site_id', $5::bigint,
                            'approved_by_employee_id', $6::bigint,
                            'completion_channel', $8::text
                        ))
                """,
                contact_id,
                f"employee:{actor_id}:{actor_name}",
                approval_key,
                tracker_customer_id,
                tracker_site_id,
                actor_id,
                from_stage,
                completion_channel,
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
            return _result(handoff, idempotent=False)

    async def prepare_eom_won_lead_loss(
        self,
        *,
        contact_id: str,
        reason_code: str,
        note: str | None,
        operation_key: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Persist the immutable facts required to lose a won EOM lead.

        The normal pre-won disposition remains in ``mark_eom_lead_lost``.
        A won lead is different because its first-clean event already exists
        outside PostgreSQL and its welcome draft may be claimable.  This method
        records the exact event/draft pair before the external DELETE, allowing
        a retry to use the same facts rather than today's configured Calendar
        or a rediscovered appointment.
        """
        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        def _result(*, idempotent: bool) -> dict[str, Any]:
            return {
                "contact_id": str(contact_id),
                "lead_stage": "lost",
                "status": "lost",
                "reason_code": reason_code,
                "from_stage": "won",
                "idempotent": idempotent,
            }

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            for lock_key in sorted(
                {
                    f"eom-lead-lost:contact:{contact_id}",
                    f"eom-lead-lost:operation:{operation_key}",
                }
            ):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )

            contact = await conn.fetchrow(
                """
                SELECT id, business_context_id, contact_type, lead_stage, status
                FROM contacts
                WHERE id = $1::uuid
                FOR UPDATE
                """,
                contact_id,
            )
            if (
                contact is None
                or contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
            ):
                raise EOMLeadConversionError(404, "EOM lead was not found")

            # Preserve the direct writer's pre-won state machine without
            # acquiring or interpreting an unrelated operation's lifecycle
            # rows. Only a current won lead, or a currently lost lead that
            # might be a completed won-loss replay, needs this protocol.
            if contact["lead_stage"] not in ("won", "lost"):
                return {"mode": "pre_won"}

            key_events = await conn.fetch(
                """
                SELECT event_type, from_stage, to_stage, reason, metadata
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1::uuid
                  AND operation_key = $2
                FOR UPDATE
                """,
                contact_id,
                operation_key,
            )
            events_by_type = {str(event["event_type"]): event for event in key_events}
            completed = events_by_type.get(_EOM_WON_LOSS_CANCELLATION_COMPLETED_EVENT)
            lost = events_by_type.get("lead_lost")
            if contact["lead_stage"] == "won":
                await self._assert_eom_won_lead_loss_cancellation_fence(
                    conn,
                    contact_id=contact_id,
                    permitted_operation_key=operation_key,
                )
            won_protocol_candidate = completed is not None or (
                lost is not None and lost["from_stage"] == "won"
            ) or contact["lead_stage"] == "won"
            if won_protocol_candidate:
                foreign_key_owner = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM eom_lead_lifecycle_events
                        WHERE operation_key = $1
                          AND contact_id <> $2::uuid
                    )
                    """,
                    operation_key,
                    contact_id,
                )
                if foreign_key_owner:
                    raise EOMLeadConversionError(
                        409, "Idempotency-Key already belongs to another EOM lead"
                    )
            # A legacy/pre-won lead_lost row intentionally remains the direct
            # writer's replay contract.  Only a won-shaped loss is owned by
            # this protocol; otherwise merely routing through the service
            # would turn an existing 200 replay into a new 409.
            if completed is not None or (
                lost is not None and lost["from_stage"] == "won"
            ):
                if (
                    completed is None
                    or lost is None
                    or lost["from_stage"] != "won"
                    or lost["to_stage"] != "lost"
                    or contact["contact_type"] != "lead"
                    or contact["lead_stage"] != "lost"
                ):
                    raise EOMLeadConversionError(
                        409, "EOM won lead loss operation was superseded"
                    )
                lost_metadata = self._eom_estimate_booking_metadata_from_row(
                    lost["metadata"]
                )
                return {
                    "mode": "completed",
                    "result": _result(
                        idempotent=True,
                    )
                    | {
                        "reason_code": str(
                            lost_metadata.get("lost_reason_code") or reason_code
                        )
                    },
                }

            # The direct writer deliberately owns all legacy/pre-won behavior,
            # including its replay and reason semantics.  The service retains
            # its execution lock while it delegates, which keeps the newer
            # fence vocabulary closed without changing that state machine.
            if contact["lead_stage"] != "won":
                return {"mode": "pre_won"}
            if any(
                str(event["event_type"]) not in _EOM_WON_LOSS_EVENT_TYPES
                or (
                    str(event["event_type"]) == "lead_lost"
                    and (
                        event["from_stage"] != "won"
                        or event["to_stage"] != "lost"
                    )
                )
                for event in key_events
            ):
                raise EOMLeadConversionError(
                    409, "Idempotency-Key already belongs to another EOM operation"
                )
            if contact["contact_type"] != "lead":
                raise EOMLeadConversionError(409, "EOM contact is not a lead")
            if contact["status"] != "active":
                raise EOMLeadConversionError(
                    409, "EOM lead must be active to mark lost"
                )

            booked_events = await conn.fetch(
                """
                SELECT operation_key, metadata
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1::uuid
                  AND event_type = 'first_clean_booked'
                ORDER BY lifecycle_sequence DESC NULLS LAST, created_at DESC, id DESC
                FOR UPDATE
                """,
                contact_id,
            )
            if len(booked_events) != 1:
                raise EOMLeadConversionError(
                    409,
                    "EOM won lead requires one reconciled first-clean booking",
                )
            booked = booked_events[0]
            booking_key = str(booked["operation_key"] or "").strip()
            booking_metadata = self._eom_estimate_booking_metadata_from_row(
                booked["metadata"]
            )
            calendar_id = str(booking_metadata.get("calendar_id") or "").strip()
            expected_event_id = str(
                booking_metadata.get("expected_calendar_event_id") or ""
            ).strip()
            calendar_event_id = str(
                booking_metadata.get("calendar_event_id") or ""
            ).strip()
            if (
                not booking_key
                or not calendar_id
                or not expected_event_id
                or calendar_event_id != expected_event_id
            ):
                raise EOMLeadConversionError(
                    409,
                    "EOM first-clean booking lacks a reconciled Calendar event",
                )
            if calendar_id.casefold() == "primary":
                raise EOMLeadConversionError(
                    409,
                    "EOM first-clean booking uses a relative Calendar identifier; "
                    "reconcile the Calendar principal before marking this won lead lost",
                )

            draft = await conn.fetchrow(
                """
                SELECT id, operation_key, status
                FROM eom_onboarding_email_drafts
                WHERE contact_id = $1::uuid
                  AND operation_key = $2
                FOR UPDATE
                """,
                contact_id,
                booking_key,
            )
            if draft is None:
                raise EOMLeadConversionError(
                    409, "EOM won lead lacks its first-clean onboarding draft"
                )
            draft_status = str(draft["status"])
            if draft_status not in ("pending", "revoked"):
                raise EOMLeadConversionError(
                    409,
                    "EOM onboarding draft delivery must be reconciled before "
                    "marking this won lead lost",
                )
            draft_id = str(draft["id"])
            if await conn.fetchval(
                "SELECT to_regclass('eom_public_onboarding_tokens') IS NOT NULL"
            ):
                issued_public_link = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM eom_public_onboarding_tokens
                        WHERE draft_id = $1::uuid
                          AND contact_id = $2::uuid
                          AND status = 'issued'
                    )
                    """,
                    draft_id,
                    contact_id,
                )
                if issued_public_link:
                    raise EOMLeadConversionError(
                        409,
                        "EOM onboarding draft has an issued public onboarding link",
                    )

            requested = events_by_type.get(_EOM_WON_LOSS_CANCELLATION_REQUESTED_EVENT)
            facts = {
                "booking_operation_key": booking_key,
                "calendar_id": calendar_id,
                "calendar_event_id": calendar_event_id,
                "expected_calendar_event_id": expected_event_id,
                "onboarding_draft_id": draft_id,
                "lost_reason_code": reason_code,
            }
            if requested is not None:
                requested_metadata = self._eom_estimate_booking_metadata_from_row(
                    requested["metadata"]
                )
                if any(
                    str(requested_metadata.get(key) or "") != str(value)
                    for key, value in facts.items()
                    if key != "lost_reason_code"
                ):
                    raise EOMLeadConversionError(
                        409,
                        "EOM won lead loss preparation no longer matches "
                        "first-clean facts",
                    )
                # The same key is a retry.  Preserve its original reason just
                # like the existing direct lost writer preserves its ledger
                # reason on replay.
                facts["lost_reason_code"] = str(
                    requested_metadata.get("lost_reason_code") or reason_code
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO eom_lead_lifecycle_events (
                        contact_id, event_type, from_stage, to_stage, actor,
                        source, operation_key, reason, metadata
                    )
                    VALUES ($1::uuid, $2, 'won', 'won', $3, 'eom_office', $4,
                            $5, $6::jsonb)
                    """,
                    contact_id,
                    _EOM_WON_LOSS_CANCELLATION_REQUESTED_EVENT,
                    f"employee:{actor_id}:{actor_name}",
                    operation_key,
                    note,
                    json.dumps(facts),
                )

            return {
                "mode": "won",
                "contact_id": str(contact["id"]),
                "calendar_id": calendar_id,
                "calendar_event_id": calendar_event_id,
                "expected_calendar_event_id": expected_event_id,
                "onboarding_draft_id": draft_id,
                "booking_operation_key": booking_key,
                "reason_code": facts["lost_reason_code"],
                "idempotent": requested is not None,
            }

    async def mark_eom_won_lead_loss_calendar_unsettled(
        self,
        *,
        contact_id: str,
        operation_key: str,
        calendar_id: str,
        calendar_event_id: str,
        calendar_error: str | None,
        calendar_message: str,
        actor_id: int,
        actor_name: str,
    ) -> None:
        """Append evidence of an uncertain DELETE without changing lead state."""
        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"eom-lead-lost:operation:{operation_key}",
            )
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, reason, metadata
                )
                SELECT c.id, $2::varchar, c.lead_stage, c.lead_stage, $3::varchar,
                       'eom_office', $4::varchar, $5::text,
                       jsonb_build_object(
                           'calendar_id', $6::text,
                           'calendar_event_id', $7::text,
                           'calendar_error', $8::text,
                           'calendar_message', $5::text
                       )
                FROM contacts AS c
                WHERE c.id = $1::uuid
                  AND EXISTS (
                      SELECT 1
                      FROM eom_lead_lifecycle_events AS requested
                      WHERE requested.contact_id = c.id
                        AND requested.event_type = $9::varchar
                        AND requested.operation_key = $4::varchar
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM eom_lead_lifecycle_events AS completed
                      WHERE completed.contact_id = c.id
                        AND completed.event_type = $10::varchar
                        AND completed.operation_key = $4::varchar
                  )
                ON CONFLICT (contact_id, event_type, operation_key)
                    WHERE operation_key IS NOT NULL
                    DO NOTHING
                """,
                contact_id,
                _EOM_WON_LOSS_CANCELLATION_UNSETTLED_EVENT,
                f"employee:{actor_id}:{actor_name}",
                operation_key,
                calendar_message,
                calendar_id,
                calendar_event_id,
                calendar_error,
                _EOM_WON_LOSS_CANCELLATION_REQUESTED_EVENT,
                _EOM_WON_LOSS_CANCELLATION_COMPLETED_EVENT,
            )

    async def complete_eom_won_lead_loss(
        self,
        *,
        contact_id: str,
        reason_code: str,
        note: str | None,
        operation_key: str,
        calendar_id: str,
        calendar_event_id: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Atomically revoke a prepared draft and commit won -> lost.

        This method is called only after Calendar returned a determinate DELETE
        result.  It re-reads every fact prepared before that external call so a
        stale/reused operation key cannot turn a different lead or draft into a
        loss.
        """
        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        def _result(*, idempotent: bool, reason_code_value: str) -> dict[str, Any]:
            return {
                "contact_id": str(contact_id),
                "lead_stage": "lost",
                "status": "lost",
                "reason_code": reason_code_value,
                "from_stage": "won",
                "idempotent": idempotent,
            }

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            for lock_key in sorted(
                {
                    f"eom-lead-lost:contact:{contact_id}",
                    f"eom-lead-lost:operation:{operation_key}",
                }
            ):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )

            contact = await conn.fetchrow(
                """
                SELECT id, business_context_id, contact_type, lead_stage, status
                FROM contacts
                WHERE id = $1::uuid
                FOR UPDATE
                """,
                contact_id,
            )
            if (
                contact is None
                or contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
            ):
                raise EOMLeadConversionError(404, "EOM lead was not found")

            events = await conn.fetch(
                """
                SELECT event_type, from_stage, to_stage, reason, metadata
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1::uuid
                  AND operation_key = $2
                  AND event_type = ANY($3::varchar[])
                FOR UPDATE
                """,
                contact_id,
                operation_key,
                list(_EOM_WON_LOSS_EVENT_TYPES),
            )
            events_by_type = {str(event["event_type"]): event for event in events}
            requested = events_by_type.get(_EOM_WON_LOSS_CANCELLATION_REQUESTED_EVENT)
            completed = events_by_type.get(_EOM_WON_LOSS_CANCELLATION_COMPLETED_EVENT)
            lost = events_by_type.get("lead_lost")
            if completed is not None or lost is not None:
                if (
                    completed is None
                    or lost is None
                    or lost["from_stage"] != "won"
                    or lost["to_stage"] != "lost"
                    or contact["contact_type"] != "lead"
                    or contact["lead_stage"] != "lost"
                ):
                    raise EOMLeadConversionError(
                        409, "EOM won lead loss operation was superseded"
                    )
                lost_metadata = self._eom_estimate_booking_metadata_from_row(
                    lost["metadata"]
                )
                return _result(
                    idempotent=True,
                    reason_code_value=str(
                        lost_metadata.get("lost_reason_code") or reason_code
                    ),
                )
            if requested is None:
                raise EOMLeadConversionError(
                    409, "EOM won lead loss was not prepared for this operation"
                )
            requested_metadata = self._eom_estimate_booking_metadata_from_row(
                requested["metadata"]
            )
            # An idempotent retry must retain the operator's original note in
            # the ledger instead of allowing a later request body to rewrite
            # the explanation attached to this prepared cancellation.
            effective_note = requested["reason"]
            prepared_calendar_id = str(
                requested_metadata.get("calendar_id") or ""
            ).strip()
            prepared_event_id = str(
                requested_metadata.get("calendar_event_id") or ""
            ).strip()
            booking_key = str(
                requested_metadata.get("booking_operation_key") or ""
            ).strip()
            draft_id = str(
                requested_metadata.get("onboarding_draft_id") or ""
            ).strip()
            if (
                not prepared_calendar_id
                or not prepared_event_id
                or not booking_key
                or not draft_id
                or prepared_calendar_id != calendar_id
                or prepared_event_id != calendar_event_id
            ):
                raise EOMLeadConversionError(
                    409,
                    "EOM won lead loss completion does not match prepared "
                    "Calendar facts",
                )
            if (
                contact["contact_type"] != "lead"
                or contact["lead_stage"] != "won"
                or contact["status"] != "active"
            ):
                raise EOMLeadConversionError(
                    409, "EOM won lead changed before Calendar cancellation completed"
                )

            booked = await conn.fetchrow(
                """
                SELECT metadata
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1::uuid
                  AND event_type = 'first_clean_booked'
                  AND operation_key = $2
                FOR UPDATE
                """,
                contact_id,
                booking_key,
            )
            booked_metadata = (
                self._eom_estimate_booking_metadata_from_row(booked["metadata"])
                if booked is not None
                else {}
            )
            if (
                str(booked_metadata.get("calendar_id") or "").strip()
                != prepared_calendar_id
                or str(booked_metadata.get("calendar_event_id") or "").strip()
                != prepared_event_id
                or str(booked_metadata.get("expected_calendar_event_id") or "").strip()
                != prepared_event_id
            ):
                raise EOMLeadConversionError(
                    409,
                    "EOM first-clean booking no longer matches cancellation facts",
                )

            draft = await conn.fetchrow(
                """
                SELECT id, contact_id, operation_key, status
                FROM eom_onboarding_email_drafts
                WHERE id = $1::uuid
                  AND contact_id = $2::uuid
                  AND operation_key = $3
                FOR UPDATE
                """,
                draft_id,
                contact_id,
                booking_key,
            )
            if draft is None:
                raise EOMLeadConversionError(
                    409, "EOM won lead onboarding draft no longer matches booking"
                )
            draft_status = str(draft["status"])
            if draft_status not in ("pending", "revoked"):
                raise EOMLeadConversionError(
                    409,
                    "EOM onboarding draft delivery must be reconciled before "
                    "marking this won lead lost",
                )
            if await conn.fetchval(
                "SELECT to_regclass('eom_public_onboarding_tokens') IS NOT NULL"
            ):
                issued_public_link = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM eom_public_onboarding_tokens
                        WHERE draft_id = $1::uuid
                          AND contact_id = $2::uuid
                          AND status = 'issued'
                    )
                    """,
                    draft_id,
                    contact_id,
                )
                if issued_public_link:
                    raise EOMLeadConversionError(
                        409,
                        "EOM onboarding draft has an issued public onboarding link",
                    )
            if draft_status == "pending":
                revoked = await conn.fetchrow(
                    """
                    UPDATE eom_onboarding_email_drafts
                    SET status = 'revoked', revoked_at = NOW()
                    WHERE id = $1::uuid
                      AND status = 'pending'
                    RETURNING id
                    """,
                    draft_id,
                )
                if revoked is None:
                    raise RuntimeError(
                        "EOM onboarding draft changed during won lead loss"
                    )

            updated = await conn.fetchrow(
                """
                UPDATE contacts
                SET lead_stage = 'lost', updated_at = NOW()
                WHERE id = $1::uuid
                  AND business_context_id = $2
                  AND contact_type = 'lead'
                  AND lead_stage = 'won'
                  AND status = 'active'
                RETURNING id
                """,
                contact_id,
                EOM_BUSINESS_CONTEXT_ID,
            )
            if updated is None:
                raise RuntimeError("EOM won lead changed during mark-lost completion")
            cancellation_metadata = {
                "booking_operation_key": booking_key,
                "calendar_id": prepared_calendar_id,
                "calendar_event_id": prepared_event_id,
                "onboarding_draft_id": draft_id,
            }
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, reason, metadata
                )
                VALUES ($1::uuid, $2::varchar, 'won', 'lost', $3::varchar,
                        'eom_office', $4::varchar, $5::text, $6::jsonb)
                """,
                contact_id,
                _EOM_WON_LOSS_CANCELLATION_COMPLETED_EVENT,
                f"employee:{actor_id}:{actor_name}",
                operation_key,
                effective_note,
                json.dumps(cancellation_metadata),
            )
            effective_reason_code = str(
                requested_metadata.get("lost_reason_code") or reason_code
            )
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, reason, metadata
                )
                VALUES ($1::uuid, 'lead_lost', 'won', 'lost', $2::varchar,
                        'eom_office', $3::varchar, $4::text,
                        jsonb_build_object(
                            'lost_reason_code', $5::text,
                            'lost_by_employee_id', $6::bigint,
                            'first_clean_cancellation_operation_key', $7::text
                        ))
                """,
                contact_id,
                f"employee:{actor_id}:{actor_name}",
                operation_key,
                effective_note,
                effective_reason_code,
                actor_id,
                operation_key,
            )
            return _result(
                idempotent=False, reason_code_value=effective_reason_code
            )

    async def mark_eom_lead_lost(
        self,
        *,
        contact_id: str,
        reason_code: str,
        note: str | None,
        operation_key: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Atomically mark one EOM lead lost, recording a reason on the ledger.

        Reversible via reopen_eom_lead; no customer/site or calendar side
        effect. Fences an in-flight booking the same way the customer handoff
        does, so a lead cannot be marked lost while a calendar call is still
        outstanding (which would otherwise land an event on a lost lead).
        """
        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        # 'won' stays deliberately excluded from this direct writer. A won lead
        # owns a booked first clean and an onboarding draft, so only
        # eom_won_lead_loss may admit it after durable Calendar cancellation
        # preparation. Keeping this guard prevents future internal callers from
        # bypassing that external-side-effect protocol.
        admission = _EOM_LOST_RESTORABLE_STAGES

        def _result(
            from_stage: str,
            *,
            idempotent: bool,
            reason_code_value: str | None = None,
        ) -> dict[str, Any]:
            return {
                "contact_id": str(contact_id),
                "lead_stage": "lost",
                "status": "lost",
                "reason_code": reason_code_value or reason_code,
                "from_stage": from_stage,
                "idempotent": idempotent,
            }

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            for lock_key in sorted(
                {
                    f"eom-lead-lost:contact:{contact_id}",
                    f"eom-lead-lost:operation:{operation_key}",
                }
            ):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            replay = await conn.fetchrow(
                """
                SELECT
                    id,
                    from_stage,
                    to_stage,
                    lifecycle_sequence,
                    metadata->>'lost_reason_code' AS reason_code
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = 'lead_lost'
                  AND operation_key = $2
                """,
                contact_id,
                operation_key,
            )
            # An Idempotency-Key belongs to exactly one lead: the same key on a
            # different contact is a client error, not a second lost lead. The
            # booking/handoff paths reject the same reuse.
            foreign_key_owner = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM eom_lead_lifecycle_events
                    WHERE operation_key = $1
                      AND event_type = 'lead_lost'
                      AND contact_id <> $2
                )
                """,
                operation_key,
                contact_id,
            )
            if foreign_key_owner:
                raise EOMLeadConversionError(
                    409, "Idempotency-Key already belongs to another EOM lead"
                )
            contact = await conn.fetchrow(
                """
                SELECT id, business_context_id, contact_type, lead_stage, status
                FROM contacts
                WHERE id = $1
                FOR UPDATE
                """,
                contact_id,
            )
            if (
                contact is None
                or contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
            ):
                raise EOMLeadConversionError(404, "EOM lead was not found")
            if replay is not None:
                if (
                    replay["from_stage"] not in _EOM_LOST_RESTORABLE_STAGES
                    or replay["to_stage"] != "lost"
                ):
                    raise EOMLeadConversionError(
                        409, "EOM lead lost operation was superseded"
                    )
                # A replay is only truthfully idempotent while the lead is still
                # lost. If it was reopened after this key, reporting "lost"
                # would assert a stage the row no longer has.
                if (
                    contact["contact_type"] != "lead"
                    or contact["lead_stage"] != "lost"
                ):
                    raise EOMLeadConversionError(
                        409, "EOM lead was reopened after this operation"
                    )
                if await _eom_disposition_replay_was_superseded(
                    conn,
                    contact_id=contact_id,
                    replay_event_type="lead_lost",
                    replay_event_id=replay["id"],
                    replay_lifecycle_sequence=replay["lifecycle_sequence"],
                ):
                    raise EOMLeadConversionError(
                        409, "EOM lead lost operation was superseded"
                    )
                return _result(
                    str(replay["from_stage"]),
                    idempotent=True,
                    reason_code_value=replay["reason_code"],
                )
            if contact["contact_type"] == "lead" and contact["lead_stage"] == "lost":
                # Already lost under a *different* key (this key has no replay
                # row). Reject rather than a keyless no-op: a 200 here would
                # report this operation_key successful with nothing durable
                # behind it, so a later reopen+retry would re-apply it.
                raise EOMLeadConversionError(409, "EOM lead is already lost")
            if contact["contact_type"] != "lead":
                raise EOMLeadConversionError(409, "EOM contact is not a lead")
            if contact["status"] != "active":
                raise EOMLeadConversionError(
                    409, "EOM lead must be active to mark lost"
                )
            if contact["lead_stage"] not in admission:
                raise EOMLeadConversionError(
                    409, "EOM lead is not in a stage that can be marked lost"
                )
            booking_events = await conn.fetch(
                """
                SELECT event_type, operation_key
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = ANY($2::varchar[])
                FOR UPDATE
                """,
                contact_id,
                list(_ALL_EOM_BOOKING_EVENT_TYPES),
            )
            booking_event_types: dict[str, set[str]] = {}
            for event in booking_events:
                booking_event_types.setdefault(event["operation_key"], set()).add(
                    event["event_type"]
                )
            for op_key, event_types in booking_event_types.items():
                if event_types & _ALL_EOM_BOOKED_EVENTS:
                    continue
                settled = bool(
                    await conn.fetchval(
                        "SELECT pg_try_advisory_xact_lock(hashtextextended($1, 0))",
                        f"eom-estimate-booking:execution:{op_key}",
                    )
                )
                if not settled:
                    raise EOMLeadConversionError(
                        409,
                        "EOM booking is still executing; retry after it settles",
                    )
            # The execution lock only proves no attempt is running right now; a
            # requested/ambiguous booking whose executor died mid-flight holds
            # no lock yet is unreconciled. Mirror the handoff fence and refuse
            # to lose a lead with an outstanding or unreconciled calendar event.
            blocking_booking = next(
                (
                    event_types
                    for event_types in booking_event_types.values()
                    if (
                        bool(event_types & _ALL_EOM_AMBIGUOUS_EVENTS)
                        and not (event_types & _ALL_EOM_BOOKED_EVENTS)
                    )
                    or (
                        bool(event_types & _ALL_EOM_REQUESTED_EVENTS)
                        and not self._eom_estimate_booking_operation_is_terminal(
                            event_types
                        )
                    )
                ),
                None,
            )
            if blocking_booking is not None:
                if blocking_booking & _ALL_EOM_AMBIGUOUS_EVENTS:
                    raise EOMLeadConversionError(
                        409,
                        "EOM booking requires calendar reconciliation",
                    )
                raise EOMLeadConversionError(
                    409,
                    "EOM booking is still pending calendar completion",
                )
            from_stage = str(contact["lead_stage"])
            updated = await conn.fetchrow(
                """
                UPDATE contacts
                SET lead_stage = 'lost', updated_at = NOW()
                WHERE id = $1
                  AND business_context_id = $2
                  AND contact_type = 'lead'
                  AND lead_stage = $3
                  AND status = 'active'
                RETURNING id
                """,
                contact_id,
                EOM_BUSINESS_CONTEXT_ID,
                from_stage,
            )
            if updated is None:
                raise RuntimeError("EOM lead changed during mark-lost")
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, reason, metadata
                )
                VALUES ($1, 'lead_lost', $2, 'lost', $3, 'eom_office', $4, $5,
                        jsonb_build_object(
                            'lost_reason_code', $6::text,
                            'lost_by_employee_id', $7::bigint
                        ))
                """,
                contact_id,
                from_stage,
                f"employee:{actor_id}:{actor_name}",
                operation_key,
                note,
                reason_code,
                actor_id,
            )
            return _result(from_stage, idempotent=False)

    async def reopen_eom_lead(
        self,
        *,
        contact_id: str,
        operation_key: str,
        actor_id: int,
        actor_name: str,
    ) -> dict[str, Any]:
        """Return a previously-lost EOM lead to its pre-loss active stage."""
        from .eom_lead_conversion import EOMLeadConversionError
        from .eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        def _result(*, lead_stage: str, idempotent: bool) -> dict[str, Any]:
            return {
                "contact_id": str(contact_id),
                "lead_stage": lead_stage,
                "status": "active",
                "idempotent": idempotent,
            }

        pool = self._get_pool()
        async with _transaction_connection(pool) as conn:
            for lock_key in sorted(
                {
                    f"eom-lead-lost:contact:{contact_id}",
                    f"eom-lead-reopen:operation:{operation_key}",
                }
            ):
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    lock_key,
                )
            replay = await conn.fetchrow(
                """
                SELECT id, from_stage, to_stage, lifecycle_sequence
                FROM eom_lead_lifecycle_events
                WHERE contact_id = $1
                  AND event_type = 'lead_reopened'
                  AND operation_key = $2
                """,
                contact_id,
                operation_key,
            )
            foreign_key_owner = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM eom_lead_lifecycle_events
                    WHERE operation_key = $1
                      AND event_type = 'lead_reopened'
                      AND contact_id <> $2
                )
                """,
                operation_key,
                contact_id,
            )
            if foreign_key_owner:
                raise EOMLeadConversionError(
                    409, "Idempotency-Key already belongs to another EOM lead"
                )
            contact = await conn.fetchrow(
                """
                SELECT id, business_context_id, contact_type, lead_stage, status
                FROM contacts
                WHERE id = $1
                FOR UPDATE
                """,
                contact_id,
            )
            if (
                contact is None
                or contact["business_context_id"] != EOM_BUSINESS_CONTEXT_ID
            ):
                raise EOMLeadConversionError(404, "EOM lead was not found")
            if replay is not None:
                replay_stage = str(replay["to_stage"])
                if (
                    replay["from_stage"] != "lost"
                    or replay_stage not in _EOM_LOST_RESTORABLE_STAGES
                ):
                    raise EOMLeadConversionError(
                        409, "EOM lead reopen operation was superseded"
                    )
                # A replay is only truthfully idempotent while the row is still
                # the active lead at the stage this key restored. If it was
                # lost again, finalized to a customer, or archived, reporting
                # active would assert a state the row no longer has.
                if not (
                    contact["contact_type"] == "lead"
                    and contact["lead_stage"] == replay_stage
                    and contact["status"] == "active"
                ):
                    raise EOMLeadConversionError(
                        409, "EOM lead changed after this reopen"
                    )
                if await _eom_disposition_replay_was_superseded(
                    conn,
                    contact_id=contact_id,
                    replay_event_type="lead_reopened",
                    replay_event_id=replay["id"],
                    replay_lifecycle_sequence=replay["lifecycle_sequence"],
                ):
                    raise EOMLeadConversionError(
                        409, "EOM lead reopen operation was superseded"
                    )
                return _result(lead_stage=replay_stage, idempotent=True)
            # Not a replay of this key: the lead must currently be lost. An
            # already-active lead reached under a *different* key is a conflict,
            # not a keyless no-op, so no operation_key is reported successful
            # without a durable replay row behind it.
            if contact["contact_type"] != "lead" or contact["lead_stage"] != "lost":
                raise EOMLeadConversionError(
                    409, "EOM lead is not lost and cannot be reopened"
                )
            if contact["status"] != "active":
                # Reopen promises the lead returns to the active review queue;
                # flipping only lead_stage on an archived/inactive contact would
                # report status 'active' while the row stays out of the queue.
                raise EOMLeadConversionError(
                    409, "EOM lead must be active to reopen"
                )
            latest_loss = await conn.fetchrow(
                """
                SELECT from_stage, legacy_loss_count, sequenced_loss_count
                FROM (
                    SELECT
                        from_stage,
                        lifecycle_sequence,
                        occurred_at,
                        created_at,
                        COUNT(*) FILTER (
                            WHERE lifecycle_sequence IS NULL
                        ) OVER () AS legacy_loss_count,
                        COUNT(*) FILTER (
                            WHERE lifecycle_sequence IS NOT NULL
                        ) OVER () AS sequenced_loss_count
                    FROM eom_lead_lifecycle_events
                    WHERE contact_id = $1
                      AND event_type = 'lead_lost'
                ) AS lead_lost
                ORDER BY lifecycle_sequence DESC NULLS LAST,
                         occurred_at DESC,
                         created_at DESC
                LIMIT 1
                """,
                contact_id,
            )
            if latest_loss is None:
                raise EOMLeadConversionError(
                    409, "EOM lead has no lost-stage evidence to reopen"
                )
            if (
                int(latest_loss["sequenced_loss_count"] or 0) == 0
                and int(latest_loss["legacy_loss_count"] or 0) > 1
            ):
                # More than one pre-migration loss row has no safe chronology;
                # a single legacy row is unambiguous, and sequenced rows were
                # written after migration 363's database default landed.
                raise EOMLeadConversionError(
                    409,
                    "EOM lead lost-stage evidence requires chronology reconciliation",
                )
            restored_stage = str(latest_loss["from_stage"] or "")
            if restored_stage not in _EOM_LOST_RESTORABLE_STAGES:
                raise EOMLeadConversionError(
                    409,
                    "EOM lead lost-stage evidence cannot be safely restored",
                )
            updated = await conn.fetchrow(
                """
                UPDATE contacts
                SET lead_stage = $3, updated_at = NOW()
                WHERE id = $1
                  AND business_context_id = $2
                  AND contact_type = 'lead'
                  AND lead_stage = 'lost'
                  AND status = 'active'
                RETURNING id
                """,
                contact_id,
                EOM_BUSINESS_CONTEXT_ID,
                restored_stage,
            )
            if updated is None:
                raise RuntimeError("EOM lead changed during reopen")
            await conn.execute(
                """
                INSERT INTO eom_lead_lifecycle_events (
                    contact_id, event_type, from_stage, to_stage, actor,
                    source, operation_key, metadata
                )
                VALUES ($1, 'lead_reopened', 'lost', $2, $3, 'eom_office', $4,
                        jsonb_build_object('reopened_by_employee_id', $5::bigint))
                """,
                contact_id,
                restored_stage,
                f"employee:{actor_id}:{actor_name}",
                operation_key,
                actor_id,
            )
            return _result(lead_stage=restored_stage, idempotent=False)


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
